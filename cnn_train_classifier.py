#Step 3: Design the Multi-Modal Model Architecture

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import numpy as np
import datetime
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, concatenate

# Image branch
image_input = Input(shape=(128, 128, 3))

# Simple CNN architecture
x = Conv2D(32, (3, 3), activation='relu')(image_input)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

# Landmark branch
landmark_input = Input(shape=(63,))  # 21 landmarks * 3 coordinates
y = Dense(128, activation='relu')(landmark_input)
y = Dropout(0.5)(y)
y = Dense(64, activation='relu')(y)
y = Dropout(0.5)(y)

# Concatenate the outputs
combined = concatenate([x, y])

# Final layers
z = Dense(256, activation='relu')(combined)
z = Dropout(0.5)(z)
z = Dense(num_classes, activation='softmax')(z)

# Create the model
model = Model(inputs=[image_input, landmark_input], outputs=z)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Model architecture:")
model.summary()

# If not already done, ensure that your data arrays are numpy arrays and have the correct data types
# Confirm that images are normalized and of type float32
X_train_img = X_train_img.astype('float32')  # Already normalized: images = images.astype('float32') / 255.0
X_test_img = X_test_img.astype('float32')    # Already normalized

# Ensure landmarks are float32
X_train_lm = X_train_lm.astype('float32')
X_test_lm = X_test_lm.astype('float32')

# Ensure labels are float32
y_train_cat = y_train_cat.astype('float32')
y_test_cat = y_test_cat.astype('float32')

# Define the ASLSequence class
class ASLSequence(Sequence):
    def __init__(self, X_img, X_lm, y, batch_size=32, augment=False):
        self.X_img = X_img
        self.X_lm = X_lm
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.indices = np.arange(len(self.y))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.y) / self.batch_size))

    def __getitem__(self, index):
        # Generate indices for the batch
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.y))
        batch_indices = self.indices[start:end]

        # Generate data
        img_batch = self.X_img[batch_indices]
        lm_batch = self.X_lm[batch_indices]
        y_batch = self.y[batch_indices]

        if self.augment:
            # Apply augmentation to images and landmarks
            img_batch, lm_batch = self.__augment_batch(img_batch, lm_batch)

        # Return inputs as a tuple
        return (img_batch, lm_batch), y_batch

    def on_epoch_end(self):
        # Shuffle indices after each epoch
        np.random.shuffle(self.indices)

    def __augment_batch(self, img_batch, lm_batch):
        # Example augmentation: Horizontal flip with 50% probability
        flip_indices = np.random.rand(len(img_batch)) > 0.5
        img_batch_flipped = img_batch[flip_indices]
        lm_batch_flipped = lm_batch[flip_indices]

        # Flip images horizontally
        img_batch_flipped = img_batch_flipped[:, :, ::-1, :]

        # Adjust landmarks (assuming x-coordinates are normalized between 0 and 1)
        lm_batch_flipped[:, 0::3] = 1.0 - lm_batch_flipped[:, 0::3]

        # Replace the original batches with the augmented data
        img_batch[flip_indices] = img_batch_flipped
        lm_batch[flip_indices] = lm_batch_flipped

        # Add more augmentations if needed (e.g., brightness, scaling)
        return img_batch, lm_batch

# Training parameters
batch_size = 32
epochs = 50

# Instantiate the sequences
train_sequence = ASLSequence(
    X_img=X_train_img,
    X_lm=X_train_lm,
    y=y_train_cat,
    batch_size=batch_size,
    augment=True  # Enable augmentation for training data
)

val_sequence = ASLSequence(
    X_img=X_test_img,
    X_lm=X_test_lm,
    y=y_test_cat,
    batch_size=batch_size,
    augment=False  # No augmentation for validation data
)

# Callbacks
checkpoint = ModelCheckpoint(
    './best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    mode='max',
    restore_best_weights=True
)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Verify the data generator (optional but recommended)
# Fetch a single batch to check shapes and types
(inputs, labels) = train_sequence[0]
print("Image batch shape:", inputs[0].shape)    # Expected: (batch_size, 128, 128, 3)
print("Landmark batch shape:", inputs[1].shape) # Expected: (batch_size, 63)
print("Label batch shape:", labels.shape)       # Expected: (batch_size, num_classes)

# Ensure the inputs are tuples
print("Type of inputs:", type(inputs))  # Should be <class 'tuple'>

# Train the model
history = model.fit(
    train_sequence,
    epochs=epochs,
    validation_data=val_sequence,
    callbacks=[checkpoint, tensorboard_callback]
)

# Save the final model
model.save('./asl_model.keras')

print("Training complete.")