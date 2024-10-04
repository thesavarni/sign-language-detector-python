import os
import numpy as np
import cv2
from tensorflow.keras import models, layers, utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle


PROCESSED_DATA_DIR = './processed_data'

# Load images and labels
images = []
labels = []

for dir_ in os.listdir(PROCESSED_DATA_DIR):
    class_dir = os.path.join(PROCESSED_DATA_DIR, dir_)
    if not os.path.isdir(class_dir):
        continue

    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        images.append(img)
        labels.append(dir_)

# Convert to numpy arrays
images = np.array(images)
labels = np.array(labels)

print(f'Total images: {images.shape[0]}')
print(f'Image shape: {images.shape[1:]}')

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

# Normalize images
images = images.astype('float32') / 255.0

# Convert labels to categorical
labels_categorical = utils.to_categorical(labels_encoded, num_classes)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    images, labels_categorical, test_size=0.2, random_state=42, stratify=labels_encoded
)

# Define the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Data Augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

datagen.fit(x_train)

# Train the model
batch_size = 32
epochs = 30

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(x_train) // batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test)
)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Model accuracy on test set: {accuracy * 100:.2f}%")

# Save the model and label encoder
model.save('gesture_recognition_cnn.h5')
with open('label_encoder_cnn.p', 'wb') as f:
    pickle.dump(label_encoder, f)

print("Model training complete. Saved to 'gesture_recognition_cnn.h5'.")
