# Step 2: Data Preprocessing and Augmentation
import pickle
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the dataset
with open('./asl_dataset.pickle', 'rb') as f:
    dataset = pickle.load(f)

# Extract data
image_paths = [item['image_path'] for item in dataset]
landmarks = [item['landmarks'] for item in dataset]
labels = [item['label'] for item in dataset]

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)
print(f"Number of classes: {num_classes}")

# Save the label encoder for future use
with open('label_encoder.pickle', 'wb') as f:
    pickle.dump(label_encoder, f)

# Load images and preprocess
images = []
for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        print(f"Failed to read {path}")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))  # Resize images as needed
    images.append(img)

images = np.array(images)
landmarks = np.array(landmarks)
labels_encoded = np.array(labels_encoded)

# Normalize images
images = images.astype('float32') / 255.0

# Split into train and test sets
X_train_img, X_test_img, X_train_lm, X_test_lm, y_train, y_test = train_test_split(
    images, landmarks, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
)

# Convert labels to categorical (one-hot encoding)
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Data augmentation for images
datagen = ImageDataGenerator(
    rotation_range=10,  # Small rotations
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8,1.2],
    zoom_range=0.1
)

# Fit the data generator to training images
datagen.fit(X_train_img)

print("Data preparation complete.")
