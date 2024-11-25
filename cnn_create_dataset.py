import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
from tqdm import tqdm

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# Paths
DATA_DIR = './data'  # Directory where your images are stored
OUTPUT_DATASET = './asl_dataset.pickle'  # Output dataset file

# Get list of class names
class_names = sorted(os.listdir(DATA_DIR))
print(f"Found classes: {class_names}")

dataset = []

for class_name in class_names:
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue

    print(f"Processing class '{class_name}'...")
    image_names = os.listdir(class_dir)
    for image_name in tqdm(image_names):
        image_path = os.path.join(class_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            # Extract landmark coordinates
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            dataset.append({
                'image_path': image_path,
                'landmarks': landmarks,
                'label': class_name
            })
        else:
            # If no landmarks are detected, you may choose to exclude the image
            print(f"No hand detected in {image_path}, skipping.")

# Save the dataset
with open(OUTPUT_DATASET, 'wb') as f:
    pickle.dump(dataset, f)

print(f"Dataset saved to {OUTPUT_DATASET}")