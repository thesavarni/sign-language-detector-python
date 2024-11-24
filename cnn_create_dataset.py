import os
import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.3
)

DATA_DIR = './data'
PROCESSED_DATA_DIR = './processed_data'

if not os.path.exists(PROCESSED_DATA_DIR):
    os.makedirs(PROCESSED_DATA_DIR)

for dir_ in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, dir_)
    processed_class_dir = os.path.join(PROCESSED_DATA_DIR, dir_)

    if not os.path.isdir(class_dir):
        continue

    if not os.path.exists(processed_class_dir):
        os.makedirs(processed_class_dir)

    print(f'Processing class "{dir_}"')

    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W, _ = img.shape

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Get bounding box coordinates
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * W)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * H)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * W)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * H)

            # Add padding
            margin = 20
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(W, x_max + margin)
            y_max = min(H, y_max + margin)

            # Crop the image
            cropped_img = img[y_min:y_max, x_min:x_max]

            # Resize the image to a fixed size (e.g., 128x128)
            resized_img = cv2.resize(cropped_img, (128, 128))

            # Save the processed image
            processed_img_path = os.path.join(processed_class_dir, img_name)
            cv2.imwrite(processed_img_path, resized_img)
        else:
            print(f'No hand detected in image: {img_name}')
