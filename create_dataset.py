import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3
)

DATA_DIR = './data'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(class_dir):
        continue

    print(f'Processing class "{dir_}"')
    for img_path in os.listdir(class_dir):
        data_aux = []
        x_ = []
        y_ = []
        z_ = []

        img = cv2.imread(os.path.join(class_dir, img_path))
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            # Extract x, y, z coordinates
            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)
                z_.append(lm.z)

            # Normalize landmarks by subtracting the minimum value
            min_x = min(x_)
            min_y = min(y_)
            min_z = min(z_)

            for x_val, y_val, z_val in zip(x_, y_, z_):
                data_aux.append(x_val - min_x)
                data_aux.append(y_val - min_y)
                data_aux.append(z_val - min_z)

            data.append(data_aux)
            labels.append(dir_)
        else:
            print(f'No hand landmarks detected in image: {img_path}')

# Save the processed data
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data processing complete. Saved to 'data.pickle'.")
