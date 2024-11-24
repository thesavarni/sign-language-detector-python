import os
import pickle
import mediapipe as mp
import cv2
import numpy as np
import re

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3
)

DATA_DIR = './data'

# Get list of class directories sorted by their class number
def get_sorted_class_dirs(data_dir):
    class_dirs = []
    for item in os.listdir(data_dir):
        dir_path = os.path.join(data_dir, item)
        if os.path.isdir(dir_path):
            # Extract class number from folder name
            match = re.match(r'(\d+)_', item)
            if match:
                class_number = int(match.group(1))
                class_dirs.append((class_number, dir_path))
    # Sort by class number
    class_dirs.sort(key=lambda x: x[0])
    return class_dirs

class_dirs = get_sorted_class_dirs(DATA_DIR)

# Group the class directories into sets of 5
group_size = 5
groups = [class_dirs[i:i+group_size] for i in range(0, len(class_dirs), group_size)]

for idx, group in enumerate(groups):
    data = []
    labels = []
    print(f'Processing group {idx+1}')
    for class_number, class_dir in group:
        class_name = os.path.basename(class_dir).split('_', 1)[1]
        print(f'  Processing class "{class_name}"')
        for img_name in os.listdir(class_dir):
            data_aux = []
            x_ = []
            y_ = []
            z_ = []

            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
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
                # [(x1,y1,z1), (x2,y2,z2), ]
                data.append(data_aux)
                labels.append(class_name)
            else:
                print(f'    No hand landmarks detected in image: {img_name}')

    # Save the processed data for this group
    pickle_file = f'data_group_{idx+1}.pickle'
    with open(pickle_file, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print(f"  Saved group {idx+1} data to '{pickle_file}'.\n")

print("Data processing complete for all groups.")
# for 1 image
# [(22 landmarks ka list)] -> A (fist image) -> 200 more images all of A
# # [(22 landmarks ka list)] -> A (fist image) -> 200 more images all of B