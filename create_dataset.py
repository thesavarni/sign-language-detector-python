import os
import pickle
import mediapipe as mp
import cv2
import numpy as np
import re

# Initialize Mediapipe Hands for ASL and ISL
mp_hands = mp.solutions.hands

# Data directories for ASL and ISL
DATA_DIRS = {
    'asl': './data/asl',
    'isl': './data/isl'
}

# Function to get sorted class directories
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

for language, data_dir in DATA_DIRS.items():
    print(f"Processing language: {language.upper()}")
    class_dirs = get_sorted_class_dirs(data_dir)

    # Group the class directories into sets of 5
    group_size = 5
    groups = [class_dirs[i:i+group_size] for i in range(0, len(class_dirs), group_size)]

    # Set max_num_hands based on language
    if language == 'asl':
        continue
        max_hands = 1
    elif language == 'isl':
        max_hands = 2
    else:
        continue  # Skip unknown languages

    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=max_hands,
        min_detection_confidence=0.3
    )

    for idx, group in enumerate(groups):
        data = []
        labels = []
        print(f'  Processing group {idx+1}')
        for class_number, class_dir in group:
            class_name = os.path.basename(class_dir).split('_', 1)[1]
            print(f'    Processing class "{class_name}"')
            for img_name in os.listdir(class_dir):
                data_aux = []
                x_all = []
                y_all = []
                z_all = []

                img_path = os.path.join(class_dir, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                results = hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        x_ = []
                        y_ = []
                        z_ = []
                        for lm in hand_landmarks.landmark:
                            x_.append(lm.x)
                            y_.append(lm.y)
                            z_.append(lm.z)
                        x_all.extend(x_)
                        y_all.extend(y_)
                        z_all.extend(z_)
                    # Normalize landmarks by subtracting the minimum value
                    if x_all and y_all and z_all:
                        min_x = min(x_all)
                        min_y = min(y_all)
                        min_z = min(z_all)

                        for x_val, y_val, z_val in zip(x_all, y_all, z_all):
                            data_aux.extend([x_val - min_x, y_val - min_y, z_val - min_z])

                        expected_length = 63 * max_hands  # 63 features per hand
                        if len(data_aux) != expected_length:
                            print(f" length {len(data_aux)} hands {max_hands}:    Skipping image {img_name} due to incorrect data length.")
                            continue
                        data.append(data_aux)
                        labels.append(class_name)
                    else:
                        print(f"      No landmarks extracted for image: {img_name}")
                else:
                    print(f'      No hand landmarks detected in image: {img_name}')

        # Save the processed data for this group
        output_dir = f'./{language}_dataset_pickle'
        os.makedirs(output_dir, exist_ok=True)
        pickle_file = os.path.join(output_dir, f'data_group_{idx+1}.pickle')
        with open(pickle_file, 'wb') as f:
            pickle.dump({'data': data, 'labels': labels}, f)
        print(f"    Saved group {idx+1} data to '{pickle_file}'.\n")

    print(f"Data processing complete for {language.upper()}.\n")

print("Data processing complete for all languages.")
