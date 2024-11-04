import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize Hands with appropriate parameters
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize variables for sentence construction
current_sentence = ""
gesture_start_time = None
gesture_threshold = 0.5  # seconds to confirm gesture
last_gesture = None
gesture_added = False

# Variables to handle model switching
current_group_number = 0
model = None
label_encoder = None

# Function to load the model based on group number
def load_model(group_number):
    global model, label_encoder
    model_file = f'model_group_{group_number}.p'
    try:
        with open(model_file, 'rb') as f:
            model_dict = pickle.load(f)
        model = model_dict['model']
        label_encoder = model_dict['label_encoder']
        print(f"Loaded model for group {group_number}.")
    except FileNotFoundError:
        print(f"Model file '{model_file}' not found.")
        model = None
        label_encoder = None

# Display initial instructions
print("Press number keys corresponding to the group number to load the model.")
print("Groups correspond to sets of 5 classes (e.g., Group 1: A-E, Group 2: F-J, etc.)")

while True:
    data_aux = []
    x_ = []
    y_ = []
    z_ = []
    current_time = time.time()

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    H, W, _ = frame.shape

    # Convert the frame to RGB as Mediapipe uses RGB images
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(frame_rgb)

    detected_gesture = None

    if results.multi_hand_landmarks and model is not None:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw hand landmarks on the frame
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # Extract x, y, z coordinates for each landmark
        for lm in hand_landmarks.landmark:
            x = lm.x
            y = lm.y
            z = lm.z
            x_.append(x)
            y_.append(y)
            z_.append(z)

        # Normalize landmarks by subtracting the minimum x, y, z
        min_x = min(x_)
        min_y = min(y_)
        min_z = min(z_)
        for x_val, y_val, z_val in zip(x_, y_, z_):
            data_aux.append(x_val - min_x)
            data_aux.append(y_val - min_y)
            data_aux.append(z_val - min_z)

        # Ensure the feature vector has 63 features (21 landmarks * 3 coordinates)
        if len(data_aux) != 63:
            print(f"Unexpected feature vector size: {len(data_aux)}. Expected 63.")
            detected_gesture = None
            continue

        # Predict the gesture
        X_input = np.array(data_aux).reshape(1, -1)
        y_pred = model.predict(X_input)
        predicted_class = label_encoder.inverse_transform(y_pred)[0]

        detected_gesture = predicted_class
    else:
        # If no hands are detected or model is not loaded, set detected_gesture to None
        detected_gesture = None

    # Gesture stability detection
    if detected_gesture != last_gesture:
        # Gesture has changed
        if detected_gesture is not None:
            gesture_start_time = current_time
            gesture_added = False
        last_gesture = detected_gesture
    else:
        # Gesture is the same as before
        if detected_gesture is not None and not gesture_added:
            if (current_time - gesture_start_time) >= gesture_threshold:
                # Add gesture to sentence
                current_sentence += detected_gesture
                gesture_added = True

    # Handle keyboard inputs for model switching, space, and clearing the sentence
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in [ord(str(i)) for i in range(1, 7)]:
        # Load model corresponding to the pressed number key
        group_number = key - ord('0')  # Convert key code to integer
        print("group number ", group_number)
        if group_number != current_group_number:
            load_model(group_number)
            current_group_number = group_number
    elif key == ord(' '):
        # Add a space to the sentence
        current_sentence += ' '
    elif key == ord('c'):
        # Clear the current sentence
        current_sentence = ""
        print("Sentence cleared.")

    # Display the prediction on the frame
    if detected_gesture is not None:
        # Calculate bounding box coordinates
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        # Draw bounding box and prediction on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, detected_gesture, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        if model is None:
            cv2.putText(frame, 'No model loaded.', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            # If no hands are detected, display a message
            cv2.putText(frame, 'Model Loaded, No hands detected', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the current sentence at the bottom of the frame
    cv2.putText(frame, f'Sentence: {current_sentence}', (10, H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Ensure that gesture_added flag resets when gesture changes
    if detected_gesture != last_gesture:
        gesture_added = False

# Release resources
cap.release()
cv2.destroyAllWindows()
