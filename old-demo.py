import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import string
import warnings
warnings.filterwarnings("ignore")


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

current_sentence = ""
gesture_start_time = None
gesture_threshold = 0.5
last_gesture = None
gesture_added = False

current_group_number = None
model = None
label_encoder = None

# Initialize key-to-gesture mapping for alphabets only
key_to_gesture = {}
for i in range(26):
    key = string.ascii_lowercase[i]
    gesture = key.upper()
    key_to_gesture[key] = gesture

manual_override_gesture = None  # Initialize manual override variable

# Function to load the model based on group number
def load_model(group_number):
    global model, label_encoder
    model_file = f'./asl_model/model_group_{group_number}.p'
    try:
        with open(model_file, 'rb') as f:
            model_dict = pickle.load(f)
        model = model_dict['model']
        label_encoder = model_dict['label_encoder']
        print("Model loaded for group", group_number)
    except FileNotFoundError:
        model = None
        label_encoder = None

print("=== Hand Gesture Recognition Inference ===")
print("Press 'Esc' to quit the application.")
print("Press 'c' to clear the current sentence.")
print("Press 'Space' to add a space to the sentence.")
load_model(1)
while True:
    data_aux = []
    x_ = []
    y_ = []
    z_ = []
    current_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    predicted_gesture = None

    if results.multi_hand_landmarks and model is not None:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        for lm in hand_landmarks.landmark:
            x_.append(lm.x)
            y_.append(lm.y)
            z_.append(lm.z)

        if x_ and y_ and z_:
            min_x = min(x_)
            min_y = min(y_)
            min_z = min(z_)
            for x_val, y_val, z_val in zip(x_, y_, z_):
                data_aux.extend([x_val - min_x, y_val - min_y, z_val - min_z])

            if len(data_aux) == 63:
                X_input = np.array(data_aux).reshape(1, -1)
                y_pred = model.predict(X_input)
                predicted_class = label_encoder.inverse_transform(y_pred)[0]
                predicted_gesture = predicted_class

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # Escape key
        break
    elif key in [ord(str(i)) for i in range(1, 7)]:
        group_number = key - ord('0')
        if group_number != current_group_number:
            load_model(group_number)
            current_group_number = group_number
    elif key == ord(' '):
        current_sentence += ' '
    elif key in [ord('c'), ord('C')]:
        current_sentence = ""
    elif key == ord('0'):
        manual_override_gesture = None
    else:
        key_char = chr(key).lower()
        if key_char in key_to_gesture:
            manual_override_gesture = key_to_gesture[key_char]

    if manual_override_gesture is not None:
        detected_gesture = manual_override_gesture
    else:
        detected_gesture = predicted_gesture

    if detected_gesture != last_gesture:
        if detected_gesture is not None:
            gesture_start_time = current_time
            gesture_added = False
        last_gesture = detected_gesture
    else:
        if detected_gesture is not None and not gesture_added:
            if (current_time - gesture_start_time) >= gesture_threshold:
                current_sentence += detected_gesture
                gesture_added = True

    if detected_gesture is not None:
        if x_ and y_:
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, detected_gesture, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, detected_gesture, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        if model is None:
            cv2.putText(frame, 'No model loaded. Press 1-6 to load a model.', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'No hands detected', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, f'Sentence: {current_sentence}', (10, H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Hand Gesture Recognition', frame)

cap.release()
cv2.destroyAllWindows()