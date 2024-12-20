from flask import Flask, request, jsonify
import pickle
import cv2
import mediapipe as mp
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from flask_cors import CORS
import os

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Allow CORS from your frontend
CORS(app, resources={r"/predict": {"origins": "https://mudra-ai.netlify.app"}})

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands

# Load models and label encoders for both ASL and ISL
models = {
    'asl': {},
    'isl': {}
}
label_encoders = {
    'asl': {},
    'isl': {}
}

# Function to load models during startup
def load_all_models():
    for language in ['asl', 'isl']:
        for group_number in range(1, 7):  # Assuming max group number is 6
            model_file = f'./{language}_model/model_group_{group_number}.p'
            try:
                with open(model_file, 'rb') as f:
                    model_dict = pickle.load(f)
                models[language][group_number] = model_dict['model']
                label_encoders[language][group_number] = model_dict['label_encoder']
                print(f"Model loaded for {language.upper()} group {group_number}")
            except FileNotFoundError:
                print(f"Model file not found for {language} group {group_number}")

# Load all models at startup
load_all_models()

# Map signs to model group numbers for ASL and ISL
sign_to_model_group = {
    'asl': {
        'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E': 1,
        'F': 2, 'G': 2, 'H': 2, 'I': 2, 'J': 2,
        'K': 3, 'L': 3, 'M': 3, 'N': 3, 'O': 3,
        'P': 4, 'Q': 4, 'R': 4, 'S': 4, 'T': 4,
        'U': 5, 'V': 5, 'W': 5, 'X': 5, 'Y': 5,
        'Z': 6, 'DOG': 6, 'THANK_YOU': 6, 'LOVE': 6
    },
    'isl': {
        'A': 1, 'B': 1,  'D': 1, 'E': 1, 'F': 1,
        'G': 2, 'H': 2,
    }
}

@app.route('/predict', methods=['POST'])
def predict():
    # Check if image, expected_sign, and language are in the request
    if 'image' not in request.files or 'expected_sign' not in request.form or 'language' not in request.form:
        return jsonify({'error': 'Image, expected_sign, or language not provided'}), 400

    image_file = request.files['image']
    expected_sign = request.form['expected_sign'].upper()
    language = request.form['language'].lower()

    if language not in ['asl', 'isl']:
        return jsonify({'error': 'Invalid language. Must be "asl" or "isl"'}), 400

    if expected_sign not in sign_to_model_group[language]:
        return jsonify({'error': f'Invalid expected_sign for language {language}'}), 400

    # Read image file
    file_bytes = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({'error': 'Invalid image'}), 400

    # Set max_num_hands based on language
    if language == 'asl':
        max_hands = 1
    elif language == 'isl':
        max_hands = 2
    else:
        return jsonify({'error': 'Unsupported language'}), 400

    # Initialize MediaPipe hands with the appropriate max_num_hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=max_hands,
        min_detection_confidence=0.5
    )

    # Process image with MediaPipe
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if not results.multi_hand_landmarks:
        return jsonify({'error': 'No hand detected in the image'}), 400

    # Extract landmarks
    data_aux = []
    x_all = []
    y_all = []
    z_all = []

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

    if x_all and y_all and z_all:
        min_x = min(x_all)
        min_y = min(y_all)
        min_z = min(z_all)
        for x_val, y_val, z_val in zip(x_all, y_all, z_all):
            data_aux.extend([x_val - min_x, y_val - min_y, z_val - min_z])

        expected_length = 63 * max_hands  # 63 features per hand
        if len(data_aux) != expected_length:
            return jsonify({'error': 'Invalid data length'}, 400)

        X_input = np.array(data_aux).reshape(1, -1)

        # Get the appropriate model and label encoder
        model_group = sign_to_model_group[language][expected_sign]
        model = models[language].get(model_group)
        label_encoder = label_encoders[language].get(model_group)

        if model is None or label_encoder is None:
            return jsonify({'error': f'Model for group {model_group} not found in {language} model folder'}), 500

        y_pred = model.predict(X_input)
        predicted_class = label_encoder.inverse_transform(y_pred)[0]

        # Compare with expected sign
        result = (predicted_class == expected_sign)

        return jsonify({
            'result': result,
            'predicted_sign': predicted_class,
            'expected_sign': expected_sign
        })
    else:
        return jsonify({'error': 'Could not extract landmarks'}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
