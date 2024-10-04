# Sign Language Detection Using Python

Welcome to the Sign Language Detection project! This repository provides a comprehensive solution for recognizing and translating sign language gestures into text using Python, OpenCV, and Mediapipe. Whether you're looking to bridge communication gaps or delve into computer vision, this project serves as a robust foundation.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Collect Images](#1-collect-images)
  - [2. Create Dataset](#2-create-dataset)
  - [3. Train Classifier](#3-train-classifier)
  - [4. Inference](#4-inference)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

This project enables real-time detection and translation of sign language gestures into readable text. It leverages the power of:

- **OpenCV**: For image and video processing.
- **Mediapipe**: For hand landmark detection.
- **Scikit-Learn**: For training a machine learning model.
- **Python**: The glue that brings everything together.

## Features

- **Custom Gesture Recognition**: Easily add or modify gestures.
- **Real-time Processing**: Efficient hand tracking and gesture recognition.
- **Modular Codebase**: Well-organized scripts for each stage of the process.
- **Extensible**: Adaptable for more complex sign language systems.

## Installation

### Prerequisites

- Python 3.6 or higher
- OpenCV
- Mediapipe
- Scikit-Learn
- Numpy

### Step-by-Step Installation

1. **Clone the Repository**

   ```bash
   git clone  https://github.com/thesavarni/sign-language-detector-python.git 
   cd sign-language-detector-python
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment**

   - **Windows**

     ```bash
     venv\Scripts\activate
     ```

   - **macOS/Linux**

     ```bash
     source venv/bin/activate
     ```

4. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   *If `requirements.txt` is not available, install manually:*

   ```bash
   pip install opencv-python mediapipe scikit-learn numpy
   ```

## Usage

Follow these steps to run the project:

### 1. Collect Images

Collect images for each gesture you want to recognize.

```bash
python collect_imgs.py
```

- **Customize Gestures**: Edit `collect_imgs.py` and modify the `class_names` list to include your desired gestures.
  
  ```python
  class_names = ["Thank You", "Love", "Dogs"]
  ```
  
- **Instructions**:
  - The script will guide you through capturing images for each gesture.
  - Press **'Q'** to start and stop capturing images.

### 2. Create Dataset

Process the collected images to create a dataset.

```bash
python create_dataset.py
```

- **Functionality**:
  - Uses Mediapipe to detect hand landmarks.
  - Extracts features and saves them into `data.pickle`.

### 3. Train Classifier

Train a machine learning model using the processed dataset.

```bash
python train_classifier.py
```

- **Details**:
  - Loads data from `data.pickle`.
  - Trains a Random Forest classifier.
  - Saves the trained model to `model.p`.

- **Output**:
  - Displays model accuracy, classification report, and confusion matrix.

### 4. Inference

Run the real-time gesture recognition.

```bash
python inference_classifier.py
```

- **Features**:
  - Captures video from your webcam.
  - Detects hand gestures and translates them into text.
  - Builds sentences as you perform gestures.

- **Controls**:
  - **'Q'**: Quit the application.
  - **Spacebar**: Add a space to the sentence.
  - **Enter**: Clear the current sentence.

## Project Structure

```
sign-language-detection/
├── collect_imgs.py
├── create_dataset.py
├── train_classifier.py
├── inference_classifier.py
├── data/
│   └── ... (Collected images)
├── model.p
├── data.pickle
├── requirements.txt
├── README.md
└── LICENSE
```

- **collect_imgs.py**: Script to collect images for each gesture.
- **create_dataset.py**: Processes images and creates a dataset.
- **train_classifier.py**: Trains the machine learning model.
- **inference_classifier.py**: Runs real-time gesture recognition.
- **data/**: Directory containing collected images.
- **model.p**: Serialized trained model.
- **data.pickle**: Processed dataset.
- **requirements.txt**: Python dependencies.
- **README.md**: Project documentation.
- **LICENSE**: License information.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch:

   ```bash
   git checkout -b feature/YourFeature
   ```

3. Commit your changes:

   ```bash
   git commit -m "Add YourFeature"
   ```

4. Push to the branch:

   ```bash
   git push origin feature/YourFeature
   ```

5. Open a Pull Request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [Mediapipe](https://github.com/google/mediapipe) by Google for hand landmark detection.
- [OpenCV](https://opencv.org/) for real-time computer vision.
- [Scikit-Learn](https://scikit-learn.org/) for machine learning algorithms.

---

Feel free to customize and enhance this README to better suit your project's needs!