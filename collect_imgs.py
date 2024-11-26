import os
import cv2 # OpenCV

DATA_DIR = './data/isl'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define all the classes and their corresponding numbers
class_names = [
    'A', 'B', 'C', 'D', 'E',
    'F', 'G', 'H', 'I', 'J',
]

# Create a list of tuples (class_number, class_name)
class_list = [(i+1, name) for i, name in enumerate(class_names)]

dataset_size = 50  # Number of images per class

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set the resolution to capture necessary body parts
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

for class_number, class_name in class_list:
    folder_name = f"{class_number}_{class_name}"
    class_dir = os.path.join(DATA_DIR, folder_name)

    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class "{class_name}"')

    # Instructional loop: Wait for user to press 'Q' to start capturing
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            cap.release()
            cv2.destroyAllWindows()
            exit()

        cv2.putText(frame, f'Ready to collect "{class_name}"? Press "Q" to start!', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            continue

        cv2.putText(frame, f'Collecting "{class_name}": Image {counter+1}/{dataset_size}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            print("Data collection interrupted by user.")
            break

        img_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(img_path, frame)
        counter += 1

    print(f'Completed data collection for class "{class_name}"\n')

cap.release()
cv2.destroyAllWindows()
