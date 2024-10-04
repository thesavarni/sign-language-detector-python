import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define your classes: 5 alphabets and 5 words
class_names  = ["Thank You", "Love", "Dogs"]


number_of_classes = len(class_names)
dataset_size = 200  # Number of images per class

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set the resolution to capture necessary body parts
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

for class_name in class_names:
    class_dir = os.path.join(DATA_DIR, class_name)

    # Corrected: Use class_dir instead of os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    # Corrected: Use class_name instead of j
    print('Collecting data for class "{}"'.format(class_name))

    # Instructional loop: Wait for user to press 'Q' to start capturing
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            cap.release()
            cv2.destroyAllWindows()
            exit()

        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            continue

        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            print("Data collection interrupted by user.")
            break

        # Corrected: Use class_dir instead of os.path.join(DATA_DIR, str(j))
        img_path = os.path.join(class_dir, '{}.jpg'.format(counter))
        cv2.imwrite(img_path, frame)
        counter += 1

    print(f'Completed data collection for class "{class_name}"\n')

cap.release()
cv2.destroyAllWindows()
