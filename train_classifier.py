import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import glob

# Get list of data group pickle files
data_files = sorted(glob.glob('./asl_dataset_pickle/data_group_*.pickle'))

for data_file in data_files:
    # Extract group number from file name
    group_number = int(data_file.split('_')[-1].split('.')[0])

    print(f"\nTraining model for {data_file} (Group {group_number})")

    # Load the processed data
    with open(data_file, 'rb') as f:
        data_dict = pickle.load(f)

    data = np.array(data_dict['data'])
    labels = np.array(data_dict['labels'])

    print(f"  Total samples: {data.shape[0]}")
    print(f"  Feature vector length: {data.shape[1]}")

    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
    )

    # Initialize and train the classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    # Evaluate the model
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Model accuracy on test set: {accuracy * 100:.2f}%")

    # Detailed classification report
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Confusion matrix
    print("\n  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save the trained model and label encoder
    model_file = f'./asl_model/model_group_{group_number}.p'
    with open(model_file, 'wb') as f:
        pickle.dump({'model': model, 'label_encoder': label_encoder}, f)

    print(f"  Model training complete. Saved to '{model_file}'.")

print("\nAll models trained and saved.")
