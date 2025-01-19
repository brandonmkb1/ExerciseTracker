import os
import pickle
import mediapipe as mp
import cv2

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'  # Directory containing subdirectories for classes

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):  # Loop through class folders
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):  # Loop through images in each class
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image for pose landmarks
        results = pose.process(img_rgb)
        if results.pose_landmarks:
            # Extract landmarks
            for landmark in results.pose_landmarks.landmark:
                x = landmark.x
                y = landmark.y

                x_.append(x)
                y_.append(y)

            # Normalize landmarks relative to the bounding box
            for landmark in results.pose_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))  # Normalize X
                data_aux.append(landmark.y - min(y_))  # Normalize Y

            # Append data and label
            data.append(data_aux)
            labels.append(dir_)

# Save the data to a pickle file
with open('pose_data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data saved successfully in 'pose_data.pickle'")
