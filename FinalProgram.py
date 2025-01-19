import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Manually define the label dictionary (if not in the pickle file)
label_dict = {0: 'Jumping Jacks', 1: 'Squats', 2: 'Squats ', 3: 'Plank', 4: 'Standing'}  # Example labels

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3)

cap = cv2.VideoCapture(0)

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for pose landmarks
    results = pose.process(frame_rgb)
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            x_.append(landmark.x)
            y_.append(landmark.y)

        for landmark in results.pose_landmarks.landmark:
            data_aux.append(landmark.x - min(x_))
            data_aux.append(landmark.y - min(y_))

        # Predict the exercise
        prediction = model.predict([np.asarray(data_aux)])
        predicted_label = label_dict[int(prediction[0])]  # Map the predicted label

        # Draw landmarks and prediction
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(frame, predicted_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

    cv2.imshow('Exercise Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
