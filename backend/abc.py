import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from collections import deque

# --- CONFIG ---
actions = np.array(['hello', 'thankyou', 'iloveyou', 'yes', 'no', 'help', 'sorry', 'please', 'okay'])
sequence_length = 30
input_size = 154

# --- MODEL DEFINITION ---
class SignLanguageLSTM(nn.Module):
    def __init__(self, input_size=154, hidden_size=64, num_layers=2, num_classes=9):
        super(SignLanguageLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- LOAD MODEL ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SignLanguageLSTM(input_size=input_size, num_classes=len(actions)).to(device)
model.load_state_dict(torch.load('sign_model.pth', map_location=device))
model.eval()

# --- MEDIAPIPE SETUP ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# --- LANDMARK FUNCTION (UPPER BODY + HANDS) ---
def extract_landmarks(results):
    upper_body_indices = [0, 11, 12, 13, 14, 15, 16]

    if results.pose_landmarks:
        pose = np.array([
            [results.pose_landmarks.landmark[i].x,
             results.pose_landmarks.landmark[i].y,
             results.pose_landmarks.landmark[i].z,
             results.pose_landmarks.landmark[i].visibility]
            for i in upper_body_indices
        ]).flatten()
    else:
        pose = np.zeros(len(upper_body_indices) * 4)

    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(63)

    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(63)

    return np.concatenate([pose, lh, rh])

# --- REAL-TIME PREDICTION ---
cap = cv2.VideoCapture(0)

sequence = deque(maxlen=sequence_length)
predictions = deque(maxlen=10)  # for smoothing
current_action = ""

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        # Convert color
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        # Draw landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Extract keypoints
        keypoints = extract_landmarks(results)
        sequence.append(keypoints)

        # When we have enough frames
        if len(sequence) == sequence_length:
            input_data = np.expand_dims(sequence, axis=0)
            input_data = torch.tensor(input_data, dtype=torch.float32).to(device)

            with torch.no_grad():
                output = model(input_data)
                _, pred = torch.max(output, 1)
                predictions.append(pred.item())

            # --- SMOOTHING ---
            if len(predictions) == 10:
                most_common = max(set(predictions), key=predictions.count)
                current_action = actions[most_common]

        # Display result
        cv2.rectangle(frame, (0, 0), (640, 50), (0, 0, 0), -1)
        cv2.putText(frame, f'Prediction: {current_action}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Sign Language Detection", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()