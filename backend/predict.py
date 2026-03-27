import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import mediapipe as mp
import pyttsx3

# --- 1. SETUP MODEL ARCHITECTURE (Must match train_model.py) ---
class SignLanguageLSTM(nn.Module):
    def __init__(self, input_size=154, hidden_size=64, num_layers=2, num_classes=9):
        super(SignLanguageLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- 2. INITIALIZE ---
actions = np.array(['hello', 'thankyou', 'iloveyou', 'yes', 'no', 'help', 'sorry', 'please', 'okay'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained brain
model = SignLanguageLSTM(input_size=154, num_classes=len(actions)).to(device)
model.load_state_dict(torch.load('sign_model.pth', map_location=device))
model.eval()

# Voice Engine
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

# MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# --- 3. HELPER FUNCTIONS ---
def extract_landmarks(results):
    upper_body_indices = [0, 11, 12, 13, 14, 15, 16] # Your optimization
    if results.pose_landmarks:
        pose = np.array([[results.pose_landmarks.landmark[i].x, results.pose_landmarks.landmark[i].y, 
                          results.pose_landmarks.landmark[i].z, results.pose_landmarks.landmark[i].visibility] 
                         for i in upper_body_indices]).flatten()
    else:
        pose = np.zeros(len(upper_body_indices) * 4)
    
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, lh, rh])

# --- 4. REAL-TIME LOOP ---
sequence = []
sentence = []
threshold = 0.8 # Only speak if confidence is > 80%

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        
        # Draw landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # 1. Prediction Logic
        keypoints = extract_landmarks(results)
        sequence.append(keypoints)
        sequence = sequence[-30:] # Keep only the last 30 frames

        if len(sequence) == 30:
            res = model(torch.tensor([sequence], dtype=torch.float32).to(device))
            probabilities = torch.softmax(res, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
            
            action = actions[predicted_idx.item()]
            
            # 2. Visualization/Output
            if confidence.item() > threshold:
                if len(sentence) > 0:
                    if action != sentence[-1]: # Avoid repeating the same word constantly
                        sentence.append(action)
                        print(f"Recognized: {action}")
                        speak(action) # VOICEOVER
                else:
                    sentence.append(action)
                    speak(action)

            # Display the result on screen
            cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(frame, f'PREDICTION: {action.upper()} ({confidence.item()*100:.1f}%)', 
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Sign Language Recognition', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()