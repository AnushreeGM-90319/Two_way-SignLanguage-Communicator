




import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
import base64
from collections import Counter
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# --- 1. CONFIGURATION ---
actions = np.array(['hello', 'thankyou', 'iloveyou', 'yes', 'no', 'help', 'sorry', 'okay'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. MODEL DEFINITION ---
class Hardcore_CNN_LSTM(nn.Module):
    def __init__(self, input_size=154, num_classes=8):
        super(Hardcore_CNN_LSTM, self).__init__() 
        self.cnn = nn.Sequential( 
            nn.Conv1d(input_size, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3)
        ) 
        self.lstm = nn.LSTM(64, 128, num_layers=2, batch_first=True, dropout=0.3) 
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x = self.cnn(x)
        x = x.permute(0, 2, 1) 
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out 

# --- 3. LOAD MODEL ---
print("🚀 Initializing main_communicator backend...")
model = Hardcore_CNN_LSTM(input_size=154, num_classes=len(actions)).to(device)
try:
    
    model.load_state_dict(torch.load('sign_model.pth', map_location=device, weights_only=False))
    model.eval()
    print("✅ Hardcore AI Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error Loading Model: {e}")

# --- 4. FASTAPI & MEDIAPIPE ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)

def extract_landmarks(results):
    # Match the 154 features used in training
    indices = [0, 11, 12, 13, 14, 15, 16] 
    pose = np.array([[results.pose_landmarks.landmark[i].x, results.pose_landmarks.landmark[i].y, results.pose_landmarks.landmark[i].z, results.pose_landmarks.landmark[i].visibility] for i in indices]).flatten() if results.pose_landmarks else np.zeros(28)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, lh, rh])

# --- 5. WEBSOCKET LOGIC ---
@app.websocket("/ws/sign")
async def websocket_sign(websocket: WebSocket):
    await websocket.accept()
    sequence = []
    prediction_buffer = [] 
    try:
        while True:
            data = await websocket.receive_text()
            if "," not in data: continue
            
            # Decode frame
            encoded_data = data.split(",")[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None: continue

            # Extract features
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            landmarks = extract_landmarks(results)
            sequence.append(landmarks)
            sequence = sequence[-30:] 

            prediction = "..."

            if len(sequence) == 30:
                input_tensor = torch.tensor(np.array(sequence, dtype=np.float32)[None, ...]).to(device)
                with torch.no_grad():
                    res = model(input_tensor)
                
                prob = torch.softmax(res, dim=1)[0]
                confidence, idx = torch.max(prob, dim=0)
                
                # Filter by confidence
                current_guess = actions[idx.item()] if confidence.item() > 0.85 else "..."
                
                
                prediction_buffer.append(current_guess)
                prediction_buffer = prediction_buffer[-10:]
                
                most_common = Counter(prediction_buffer).most_common(1)[0]
                
                if most_common[1] >= 6:
                    prediction = most_common[0]

            await websocket.send_json({"prediction": prediction})

    except Exception as e:
        print(f"🔴 Client Disconnected: {e}")