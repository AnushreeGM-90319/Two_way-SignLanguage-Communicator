import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('TkAgg')

# --- 1. MODEL DEFINITION ---
class SignLanguageLSTM(nn.Module):
    def __init__(self, input_size=154, hidden_size=64, num_layers=2, num_classes=9):
        super(SignLanguageLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- 2. CONFIGURATION ---
actions = np.array(['hello', 'thankyou', 'iloveyou', 'yes', 'no', 'help', 'sorry', 'please', 'okay'])
DATA_PATH = 'MP_Data'
sequence_length = 30
input_size = 154

# --- 3. LOAD TEST DATA ---
print("Loading data for testing...")
label_map = {label: num for num, label in enumerate(actions)}
sequences, labels = [], []

for action in actions:
    for sequence in range(30):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = torch.tensor(np.array(sequences), dtype=torch.float32)
y_true = np.array(labels)

# --- 4. LOAD MODEL ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SignLanguageLSTM(input_size=input_size, num_classes=len(actions)).to(device)
model.load_state_dict(torch.load('sign_model.pth', map_location=device))
model.eval()

# --- 5. RUN EVALUATION ---
print("Evaluating model...")
with torch.no_grad():
    outputs = model(X.to(device))
    _, predicted = torch.max(outputs, 1)
    y_pred = predicted.cpu().numpy()

# --- 6. RESULTS ---
accuracy = accuracy_score(y_true, y_pred)
print(f"\n✅ MODEL ACCURACY: {accuracy * 100:.2f}%")

# Create a Confusion Matrix to see which signs the AI confuses
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=actions, yticklabels=actions, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Sign Language Recognition')
plt.show()