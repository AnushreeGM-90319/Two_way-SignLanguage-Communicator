import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
DATA_PATH = 'MP_Data'
actions = np.array(['hello', 'thankyou', 'iloveyou', 'yes', 'no', 'help', 'sorry', 'please', 'okay'])
sequence_length = 30
input_size = 154

# 1. Load Data
label_map = {label: num for num, label in enumerate(actions)}
sequences, labels = [], []

print("Loading data...")
for action in actions:
    for sequence in range(30):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences).astype(np.float32)
y = np.array(labels).astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# 2. Define the Architecture
class SignLanguageLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=9):
        super(SignLanguageLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # Take output of the last frame
        return out

# 3. Setup Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SignLanguageLSTM(input_size, num_classes=len(actions)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert to Tensors
X_train_t = torch.tensor(X_train).to(device)
y_train_t = torch.tensor(y_train).to(device)

# 4. Training Loop
print("Training started...")
for epoch in range(150):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/150], Loss: {loss.item():.4f}')

# 5. Save the "Knowledge"
torch.save(model.state_dict(), 'sign_model.pth')
print("SUCCESS: Model trained and saved as 'sign_model.pth'")