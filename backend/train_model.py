import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split 

# --- CONFIGURATION --- 
DATA_PATH = 'MP_Data'
actions = np.array(['hello', 'thankyou', 'iloveyou', 'yes', 'no', 'help', 'sorry', 'okay'])
input_size = 154 

# Auto-detect sequence length
sample_path = os.path.join(DATA_PATH, actions[0], '0')
sequence_length = len([name for name in os.listdir(sample_path) if name.endswith('.npy')])
print(f"🔥 Auto-detected {sequence_length} frames per video.")

# ================= LOAD DATA ================= 
label_map = {label: num for num, label in enumerate(actions)}
sequences, labels = [],[] 

print("Loading dataset...")
for action in actions:
    for sequence in range(30):
        window =[] 
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])
        
X = np.array(sequences).astype(np.float32)
y = np.array(labels).astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42) 

# 🔥 DATA AUGMENTATION: Adds slight random movements so the AI doesn't overfit
# 🔥 Ensure everything stays as float32
def add_noise(data, noise_factor=0.01):
    noise = np.random.normal(0, noise_factor, data.shape).astype(np.float32)
    return (data + noise).astype(np.float32)
# 🔥 ADVANCED AUGMENTATION: Noise + Mirroring
def augment_data(data):
    # 1. Add slight random noise
    noise = np.random.normal(0, 0.005, data.shape).astype(np.float32)
    noised_data = data + noise
    
    # 2. Mirroring (Flip X coordinates)
    # Since landmarks are [x, y, z], flipping X helps with hand orientation
    mirrored_data = data.copy()
    mirrored_data[:, :, 0] = 1.0 - mirrored_data[:, :, 0] 
    
    return noised_data, mirrored_data

# Create two new versions of your data
X_noise, X_mirrored = augment_data(X_train)

# Combine original + noise + mirrored
X_train_final = np.concatenate((X_train, X_noise, X_mirrored), axis=0).astype(np.float32)
y_train_final = np.concatenate((y_train, y_train, y_train), axis=0).astype(np.int64)
X_train_augmented = add_noise(X_train)
#X_train_final = np.concatenate((X_train, X_train_augmented), axis=0).astype(np.float32)
#y_train_final = np.concatenate((y_train, y_train), axis=0).astype(np.int64)

# Convert to tensors and force types again just to be safe
X_train_t = torch.from_numpy(X_train_final).float() 
y_train_t = torch.from_numpy(y_train_final).long()

train_data = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

# ================= ADVANCED CNN + LSTM MODEL ================= 
class Hardcore_CNN_LSTM(nn.Module):
    def __init__(self, input_size=154, num_classes=8):
        super(Hardcore_CNN_LSTM, self).__init__() 
        
        # Deeper CNN with Batch Norm and Dropout
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
        
        # 2-Layer LSTM for deep sequence understanding
        self.lstm = nn.LSTM(64, 128, num_layers=2, batch_first=True, dropout=0.3) 
        
        # Smarter final classifier
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

# ================= TRAIN SETUP ================= 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Hardcore_CNN_LSTM(input_size=input_size, num_classes=len(actions)).to(device)

criterion = nn.CrossEntropyLoss()
# AdamW adds weight decay to prevent overfitting
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4) 
# Scheduler lowers the learning rate when learning stalls
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

# ================= TRAINING ================= 
print("🚀 Starting Hardcore Training (500 Epochs)...")
epochs = 500 

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs = inputs.to(device).float() 
        labels = labels.to(device).long()
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    avg_loss = running_loss / len(train_loader)
    scheduler.step(avg_loss)
    
    if (epoch+1) % 25 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')

# ================= SAVE MODEL =================
torch.save(model.state_dict(), 'sign_model.pth')
print("✅ SUCCESS: Hardcore AI Model trained and saved!")