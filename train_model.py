import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# ── Must match collect_data.py ─────────────────────────
SIGNS = ['hello', 'thanks', 'iloveyou', 'yes', 'no']
SEQUENCES   = 100
FRAMES      = 30
DATA_FOLDER = 'MP_Data'
# ──────────────────────────────────────────────────────

# Load data
print("Loading data...")
X, y = [], []
label_map = {sign: i for i, sign in enumerate(SIGNS)}

for sign in SIGNS:
    for seq in range(SEQUENCES):
        frames = []
        for frame_num in range(FRAMES):
            path = os.path.join(DATA_FOLDER, sign, str(seq),
                                str(frame_num) + '.npy')
            frames.append(np.load(path))
        X.append(frames)
        y.append(label_map[sign])

X = np.array(X, dtype=np.float32)  # shape: (90, 30, 126)
y = np.array(y)

print(f"Data shape: {X.shape}")  # should be (90, 30, 126)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42)

X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test  = torch.tensor(X_test)
y_test  = torch.tensor(y_test,  dtype=torch.long)

dataset    = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# LSTM model
class SignModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 64,  batch_first=True)
        self.lstm2 = nn.LSTM(64,          128, batch_first=True)
        self.lstm3 = nn.LSTM(128,         64,  batch_first=True)
        self.fc1   = nn.Linear(64, 64)
        self.fc2   = nn.Linear(64, 32)
        self.fc3   = nn.Linear(32, num_classes)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = x[:, -1, :]
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

input_size  = X.shape[2]   # 126
num_classes = len(SIGNS)   # 3
model       = SignModel(input_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train
print("Training... please wait")
for epoch in range(200):
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss   = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/200  Loss: {loss.item():.4f}")

# Test accuracy
with torch.no_grad():
    preds    = torch.argmax(model(X_test), dim=1)
    accuracy = (preds == y_test).float().mean() * 100
    print(f"\nAccuracy: {accuracy:.1f}%")

torch.save(model.state_dict(), 'sign_model.pth')
print("Model saved! Now run: python detect_live.py")