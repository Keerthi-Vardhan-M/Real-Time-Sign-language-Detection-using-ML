import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
import torch.nn as nn
from collections import deque

# ── Must match your other files ────────────────────────
SIGNS = ['hello', 'thanks', 'iloveyou', 'yes', 'no']
FRAMES    = 30
THRESHOLD = 0.85   # lowered so it detects easier with 55% accuracy
# ──────────────────────────────────────────────────────

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (5,6),(6,7),(7,8),
    (9,10),(10,11),(11,12),
    (13,14),(14,15),(15,16),
    (17,18),(18,19),(19,20),
    (0,5),(5,9),(9,13),(13,17),(0,17)
]

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

# Load model
model = SignModel(126, len(SIGNS))
model.load_state_dict(torch.load('sign_model.pth'))
model.eval()

# Load detector
MODEL = 'hand_landmarker.task'
options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL),
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

def get_keypoints(result):
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)
    if result.hand_landmarks:
        for i, hand in enumerate(result.hand_landmarks):
            coords     = np.array([[lm.x, lm.y, lm.z]
                                    for lm in hand]).flatten()
            handedness = result.handedness[i][0].category_name
            if handedness == 'Left':
                lh = coords
            else:
                rh = coords
    return np.concatenate([lh, rh])

sequence  = deque(maxlen=FRAMES)
sentence  = []
last_sign = None

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("Live detection started! Press Q to quit.")

while True:
    ret, frame = camera.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = detector.detect(mp_image)

    # Draw hand landmarks
    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            for lm in hand:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0,255,0), -1)
            for a, b in HAND_CONNECTIONS:
                ax, ay = int(hand[a].x * w), int(hand[a].y * h)
                bx, by = int(hand[b].x * w), int(hand[b].y * h)
                cv2.line(frame, (ax,ay), (bx,by), (0,200,255), 2)

    # Add frame to buffer
    sequence.append(get_keypoints(result))

    if len(sequence) == FRAMES:
        input_tensor = torch.tensor(
            np.expand_dims(list(sequence), axis=0),
            dtype=torch.float32)

        with torch.no_grad():
            output     = torch.softmax(model(input_tensor), dim=1)[0]
            confidence = float(output.max())
            sign       = SIGNS[output.argmax()]

        if confidence > THRESHOLD:
            if sign != last_sign:
                sentence.append(sign)
                last_sign = sign
                if len(sentence) > 5:
                    sentence = sentence[-5:]
        else:
            last_sign = None

        # Confidence bar
        cv2.rectangle(frame, (0,0), (int(confidence * 300), 25),
                      (0,255,0), -1)
        cv2.putText(frame, f"{sign} {confidence:.0%}",
                    (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,0,0), 2)

    # Sentence bar at bottom
    cv2.rectangle(frame, (0, h-50), (w, h), (0,0,0), -1)
    cv2.putText(frame, ' '.join(sentence),
                (10, h-15), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255,255,255), 2)

    cv2.imshow("Sign Language Detection - Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()