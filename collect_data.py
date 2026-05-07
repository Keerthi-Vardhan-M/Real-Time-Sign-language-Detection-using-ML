import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# ── Settings ──────────────────────────────────────────
SIGNS = ['hello', 'thanks', 'iloveyou', 'yes', 'no']
SEQUENCES   = 100
FRAMES      = 30
DATA_FOLDER = 'MP_Data'
# ──────────────────────────────────────────────────────

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (5,6),(6,7),(7,8),
    (9,10),(10,11),(11,12),
    (13,14),(14,15),(15,16),
    (17,18),(18,19),(19,20),
    (0,5),(5,9),(9,13),(13,17),(0,17)
]

MODEL = 'hand_landmarker.task'
options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL),
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

def get_keypoints(result):
    # Each hand has 21 landmarks (x, y, z) = 63 numbers
    # We store 2 hands = 126 numbers total
    # If hand not detected, store zeros
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)

    if result.hand_landmarks:
        for i, hand in enumerate(result.hand_landmarks):
            coords = np.array([[lm.x, lm.y, lm.z]
                                for lm in hand]).flatten()
            handedness = result.handedness[i][0].category_name
            if handedness == 'Left':
                lh = coords
            else:
                rh = coords

    return np.concatenate([lh, rh])  # 126 numbers

# Create folders
for sign in SIGNS:
    for seq in range(SEQUENCES):
        os.makedirs(os.path.join(DATA_FOLDER, sign, str(seq)), exist_ok=True)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("Starting data collection...")

for sign in SIGNS:
    for seq in range(SEQUENCES):

        # Countdown
        for countdown in range(3, 0, -1):
            ret, frame = camera.read()
            frame = cv2.flip(frame, 1)
            cv2.putText(frame,
                f"Sign: '{sign}'  Recording {seq+1}/{SEQUENCES}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f"Starting in {countdown}...",
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
            cv2.imshow("Collecting Data - press Q to quit", frame)
            cv2.waitKey(1000)

        # Record frames
        for frame_num in range(FRAMES):
            ret, frame = camera.read()
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = detector.detect(mp_image)

            # Draw landmarks
            if result.hand_landmarks:
                for hand in result.hand_landmarks:
                    for lm in hand:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 5, (0,255,0), -1)
                    for a, b in HAND_CONNECTIONS:
                        ax, ay = int(hand[a].x * w), int(hand[a].y * h)
                        bx, by = int(hand[b].x * w), int(hand[b].y * h)
                        cv2.line(frame, (ax,ay), (bx,by), (0,200,255), 2)

            cv2.putText(frame,
                f"Recording '{sign}' - frame {frame_num+1}/{FRAMES}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow("Collecting Data - press Q to quit", frame)
            cv2.waitKey(1)

            # Save keypoints
            keypoints = get_keypoints(result)
            save_path = os.path.join(DATA_FOLDER, sign, str(seq), str(frame_num))
            np.save(save_path, keypoints)

        print(f"Done: '{sign}' sequence {seq+1}/{SEQUENCES}")

camera.release()
cv2.destroyAllWindows()
print("\nAll data collected! Now run: python train_model.py")