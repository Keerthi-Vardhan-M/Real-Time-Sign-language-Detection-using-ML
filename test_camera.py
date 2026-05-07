import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

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

# Try camera indexes 0, 1, 2 until one works
camera = None
for i in range(3):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            camera = cap
            print(f"Camera found at index {i}")
            break
        cap.release()

if camera is None:
    print("ERROR: No camera found! Check if your webcam is connected.")
    exit()

print("Camera on! Show your hand. Press Q to quit.")

while True:
    ret, frame = camera.read()

    if not ret or frame is None:
        print("ERROR: Cant read from camera")
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = detector.detect(mp_image)

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            for lm in hand:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            for a, b in HAND_CONNECTIONS:
                ax, ay = int(hand[a].x * w), int(hand[a].y * h)
                bx, by = int(hand[b].x * w), int(hand[b].y * h)
                cv2.line(frame, (ax, ay), (bx, by), (0, 200, 255), 2)

    cv2.imshow("Test - press Q to quit", frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()