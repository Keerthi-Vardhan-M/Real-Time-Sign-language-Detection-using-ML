# Real-Time-Sign-language-Detection-using-ML
Real-time sign language detection using MediaPipe and PyTorch LSTM — detects hand signs and builds sentences live from webcam

## How to use this on your device

1. Install dependencies
   pip install mediapipe opencv-python torch numpy scikit-learn matplotlib

2. Collect your own hand data
   python collect_data.py

3. Train the model on your hand
   python train_model.py

4. Run live detection
   python detect_live.py
