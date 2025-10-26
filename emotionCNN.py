import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TFSMLayer
from huggingface_hub import snapshot_download
import os

# === 1. Download your model from Hugging Face Hub ===
repo_id = "Ganaa614/emotion-cnn-model3_feats_balanced"
model_dir = snapshot_download(repo_id=repo_id)
print(f"Model downloaded to: {model_dir}")

# === FIX 1: Load the model using TFSMLayer for Keras 3 ===
input_tensor = Input(shape=(48, 48, 1))
layer = TFSMLayer(model_dir, call_endpoint='serving_default')
output_tensor = layer(input_tensor)
model = Model(inputs=input_tensor, outputs=output_tensor)
print("✅ Model loaded successfully using TFSMLayer!")

# === 2. Define emotion labels (assuming 7 classes) ===
emotion_labels = ['Angry', 'Happy', 'Neutral']

# === 3. Load face detector ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# === 4. Open webcam ===
# Note: If you get the "Webcam not found" error again,
# try changing the index from 0 to 1, 2, or -1
cap = cv2.VideoCapture(0) 
if not cap.isOpened():
    raise Exception("Webcam not found or cannot be accessed!")
print("🎥 Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # --- Preprocess face ---
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        
        # === FIX 3: Preprocessing Correction ===
        # Your model has a Rescaling(1./255) layer,
        # so we MUST pass in the raw 0-255 pixel values.
        # We do NOT divide by 255.0 here.
        roi = roi_gray.reshape(1, 48, 48, 1).astype(np.float32) 
        # =======================================

        # --- Predict ---
        
        # === FIX 2: Unwrap the dictionary ===
        # 1. Predict - this returns a dictionary
        preds_dict = model.predict(roi, verbose=0)
        
        # 2. Get the actual prediction array from the dictionary's values
        preds_array = list(preds_dict.values())[0]
        # ==================================

        # 3. Get final emotion and confidence
        emotion_idx = np.argmax(preds_array)
        emotion = emotion_labels[emotion_idx]
        
        # The model's final layer is "softmax", so the outputs
        # are already probabilities that sum to 1.
        confidence = np.max(preds_array) * 100
        
        # --- Draw text on frame ---
        cv2.putText(frame, f"{emotion} ({confidence:.1f}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # --- Show the frame ---
    cv2.imshow('Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === 5. Release resources ===
cap.release()
cv2.destroyAllWindows()