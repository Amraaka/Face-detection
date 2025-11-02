import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TFSMLayer
from huggingface_hub import snapshot_download
import os

repo_id = "Ganaa614/emotion-cnn-model_binary"
model_dir = snapshot_download(repo_id=repo_id)
print(f"Model downloaded to: {model_dir}")

input_tensor = Input(shape=(48, 48, 1))
layer = TFSMLayer(model_dir, call_endpoint='serving_default')
output_tensor = layer(input_tensor)
model = Model(inputs=input_tensor, outputs=output_tensor)
print("✅ Model loaded successfully using TFSMLayer!")

emotion_labels = [ "stressed", "non_stressed"]
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(1) 
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
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))

        roi = roi_gray.reshape(1, 48, 48, 1).astype(np.float32) 
        preds_dict = model.predict(roi, verbose=0)

        preds_array = list(preds_dict.values())[0]

        emotion_idx = np.argmax(preds_array)
        emotion = emotion_labels[emotion_idx]

        confidence = np.max(preds_array) * 100

        cv2.putText(frame, f"{emotion} ({confidence:.1f}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()