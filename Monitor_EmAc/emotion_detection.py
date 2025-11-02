import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TFSMLayer
from huggingface_hub import snapshot_download
import os

class EmotionDetector:
    def __init__(self, model_repo="Ganaa614/emotion-cnn-model3_feats_balanced"):
    # def __init__(self, model_repo="Ganaa614/emotion-cnn-model_binary"):

        self.emotion_labels = ['Angry', 'Happy', 'Neutral']
        # self.emotion_labels = ['non_stressed', 'stressed']
        self.model = None
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self._load_model(model_repo)
        
    def _load_model(self, repo_id):
        try:
            print(f"[info] Loading emotion detection model from {repo_id}...")
            model_dir = snapshot_download(repo_id=repo_id)
            print(f"[info] Model downloaded to: {model_dir}")
            
            input_tensor = Input(shape=(48, 48, 1))
            layer = TFSMLayer(model_dir, call_endpoint='serving_default')
            output_tensor = layer(input_tensor)
            self.model = Model(inputs=input_tensor, outputs=output_tensor)
            print("[info] ✅ Emotion detection model loaded successfully!")
        except Exception as e:
            print(f"[warn] Failed to load emotion model: {e}")
            self.model = None
    
    def detect_emotion(self, img, face_roi=None):

        if self.model is None:
            return None, 0, None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if face_roi is None:
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5
            )
            if len(faces) == 0:
                return None, 0, None
            face_roi = faces[0]
        
        x, y, w, h = face_roi
        
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        
        roi = roi_gray.reshape(1, 48, 48, 1).astype(np.float32)
        
        preds_dict = self.model.predict(roi, verbose=0)
        preds_array = list(preds_dict.values())[0]
        
        emotion_idx = np.argmax(preds_array)
        emotion = self.emotion_labels[emotion_idx]
        confidence = np.max(preds_array) * 100
        
        return emotion, confidence, (x, y, w, h)
    
    def draw_emotion(self, img, emotion, confidence, face_box):
        if emotion is None or face_box is None:
            return
        
        x, y, w, h = face_box
        
        color_map = {
            'Happy': (0, 255, 0),      # Green
            'Neutral': (255, 255, 0),   # Yellow
            'Angry': (0, 0, 255)        # Red
        }
        color = color_map.get(emotion, (255, 255, 255))
        
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        
        text = f"{emotion} ({confidence:.1f}%)"
        cv2.putText(img, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
