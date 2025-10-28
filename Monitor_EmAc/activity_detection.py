"""
Activity detection module for dangerous driver actions.
Detects: drinking, talking on phone, yawning, and other activities.
"""
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
from typing import Tuple, Optional
import cv2


class ActivityDetector:
    """Detects dangerous driver activities using Vision Transformer model."""
    
    # Activity label mapping
    ID2LABEL = {
        0: "drinking",
        1: "other_activities",
        2: "talking_phone",
        3: "yawning"
    }
    
    def __init__(self, model_name: str = "Ganaa614/vit-tiny-patch16-224activity_recognition_4feats", device: str = "cpu"):
        """
        Initialize the activity detector.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run inference on ('cpu' or 'cuda')
        """
        print(f"[info] Loading activity detection model: {model_name}")
        self.model_name = model_name
        self.device = device
        
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageClassification.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()
            print(f"[info] Activity detector loaded successfully on {device}")
        except Exception as e:
            print(f"[error] Failed to load activity detection model: {e}")
            raise
    
    def detect_activity(self, frame: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Detect activity in the given frame.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            Tuple of (activity_label, confidence_score)
            Returns (None, 0.0) if detection fails
        """
        try:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            # Preprocess and run inference
            inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                preds = torch.nn.functional.softmax(outputs.logits, dim=-1)
                label_id = preds.argmax(-1).item()
                confidence = preds[0][label_id].item()
            
            label = self.ID2LABEL.get(label_id, "unknown")
            
            return label, confidence
            
        except Exception as e:
            print(f"[warn] Activity detection failed: {e}")
            return None, 0.0
    
    def get_label_name(self, label_id: int) -> str:
        """Get human-readable label name from ID."""
        return self.ID2LABEL.get(label_id, "unknown")
    
    @staticmethod
    def get_all_labels():
        """Return all possible activity labels."""
        return list(ActivityDetector.ID2LABEL.values())
