import cv2
import numpy as np
import cvzone

class FaceOnlyView:
    
    def __init__(self, face_size=(480, 640)):
        self.face_height, self.face_width = face_size
        self.margin = 50  # Extra margin around face
        
    def extract_face_region(self, img, face_landmarks=None, face_box=None):
        if face_box is not None:
            x, y, w, h = face_box
            
            x1 = max(0, x - self.margin)
            y1 = max(0, y - self.margin)
            x2 = min(img.shape[1], x + w + self.margin)
            y2 = min(img.shape[0], y + h + self.margin)
            
            face_img = img[y1:y2, x1:x2]
            
            if face_img.shape[0] > 0 and face_img.shape[1] > 0:
                face_img = cv2.resize(face_img, (self.face_width, self.face_height))
                return face_img
        
        return cv2.resize(img, (self.face_width, self.face_height))
    
    def create_info_panel(self, height, width, emotion=None, confidence=0, 
                         blink_count=0, eyes_closed=False, yaw=None, pitch=None,
                         phone_detected=False, mouth_ratio=None):
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        
        y_offset = 30
        line_height = 35
        
        cv2.putText(panel, "DRIVER STATUS", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += line_height + 10
        
        if emotion:
            if emotion == 'Happy':
                color = (0, 255, 0)      # Green
                bg_color = (0, 100, 0)   # Dark green background
            elif emotion == 'Neutral':
                color = (0, 255, 255)    # Yellow
                bg_color = (0, 100, 100) # Dark yellow background
            elif emotion == 'Angry':
                color = (0, 0, 255)      # Red
                bg_color = (0, 0, 100)   # Dark red background
            else:
                color = (255, 255, 255)  # White
                bg_color = (50, 50, 50)  # Gray background
            
            cv2.rectangle(panel, (5, y_offset - 25), (width - 5, y_offset + 45), bg_color, -1)
            cv2.rectangle(panel, (5, y_offset - 25), (width - 5, y_offset + 45), color, 2)
            
            cv2.putText(panel, f"{emotion}", (15, y_offset + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
            y_offset += line_height
            
            cv2.putText(panel, f"{confidence:.1f}%", (15, y_offset + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += line_height + 20
        else:
            cv2.rectangle(panel, (5, y_offset - 25), (width - 5, y_offset + 20), (50, 50, 50), -1)
            cv2.putText(panel, "No Emotion", (15, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
            y_offset += line_height + 20
        
        eye_color = (0, 0, 255) if eyes_closed else (0, 255, 0)
        eye_status = "CLOSED" if eyes_closed else "OPEN"
        cv2.putText(panel, f"Eyes: {eye_status}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_color, 2)
        y_offset += line_height
        cv2.putText(panel, f"Blinks: {blink_count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y_offset += line_height + 10
        
        if yaw is not None and pitch is not None:
            cv2.putText(panel, f"Head Yaw: {yaw:.1f}°", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y_offset += line_height
            cv2.putText(panel, f"Head Pitch: {pitch:.1f}°", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y_offset += line_height + 10
        
        if mouth_ratio is not None:
            yawn_status = "YAWNING" if mouth_ratio > 0.5 else "Normal"
            mouth_color = (0, 165, 255) if mouth_ratio > 0.5 else (200, 200, 200)
            cv2.putText(panel, f"Mouth: {yawn_status}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, mouth_color, 1)
            y_offset += line_height + 10
        
        if phone_detected:
            cv2.putText(panel, "WARNING: PHONE!", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += line_height
        
        return panel
    
    def create_display(self, face_img, emotion=None, confidence=0, 
                      blink_count=0, eyes_closed=False, yaw=None, pitch=None,
                      phone_detected=False, mouth_ratio=None):

        face_display = face_img.copy()
        
        if emotion:
            # Color based on emotion
            if emotion == 'Happy':
                emoji = "😊"
                color = (0, 255, 0)      # Green
                bg_color = (0, 150, 0)   # Dark green
            elif emotion == 'Neutral':
                emoji = "😐"
                color = (0, 255, 255)    # Yellow
                bg_color = (0, 150, 150) # Dark yellow
            elif emotion == 'Angry':
                emoji = "😠"
                color = (0, 0, 255)      # Red
                bg_color = (0, 0, 150)   # Dark red
            else:
                emoji = ""
                color = (255, 255, 255)
                bg_color = (100, 100, 100)
            
            overlay = face_display.copy()
            cv2.rectangle(overlay, (0, 0), (self.face_width, 80), bg_color, -1)
            cv2.addWeighted(overlay, 0.7, face_display, 0.3, 0, face_display)
            
            text = f"{emotion} {emoji}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_x = (self.face_width - text_size[0]) // 2
            cv2.putText(face_display, text, (text_x, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            conf_text = f"{confidence:.1f}%"
            conf_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            conf_x = (self.face_width - conf_size[0]) // 2
            cv2.putText(face_display, conf_text, (conf_x, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Create info panel (smaller, to the right of face)
        info_width = 300
        info_panel = self.create_info_panel(
            self.face_height, info_width,
            emotion, confidence, blink_count, eyes_closed,
            yaw, pitch, phone_detected, mouth_ratio
        )
        
        # Stack face and info panel horizontally
        display = np.hstack([face_display, info_panel])
        
        return display
