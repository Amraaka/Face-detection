import cv2
import numpy as np
import cvzone
from collections import deque
from time import time

class FaceOnlyView:
    
    def __init__(self, face_size=(480, 640)):
        self.face_height, self.face_width = face_size
        self.margin = 50
        
        # Store emotion history for statistics (last 100 frames)
        self.emotion_history = deque(maxlen=100)
        self.emotion_counts = {'stressed': 0, 'not_stressed': 0}
        self.last_update_time = time()  
        
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
    
    def _draw_section_header(self, panel, text, y_offset, width, color=(100, 150, 255)):
        """Draw a section header with background"""
        cv2.rectangle(panel, (0, y_offset - 25), (width, y_offset + 5), (30, 30, 30), -1)
        cv2.putText(panel, text, (10, y_offset - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        return y_offset + 15
    
    def _draw_progress_bar(self, panel, x, y, w, h, percentage, color, bg_color=(40, 40, 40)):
        """Draw a progress bar"""
        # Background
        cv2.rectangle(panel, (x, y), (x + w, y + h), bg_color, -1)
        # Progress
        fill_width = int(w * min(percentage, 1.0))
        if fill_width > 0:
            cv2.rectangle(panel, (x, y), (x + fill_width, y + h), color, -1)
        # Border
        cv2.rectangle(panel, (x, y), (x + w, y + h), (80, 80, 80), 1)
    
    def _update_emotion_stats(self, emotion):
        """Update emotion statistics tracking"""
        if emotion:
            self.emotion_history.append(emotion.lower())
            # Recalculate counts
            self.emotion_counts = {'stressed': 0, 'not_stressed': 0}
            for e in self.emotion_history:
                if e in self.emotion_counts:
                    self.emotion_counts[e] += 1
    
    def create_top_info_bar(self, width, emotion=None, confidence=0, 
                           blink_count=0, eyes_closed=False, yaw=None, pitch=None,
                           phone_detected=False, mouth_ratio=None, activity_label=None,
                           activity_confidence=0.0):
        """Create a detailed horizontal info bar at the top"""
        bar_height = 120
        panel = np.zeros((bar_height, width, 3), dtype=np.uint8)
        panel[:] = (20, 20, 25)  # Dark background
        
        # Draw border
        cv2.rectangle(panel, (0, 0), (width-1, bar_height-1), (60, 60, 80), 2)
        
        # Calculate section widths for 4 columns
        col_width = width // 4
        
        # ==================== COLUMN 1: EMOTION ====================
        x_offset = 10
        y_start = 15
        
        cv2.putText(panel, "EMOTION STATE", (x_offset, y_start),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        
        if emotion:
            display_emotion = emotion.lower()
            if display_emotion == 'stressed':
                color = (0, 0, 255)
                emoji = "😰"
                status = "STRESSED"
            else:
                color = (0, 255, 0)
                emoji = "😊"
                status = "CALM"
            
            cv2.putText(panel, f"{emoji} {status}", (x_offset, y_start + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.putText(panel, f"Conf: {confidence:.0f}%", (x_offset, y_start + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            
            # Mini confidence bar
            bar_width = col_width - 20
            self._draw_progress_bar(panel, x_offset, y_start + 70, bar_width, 8, 
                                   confidence/100, color)
            
            # History stats
            total = len(self.emotion_history)
            if total > 0:
                stressed_pct = (self.emotion_counts['stressed'] / total) * 100
                cv2.putText(panel, f"History: S:{stressed_pct:.0f}%", (x_offset, y_start + 95),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 200), 1)
        else:
            cv2.putText(panel, "No Detection", (x_offset, y_start + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)
        
        # ==================== COLUMN 2: EYES & HEAD ====================
        x_offset = col_width + 10
        
        cv2.putText(panel, "EYES & HEAD", (x_offset, y_start),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
        
        # Eyes
        eye_color = (0, 0, 255) if eyes_closed else (0, 255, 0)
        eye_status = "CLOSED ⚠" if eyes_closed else "OPEN ✓"
        cv2.putText(panel, f"Eyes: {eye_status}", (x_offset, y_start + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, eye_color, 1)
        
        cv2.putText(panel, f"Blinks: {blink_count}", (x_offset, y_start + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        # Head pose
        if yaw is not None and pitch is not None:
            direction = "FWD"
            dir_color = (0, 255, 0)
            if abs(yaw) > 20 or abs(pitch) > 20:
                if abs(yaw) > abs(pitch):
                    direction = "LEFT" if yaw > 0 else "RIGHT"
                else:
                    direction = "UP" if pitch < 0 else "DOWN"
                dir_color = (255, 165, 0)
            
            cv2.putText(panel, f"Head: {direction}", (x_offset, y_start + 74),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, dir_color, 1)
            cv2.putText(panel, f"Y:{yaw:.0f}° P:{pitch:.0f}°", (x_offset, y_start + 96),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 200), 1)
        
        # ==================== COLUMN 3: ACTIVITY ====================
        x_offset = col_width * 2 + 10
        
        cv2.putText(panel, "ACTIVITY", (x_offset, y_start),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 255), 1)
        
        if activity_label:
            if activity_label == "drinking":
                activity_color = (0, 165, 255)
                emoji = "🥤"
            elif activity_label == "talking_phone":
                activity_color = (0, 0, 255)
                emoji = "📱"
            elif activity_label == "yawning":
                activity_color = (255, 255, 0)
                emoji = "🥱"
            else:
                activity_color = (200, 200, 200)
                emoji = "⚙️"
            
            display_label = activity_label.replace("_", " ").title()
            cv2.putText(panel, f"{emoji} {display_label}", (x_offset, y_start + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, activity_color, 2)
            cv2.putText(panel, f"Conf: {activity_confidence*100:.0f}%", (x_offset, y_start + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            
            # Mini confidence bar
            bar_width = col_width - 20
            self._draw_progress_bar(panel, x_offset, y_start + 70, bar_width, 8, 
                                   activity_confidence, activity_color)
        else:
            cv2.putText(panel, "Normal Driving", (x_offset, y_start + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 100), 1)
        
        # Mouth status
        if mouth_ratio is not None:
            yawn_status = "YAWNING" if mouth_ratio > 0.5 else "Normal"
            mouth_color = (0, 165, 255) if mouth_ratio > 0.5 else (150, 150, 150)
            cv2.putText(panel, f"Mouth: {yawn_status}", (x_offset, y_start + 96),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, mouth_color, 1)
        
        # ==================== COLUMN 4: WARNINGS & STATUS ====================
        x_offset = col_width * 3 + 10
        
        cv2.putText(panel, "WARNINGS", (x_offset, y_start),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 100), 1)
        
        warning_y = y_start + 30
        has_warning = False
        
        if phone_detected:
            cv2.rectangle(panel, (x_offset - 5, warning_y - 20), 
                         (x_offset + col_width - 15, warning_y + 10), (0, 0, 100), -1)
            cv2.rectangle(panel, (x_offset - 5, warning_y - 20), 
                         (x_offset + col_width - 15, warning_y + 10), (0, 0, 255), 2)
            cv2.putText(panel, "📱 PHONE!", (x_offset + 5, warning_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            warning_y += 35
            has_warning = True
        
        if eyes_closed:
            cv2.rectangle(panel, (x_offset - 5, warning_y - 20), 
                         (x_offset + col_width - 15, warning_y + 10), (0, 0, 100), -1)
            cv2.rectangle(panel, (x_offset - 5, warning_y - 20), 
                         (x_offset + col_width - 15, warning_y + 10), (0, 0, 255), 2)
            cv2.putText(panel, "😴 DROWSY!", (x_offset + 5, warning_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            has_warning = True
        
        if not has_warning:
            cv2.rectangle(panel, (x_offset - 5, warning_y - 20), 
                         (x_offset + col_width - 15, warning_y + 10), (0, 50, 0), -1)
            cv2.putText(panel, "✓ ALL CLEAR", (x_offset + 5, warning_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add timestamp at bottom
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(panel, timestamp, (x_offset + 10, bar_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 150), 1)
        
        return panel
    
    def create_info_panel(self, height, width, emotion=None, confidence=0, 
                         blink_count=0, eyes_closed=False, yaw=None, pitch=None,
                         phone_detected=False, mouth_ratio=None, activity_label=None,
                         activity_confidence=0.0):
        # Create darker panel
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (15, 15, 15)  # Dark background
        
        y_offset = 20
        line_height = 28
        
        # ==================== MAIN HEADER ====================
        cv2.rectangle(panel, (0, 0), (width, 45), (40, 60, 100), -1)
        cv2.putText(panel, "DRIVER MONITOR", (width//2 - 95, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_offset = 60
        
        # ==================== EMOTION SECTION ====================
        y_offset = self._draw_section_header(panel, "EMOTION STATE", y_offset, width, (100, 200, 255))
        
        # Update emotion stats
        self._update_emotion_stats(emotion)
        
        if emotion:
            display_emotion = emotion.lower()
            
            # Color coding based on emotion
            if display_emotion == 'stressed':
                color = (0, 0, 255)      # Red
                bg_color = (0, 0, 100)   # Dark red
                emoji = "😰"
            else:  # not_stressed
                color = (0, 255, 0)      # Green
                bg_color = (0, 100, 0)   # Dark green
                emoji = "😊"
            
            # Main emotion display box
            cv2.rectangle(panel, (8, y_offset), (width - 8, y_offset + 70), bg_color, -1)
            cv2.rectangle(panel, (8, y_offset), (width - 8, y_offset + 70), color, 2)
            
            emotion_text = display_emotion.replace('_', ' ').upper()
            cv2.putText(panel, f"{emoji} {emotion_text}", (20, y_offset + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            
            # Confidence bar
            cv2.putText(panel, f"Confidence: {confidence:.1f}%", (20, y_offset + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            self._draw_progress_bar(panel, 20, y_offset + 60, width - 40, 8, 
                                   confidence/100, color)
            y_offset += 85
        else:
            cv2.rectangle(panel, (8, y_offset), (width - 8, y_offset + 40), (50, 50, 50), -1)
            cv2.putText(panel, "No Emotion Detected", (20, y_offset + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            y_offset += 55
        
        # Emotion Statistics
        total_frames = len(self.emotion_history)
        if total_frames > 0:
            stressed_pct = (self.emotion_counts['stressed'] / total_frames) * 100
            not_stressed_pct = (self.emotion_counts['not_stressed'] / total_frames) * 100
            
            cv2.putText(panel, "Recent History (100 frames):", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
            y_offset += 20
            
            # Stressed bar
            cv2.putText(panel, f"Stressed: {stressed_pct:.0f}%", (15, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 100), 1)
            y_offset += 15
            self._draw_progress_bar(panel, 15, y_offset, width - 30, 12, 
                                   stressed_pct/100, (0, 0, 255))
            y_offset += 20
            
            # Not stressed bar
            cv2.putText(panel, f"Not Stressed: {not_stressed_pct:.0f}%", (15, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)
            y_offset += 15
            self._draw_progress_bar(panel, 15, y_offset, width - 30, 12, 
                                   not_stressed_pct/100, (0, 255, 0))
            y_offset += 25
        
        # ==================== EYE & BLINK SECTION ====================
        y_offset = self._draw_section_header(panel, "EYE MONITORING", y_offset, width, (255, 200, 100))
        
        eye_color = (0, 0, 255) if eyes_closed else (0, 255, 0)
        eye_status = "⚠ CLOSED" if eyes_closed else "✓ OPEN"
        eye_emoji = "😴" if eyes_closed else "👁"
        
        cv2.putText(panel, f"{eye_emoji} Eyes: {eye_status}", (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_color, 2)
        y_offset += line_height
        
        # Blink counter with icon
        cv2.putText(panel, f"👁️ Blink Count: {blink_count}", (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        y_offset += line_height + 5
        
        # ==================== HEAD POSE SECTION ====================
        if yaw is not None and pitch is not None:
            y_offset = self._draw_section_header(panel, "HEAD POSITION", y_offset, width, (255, 150, 200))
            
            # Direction indicator
            direction = "FORWARD"
            if abs(yaw) > 20 or abs(pitch) > 20:
                if abs(yaw) > abs(pitch):
                    direction = "LEFT ←" if yaw > 0 else "RIGHT →"
                else:
                    direction = "UP ↑" if pitch < 0 else "DOWN ↓"
            
            dir_color = (0, 255, 0) if direction == "FORWARD" else (255, 165, 0)
            cv2.putText(panel, f"Direction: {direction}", (15, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, dir_color, 1)
            y_offset += line_height
            
            cv2.putText(panel, f"Yaw: {yaw:.1f}°  Pitch: {pitch:.1f}°", (15, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            y_offset += line_height + 5
        
        # ==================== MOUTH ANALYSIS SECTION ====================
        if mouth_ratio is not None:
            yawn_status = "YAWNING ⚠" if mouth_ratio > 0.5 else "Normal"
            mouth_color = (0, 165, 255) if mouth_ratio > 0.5 else (200, 200, 200)
            
            cv2.putText(panel, f"Mouth: {yawn_status}", (15, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, mouth_color, 1)
            
            # Mouth opening bar
            if mouth_ratio > 0.3:  # Only show bar if mouth is opening
                y_offset += 18
                self._draw_progress_bar(panel, 15, y_offset, width - 30, 10, 
                                       min(mouth_ratio, 1.0), mouth_color)
            y_offset += line_height + 5
        
        # ==================== ACTIVITY DETECTION SECTION ====================
        if activity_label:
            y_offset = self._draw_section_header(panel, "ACTIVITY DETECTED", y_offset, width, (255, 100, 255))
            
            # Color coding for different activities
            if activity_label == "drinking":
                activity_color = (0, 165, 255)  # Orange
                bg_color = (0, 80, 120)
                emoji = "🥤"
            elif activity_label == "talking_phone":
                activity_color = (0, 0, 255)  # Red
                bg_color = (0, 0, 100)
                emoji = "📱"
            elif activity_label == "yawning":
                activity_color = (255, 255, 0)  # Cyan
                bg_color = (100, 100, 0)
                emoji = "🥱"
            else:  # other_activities
                activity_color = (200, 200, 200)  # Gray
                bg_color = (50, 50, 50)
                emoji = "⚙️"
            
            # Activity display box
            cv2.rectangle(panel, (8, y_offset), (width - 8, y_offset + 65), bg_color, -1)
            cv2.rectangle(panel, (8, y_offset), (width - 8, y_offset + 65), activity_color, 2)
            
            display_label = activity_label.replace("_", " ").title()
            cv2.putText(panel, f"{emoji} {display_label}", (20, y_offset + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, activity_color, 2)
            
            # Confidence display
            cv2.putText(panel, f"Confidence: {activity_confidence*100:.1f}%", (20, y_offset + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            self._draw_progress_bar(panel, 20, y_offset + 50, width - 40, 8, 
                                   activity_confidence, activity_color)
            y_offset += 75
        
        # ==================== WARNINGS SECTION ====================
        if phone_detected:
            y_offset = self._draw_section_header(panel, "⚠ WARNINGS ⚠", y_offset, width, (255, 0, 0))
            cv2.rectangle(panel, (8, y_offset), (width - 8, y_offset + 40), (0, 0, 100), -1)
            cv2.rectangle(panel, (8, y_offset), (width - 8, y_offset + 40), (0, 0, 255), 2)
            cv2.putText(panel, "📱 PHONE DETECTED!", (20, y_offset + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            y_offset += 50
        
        return panel
    
    def create_display(self, face_img, emotion=None, confidence=0, 
                      blink_count=0, eyes_closed=False, yaw=None, pitch=None,
                      phone_detected=False, mouth_ratio=None, activity_label=None,
                      activity_confidence=0.0):

        face_display = face_img.copy()
        
        if emotion:
            # Normalize emotion label to lowercase
            emotion_normalized = emotion.lower()
            
            # Color coding based on emotion
            if emotion_normalized == 'stressed':
                display_text = "STRESSED"
                emoji = "�"
                color = (0, 0, 255)      # Red
                bg_color = (0, 0, 150)   # Dark red
            else:  # not_stressed
                display_text = "NOT STRESSED"
                emoji = "😊"
                color = (0, 255, 0)      # Green
                bg_color = (0, 150, 0)   # Dark green
            
            # Semi-transparent overlay at top
            overlay = face_display.copy()
            cv2.rectangle(overlay, (0, 0), (self.face_width, 90), bg_color, -1)
            cv2.addWeighted(overlay, 0.7, face_display, 0.3, 0, face_display)
            
            # Emotion text with emoji
            text = f"{emoji} {display_text}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
            text_x = (self.face_width - text_size[0]) // 2
            cv2.putText(face_display, text, (text_x, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            
            # Confidence text
            conf_text = f"Confidence: {confidence:.1f}%"
            conf_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
            conf_x = (self.face_width - conf_size[0]) // 2
            cv2.putText(face_display, conf_text, (conf_x, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        
        # Create wider info panel for better layout
        info_width = 350
        info_panel = self.create_info_panel(
            self.face_height, info_width,
            emotion, confidence, blink_count, eyes_closed,
            yaw, pitch, phone_detected, mouth_ratio, activity_label,
            activity_confidence
        )
        
        # Stack face and info panel horizontally
        display = np.hstack([face_display, info_panel])
        
        return display
