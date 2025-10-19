"""
Eye and Mouth Mesh Detection Demo
Demonstrates facial landmark detection focusing on eyes and mouth
Uses EXACT implementation from MonitorTest1
"""

import cv2
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector
import cvzone
import time


class EyeAnalyzer:
    """
    EXACT implementation from MonitorTest1/eye_analysis.py
    Analyzes eye movements and detects blinks
    """
    def __init__(self, blink_ratio_thresh=31, eye_closed_seconds=2):
        self.BLINK_RATIO_THRESH = blink_ratio_thresh
        self.EYE_CLOSED_SECONDS = eye_closed_seconds
        self.ratioList = []
        self.blinkCounter = 0
        self.counter = 0
        self.color = (255, 0, 255)
        self.eye_closed_start_time = None
        self.eye_closed_warning_given = False

    @staticmethod
    def _dist(p1, p2):
        p1 = np.array(p1, dtype=float)
        p2 = np.array(p2, dtype=float)
        return float(np.linalg.norm(p1 - p2))

    def analyze(self, img, face, now, on_warning=None):
        """
        EXACT method from MonitorTest1
        Analyzes eye state and detects blinks
        """
        # Get eye landmarks (left eye only in MonitorTest1)
        leftUp, leftDown = face[159], face[23]
        leftLeft, leftRight = face[130], face[243]

        # Draw eye measurement lines (as in MonitorTest1)
        cv2.line(img, leftUp, leftDown, (0, 200, 0), 3)
        cv2.line(img, leftLeft, leftRight, (0, 200, 0), 3)

        lengthVer = self._dist(leftUp, leftDown)
        lengthHor = max(self._dist(leftLeft, leftRight), 1e-5)
        ratio = int((lengthVer / lengthHor) * 100)

        self.ratioList.append(ratio)
        if len(self.ratioList) > 3:
            self.ratioList.pop(0)
        ratioAvg = sum(self.ratioList) / len(self.ratioList)

        # Determine if eyes are closed
        eyes_closed = ratioAvg < self.BLINK_RATIO_THRESH

        # Blink/closed detection
        if eyes_closed:
            if self.counter == 0:
                self.blinkCounter += 1
                self.color = (0, 200, 0)
                self.counter = 1

            if self.eye_closed_start_time is None:
                self.eye_closed_start_time = now
                self.eye_closed_warning_given = False
            elif not self.eye_closed_warning_given and (now - self.eye_closed_start_time) > self.EYE_CLOSED_SECONDS:
                cv2.putText(img, "WARNING: EYES CLOSED!", (50, 350),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                if on_warning:
                    on_warning(now)
                self.eye_closed_warning_given = True
        else:
            self.eye_closed_start_time = None
            self.eye_closed_warning_given = False

        if self.counter != 0:
            self.counter += 1
            if self.counter > 10:
                self.counter = 0
                self.color = (255, 0, 255)

        cvzone.putTextRect(img, f"Blink Count: {self.blinkCounter}", (50, 100), scale=2, thickness=2, colorR=self.color)
        return ratioAvg, self.color, eyes_closed


class MouthAnalyzer:
    """
    EXACT implementation from MonitorTest1/mouth_analysis.py
    Analyzes mouth opening for yawn detection
    """
    def __init__(self):
        # Mouth landmarks for yawn detection
        self.mouth_landmarks = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
        
    @staticmethod
    def _dist(p1, p2):
        p1 = np.array(p1, dtype=float)
        p2 = np.array(p2, dtype=float)
        return float(np.linalg.norm(p1 - p2))
    
    def analyze_mouth(self, face):
        """
        Analyze mouth opening ratio to detect yawning
        Returns mouth opening ratio (higher values indicate wider mouth opening)
        """
        try:
            # Get mouth corner points
            mouth_left = face[61]
            mouth_right = face[291]
            mouth_top = face[13]
            mouth_bottom = face[14]
            
            # Calculate mouth dimensions
            mouth_width = self._dist(mouth_left, mouth_right)
            mouth_height = self._dist(mouth_top, mouth_bottom)
            
            # Calculate mouth opening ratio
            if mouth_width > 0:
                mouth_ratio = (mouth_height / mouth_width) * 100
            else:
                mouth_ratio = 0
                
            return mouth_ratio
        except (IndexError, KeyError):
            # If landmarks are not available, return 0
            return 0


class EyeMouthDemo:
    """Demo wrapper to combine eye and mouth analysis"""
    
    def __init__(self):
        self.face_detector = FaceMeshDetector(maxFaces=1)
        self.eye_analyzer = EyeAnalyzer(blink_ratio_thresh=31, eye_closed_seconds=2)
        self.mouth_analyzer = MouthAnalyzer()
        
        # Eye landmarks for visualization
        self.left_eye_landmarks = [159, 23, 130, 243]
        self.right_eye_landmarks = [386, 374, 359, 263]
        
        # Mouth landmarks for visualization
        self.mouth_key_landmarks = [61, 291, 13, 14]
        
    def draw_eye_dots(self, frame, face):
        """Draw prominent dots on key eye landmarks"""
        for idx in self.left_eye_landmarks:
            pt = tuple(map(int, face[idx][:2]))
            cv2.circle(frame, pt, 8, (0, 255, 255), -1)
            cv2.circle(frame, pt, 10, (255, 255, 255), 2)
        
        for idx in self.right_eye_landmarks:
            pt = tuple(map(int, face[idx][:2]))
            cv2.circle(frame, pt, 8, (0, 255, 255), -1)
            cv2.circle(frame, pt, 10, (255, 255, 255), 2)
    
    def draw_mouth_dots(self, frame, face):
        """Draw prominent dots on key mouth landmarks"""
        for idx in self.mouth_analyzer.mouth_landmarks:
            pt = tuple(map(int, face[idx][:2]))
            cv2.circle(frame, pt, 6, (255, 0, 255), -1)
            cv2.circle(frame, pt, 8, (255, 255, 255), 2)
        
        for idx in self.mouth_key_landmarks:
            pt = tuple(map(int, face[idx][:2]))
            cv2.circle(frame, pt, 10, (0, 0, 255), -1)
            cv2.circle(frame, pt, 12, (255, 255, 255), 2)
    
    def process_frame(self, frame, now):
        """Process frame with MonitorTest1 implementation"""
        frame, faces = self.face_detector.findFaceMesh(frame, draw=False)
        
        if not faces:
            return frame, None
        
        face = faces[0]
        
        eye_ratio, eye_color, eyes_closed = self.eye_analyzer.analyze(frame, face, now)
        mouth_ratio = self.mouth_analyzer.analyze_mouth(face)
        
        self.draw_eye_dots(frame, face)
        self.draw_mouth_dots(frame, face)
        
        results = {
            'eye_ratio': eye_ratio,
            'eye_color': eye_color,
            'eyes_closed': eyes_closed,
            'mouth_ratio': mouth_ratio,
            'blink_count': self.eye_analyzer.blinkCounter
        }
        
        return frame, results


def main():
    """Main function to run the demo"""
    print("=" * 80)
    print("Eye and Mouth Detection Demo - MonitorTest1 Implementation")
    print("=" * 80)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    demo = EyeMouthDemo()
    prev_time = time.time()
    
    print("\nStarting camera feed...")
    print("Press 'q' to quit, 'r' to reset blink counter\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        curr_time = time.time()
        
        frame, results = demo.process_frame(frame, curr_time)
        
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Eye & Mouth Detection - MonitorTest1", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            demo.eye_analyzer.blinkCounter = 0
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nTotal blinks: {demo.eye_analyzer.blinkCounter}")


if __name__ == "__main__":
    main()
