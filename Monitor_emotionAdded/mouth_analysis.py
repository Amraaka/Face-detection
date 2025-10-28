import cv2
import numpy as np

class MouthAnalyzer:
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