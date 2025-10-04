import cv2
import cvzone
import numpy as np

class EyeAnalyzer:
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
        leftUp, leftDown = face[159], face[23]
        leftLeft, leftRight = face[130], face[243]

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