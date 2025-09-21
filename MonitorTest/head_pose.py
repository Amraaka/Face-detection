import cv2
import numpy as np

class HeadPoseEstimator:
    def __init__(self, yaw_threshold=20, pitch_threshold=20, hold_seconds=3):
        self.YAW_THRESHOLD = yaw_threshold
        self.PITCH_THRESHOLD = pitch_threshold
        self.HEAD_TURN_SECONDS = hold_seconds
        self.head_turned = False
        self.turned_start_time = 0.0

    def check(self, img, face, now, on_warning=None):
        image_points = np.array([
            face[1], face[152], face[263], face[33], face[287], face[57]
        ], dtype="double")

        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -63.6, -12.5),
            (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0),
            (-28.9, -28.9, -24.1),
            (28.9, -28.9, -24.1)
        ])

        h, w = img.shape[:2]
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        dist_coeffs = np.zeros((4, 1))

        success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
        if not success:
            return None, None

        rmat, _ = cv2.Rodrigues(rvec)
        proj = np.hstack((rmat, tvec))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj)
        pitch, yaw, roll = [angle[0] for angle in euler]

        cv2.putText(img, f"Yaw: {int(yaw)}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(img, f"Pitch: {int(pitch)}", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if abs(yaw) > self.YAW_THRESHOLD or abs(pitch) > self.PITCH_THRESHOLD:
            if not self.head_turned:
                self.turned_start_time = now
                self.head_turned = True
            if now - self.turned_start_time > self.HEAD_TURN_SECONDS:
                cv2.putText(img, "WARNING: LOOKING AWAY!", (50, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                if on_warning:
                    on_warning(now)
        else:
            self.head_turned = False

        return yaw, pitch