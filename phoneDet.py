import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
from ultralytics import YOLO
import numpy as np
import time
import pygame


class DriverMonitor:
    def __init__(self, cam_index=1, yolo_model_path="yolov8n.pt", warning_sound_path="warning.mp3"):
        # Audio
        pygame.mixer.init()
        self.warning_sound = pygame.mixer.Sound(warning_sound_path)
        self.last_warning_time = 0.0  # debounce audio

        # Video
        self.cap = cv2.VideoCapture(cam_index)

        # Detectors / Models
        self.face_detector = FaceMeshDetector(maxFaces=1)
        self.plotY = LivePlot(400, 600, [25, 40])
        self.model = YOLO(yolo_model_path)

        # Eye/bink state
        self.leftEyeIdList = [22, 23, 24, 25, 26, 110, 157, 158, 159, 160, 161, 130, 243]
        self.rightEyeIdList = [252, 253, 254, 255, 256, 339, 384, 385, 386, 387, 388, 446, 463]
        self.ratioList = []
        self.blinkCounter = 0
        self.counter = 0
        self.color = (255, 0, 255)

        self.eye_closed_start_time = None
        self.eye_closed_warning_given = False

        # Head pose state
        self.head_turned = False
        self.turned_start_time = 0.0

        # Thresholds
        self.BLINK_RATIO_THRESH = 31
        self.EYE_CLOSED_SECONDS = 2
        self.HEAD_TURN_SECONDS = 3
        self.AUDIO_DEBOUNCE_SECONDS = 3

        self.YAW_THRESHOLD = 20
        self.PITCH_THRESHOLD = 20

    def play_warning(self, current_time):
        if current_time - self.last_warning_time > self.AUDIO_DEBOUNCE_SECONDS:
            self.warning_sound.play()
            self.last_warning_time = current_time

    def detect_phone(self, img, current_time):
        phone_detected = False
        results = self.model(img, stream=True, verbose=False)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.model.names[cls]
                if label and label.lower() in ['cell phone', 'phone', 'mobile phone']:
                    phone_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, f'{label} {conf:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if phone_detected:
            cv2.putText(img, 'WARNING: PHONE DETECTED!', (50, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            self.play_warning(current_time)
        return phone_detected

    def find_face(self, img):
        img, faces = self.face_detector.findFaceMesh(img, draw=False)
        return img, faces

    def analyze_eyes(self, img, face, current_time):
        # Landmarks for left eye
        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]

        # Draw eye lines
        cv2.line(img, leftUp, leftDown, (0, 200, 0), 3)
        cv2.line(img, leftLeft, leftRight, (0, 200, 0), 3)

        # Eye aspect ratio proxy (vertical/horizontal)
        lengthVer, _ = self.face_detector.findDistance(leftUp, leftDown)
        lengthHor, _ = self.face_detector.findDistance(leftLeft, leftRight)
        ratio = int((lengthVer / max(lengthHor, 1e-5)) * 100)

        # Smooth ratio
        self.ratioList.append(ratio)
        if len(self.ratioList) > 3:
            self.ratioList.pop(0)
        ratioAvg = sum(self.ratioList) / len(self.ratioList)

        # Blink and closed-eye detection
        if ratioAvg < self.BLINK_RATIO_THRESH:
            if self.counter == 0:
                self.blinkCounter += 1
                self.color = (0, 200, 0)
                self.counter = 1

            if self.eye_closed_start_time is None:
                self.eye_closed_start_time = current_time
                self.eye_closed_warning_given = False
            elif not self.eye_closed_warning_given and (current_time - self.eye_closed_start_time) > self.EYE_CLOSED_SECONDS:
                cv2.putText(img, 'WARNING: EYES CLOSED!', (50, 350),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                self.play_warning(current_time)
                self.eye_closed_warning_given = True
        else:
            self.eye_closed_start_time = None
            self.eye_closed_warning_given = False

        # Reset blink color after a short window
        if self.counter != 0:
            self.counter += 1
            if self.counter > 10:
                self.counter = 0
                self.color = (255, 0, 255)

        cvzone.putTextRect(img, f'Blink Count: {self.blinkCounter}', (50, 100), scale=2, thickness=2, colorR=self.color)

        return ratioAvg

    def head_pose_and_attention(self, img, face, current_time):
        # 2D facial landmarks corresponding to model points
        image_points = np.array([
            face[1],    # Nose tip
            face[152],  # Chin
            face[263],  # Left eye left corner
            face[33],   # Right eye right corner
            face[287],  # Left mouth corner
            face[57]    # Right mouth corner
        ], dtype="double")

        # 3D model points
        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -63.6, -12.5),
            (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0),
            (-28.9, -28.9, -24.1),
            (28.9, -28.9, -24.1)
        ])

        size = img.shape
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs
        )
        if not success:
            return

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((rotation_matrix, translation_vector))
        _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)
        pitch, yaw, roll = [angle[0] for angle in eulerAngles]

        cv2.putText(img, f'Yaw: {int(yaw)}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(img, f'Pitch: {int(pitch)}', (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Attention state
        if abs(yaw) > self.YAW_THRESHOLD or abs(pitch) > self.PITCH_THRESHOLD:
            if not self.head_turned:
                self.turned_start_time = current_time
                self.head_turned = True

            if current_time - self.turned_start_time > self.HEAD_TURN_SECONDS:
                cv2.putText(img, 'WARNING: LOOKING AWAY!', (50, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                self.play_warning(current_time)
        else:
            self.head_turned = False

    def stack_view(self, img, ratioAvg):
        imgPlot = self.plotY.update(ratioAvg, self.color)
        return cvzone.stackImages([img, imgPlot], 2, 1)

    def run(self):
        while True:
            success, img = self.cap.read()
            if not success:
                break

            current_time = time.time()

            # Phone detection
            self.detect_phone(img, current_time)

            # Face mesh and per-face analysis
            img, faces = self.find_face(img)
            if faces:
                face = faces[0]
                ratioAvg = self.analyze_eyes(img, face, current_time)
                self.head_pose_and_attention(img, face, current_time)
                imgStack = self.stack_view(img, ratioAvg)
            else:
                imgStack = cvzone.stackImages([img, img], 2, 1)

            cv2.imshow("Driver Monitoring", imgStack)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = DriverMonitor(
        cam_index=0,
        yolo_model_path="yolov8n.pt",
        warning_sound_path="warning.mp3",
    )
    app.run()