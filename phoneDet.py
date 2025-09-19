import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
from ultralytics import YOLO
import numpy as np
import time
import pygame  

pygame.mixer.init()

warning_sound = pygame.mixer.Sound('warning.mp3')  

# cap = cv2.VideoCapture("https://172.20.10:8080/video")
cap = cv2.VideoCapture(1)
# cap = cv2.VideoCapture('/Users/amara/SideProjects/Research/Eye_Blink_Detection/Blinking_Video.mp4')

detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(400, 600, [25, 40])

model = YOLO("yolov8n.pt")  

ratioList = []
blinkCounter = 0
counter = 0
color = (255, 0, 255)

head_turned = False
turned_start_time = 0
last_warning_time = 0  # To prevent audio spamming

leftEyeIdList = [22, 23, 24, 25, 26, 110, 157, 158, 159, 160, 161, 130, 243]
rightEyeIdList = [252, 253, 254, 255, 256, 339, 384, 385, 386, 387, 388, 446, 463]

eye_closed_start_time = None
eye_closed_warning_given = False

while True:
    success, img = cap.read()
    if not success:
        break
        
    current_time = time.time()
    
    results = model(img, stream=True, verbose=False)

    phone_detected = False
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            if label.lower() in ['cell phone', 'phone', 'mobile phone']:  # depends on the model's labels
                phone_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, f'{label} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if phone_detected:
        cv2.putText(img, 'WARNING: PHONE DETECTED!', (50, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        if current_time - last_warning_time > 3:
            warning_sound.play()
            last_warning_time = current_time

    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]

        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]
        lengthVer, _ = detector.findDistance(leftUp, leftDown)
        lengthHor, _ = detector.findDistance(leftLeft, leftRight)
        cv2.line(img, leftUp, leftDown, (0, 200, 0), 3)
        cv2.line(img, leftLeft, leftRight, (0, 200, 0), 3)

        ratio = int((lengthVer / lengthHor) * 100)
        ratioList.append(ratio)
        if len(ratioList) > 3:
            ratioList.pop(0)
        ratioAvg = sum(ratioList) / len(ratioList)

        if ratioAvg < 31:
            if counter == 0:
                blinkCounter += 1
                color = (0, 200, 0)
                counter = 1
            if eye_closed_start_time is None:
                eye_closed_start_time = current_time
                eye_closed_warning_given = False
            elif not eye_closed_warning_given and current_time - eye_closed_start_time > 2:
                cv2.putText(img, 'WARNING: EYES CLOSED!', (50, 350),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                warning_sound.play()
                eye_closed_warning_given = True
        else:
            eye_closed_start_time = None
            eye_closed_warning_given = False

        if len(ratioList) > 3:
            ratioList.pop(0)
        ratioAvg = sum(ratioList) / len(ratioList)

        if ratioAvg < 31 and counter == 0:
            blinkCounter += 1
            color = (0, 200, 0)
            counter = 1
        if counter != 0:
            counter += 1
            if counter > 10:
                counter = 0
                color = (255, 0, 255)

        cvzone.putTextRect(img, f'Blink Count: {blinkCounter}', (50, 100), scale=2, thickness=2, colorR=color)

        image_points = np.array([
            face[1],    # Nose tip
            face[152],  # Chin
            face[263],  # Left eye left corner
            face[33],   # Right eye right corner
            face[287],  # Left mouth corner
            face[57]    # Right mouth corner
        ], dtype="double")

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
            model_points, image_points, camera_matrix, dist_coeffs)

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((rotation_matrix, translation_vector))
        _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)

        pitch, yaw, roll = [angle[0] for angle in eulerAngles]

        cv2.putText(img, f'Yaw: {int(yaw)}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(img, f'Pitch: {int(pitch)}', (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        YAW_THRESHOLD = 20
        PITCH_THRESHOLD = 20

        if abs(yaw) > YAW_THRESHOLD or abs(pitch) > PITCH_THRESHOLD:
            if not head_turned:
                turned_start_time = time.time()
                head_turned = True

            if time.time() - turned_start_time > 3:
                cv2.putText(img, 'WARNING: LOOKING AWAY!', (50, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                if current_time - last_warning_time > 3:
                    warning_sound.play()
                    last_warning_time = current_time
        else:
            head_turned = False

        imgPlot = plotY.update(ratioAvg, color)
        imgStack = cvzone.stackImages([img, imgPlot], 2, 1)

    else:
        imgStack = cvzone.stackImages([img, img], 2, 1)

    cv2.imshow("Driver Monitoring", imgStack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()