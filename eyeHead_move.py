import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import numpy as np
import time

# cap = cv2.VideoCapture('/Users/amara/SideProjects/Research/Eye_Blink_Detection/Blinking_Video.mp4')
cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(400, 600, [25, 40])

ratioList = []
blinkCounter = 0
counter = 0
color = (255, 0, 255)

# For distraction detection
head_turned = False
turned_start_time = 0

# Eye landmark indices
leftEyeIdList = [22, 23, 24, 25, 26, 110, 157, 158, 159, 160, 161, 130, 243]
rightEyeIdList = [252, 253, 254, 255, 256, 339, 384, 385, 386, 387, 388, 446, 463]

while True:
    success, img = cap.read()
    if not success:
        break

    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]

        # --- Draw eye landmarks with green dots ---
        # Left eye landmarks
        for idx in leftEyeIdList:
            pt = tuple(map(int, face[idx][:2]))
            cv2.circle(img, pt, 3, (0, 255, 0), -1)  # Green dots
        
        # Right eye landmarks
        for idx in rightEyeIdList:
            pt = tuple(map(int, face[idx][:2]))
            cv2.circle(img, pt, 3, (0, 255, 0), -1)  # Green dots
    
        # --- Draw mouth landmarks with green dots ---
        mouth_landmarks = [0, 267, 37, 39, 40 , 185, 17, 181, 84, 91, 146, 314, 405, 321, 375, 269, 270, 409, 76, 308, 82, 13, 81, 80, 14, 87, 178, 88, 402]
        for idx in mouth_landmarks:
            pt = tuple(map(int, face[idx][:2]))
            cv2.circle(img, pt, 3, (0, 255, 0), -1)  # Green dots

        # --- EAR calculation for left eye ---
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

        # --- Head pose estimation ---
        image_points = np.array([
            face[1],    # Nose tip
            face[152],  # Chin
            face[263],  # Left eye left corner
            face[33],   # Right eye right corner
            face[287],  # Left mouth corner
            face[57]    # Right mouth corner
        ], dtype="double")

        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -63.6, -12.5),         # Chin
            (-43.3, 32.7, -26.0),        # Left eye left corner
            (43.3, 32.7, -26.0),         # Right eye right corner
            (-28.9, -28.9, -24.1),       # Left mouth corner
            (28.9, -28.9, -24.1)         # Right mouth corner
        ])

        size = img.shape
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs)

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((rotation_matrix, translation_vector))
        _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)

        pitch, yaw, roll = [angle[0] for angle in eulerAngles]

        cv2.putText(img, f'Yaw: {int(yaw)}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(img, f'Pitch: {int(pitch)}', (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        YAW_THRESHOLD = 20  # degrees
        PITCH_THRESHOLD = 20  # degrees

        if abs(yaw) > YAW_THRESHOLD or abs(pitch) > PITCH_THRESHOLD:
            if not head_turned:
                turned_start_time = time.time()
                head_turned = True

            if time.time() - turned_start_time > 5:
                cv2.putText(img, 'WARNING: LOOKING AWAY!', (50, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            head_turned = False

        imgPlot = plotY.update(ratioAvg, color)
        imgStack =  cvzone.stackImages([img, imgPlot], 2, 1)

    else:
        imgStack = cvzone.stackImages([img, img], 2, 1)

    cv2.imshow("Driver Monitoring", imgStack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
