import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import time

# Video input
cap = cv2.VideoCapture('/Users/amara/SideProjects/Research/Eye_Blink_Detection/Blinking_Video.mp4')
# cap = cv2.VideoCapture(0)  # For webcam

detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(400, 600, [0, 50])

# Lists for smoothing
ratioList = []
blinkCounter = 0
counter = 0
color = (255, 0, 255)

# Time tracking
eyeClosed = False
closedStartTime = 0

# Eye landmark IDs
leftEye = [33, 160, 158, 133, 153, 144]  # P1, P2, P3, P4, P5, P6
rightEye = [362, 385, 387, 263, 373, 380]  # P1, P2, P3, P4, P5, P6

def findEAR(face, eye):
    # Compute EAR using 6 key landmarks
    # Vertical distances
    A = detector.findDistance(face[eye[1]], face[eye[5]])[0]
    B = detector.findDistance(face[eye[2]], face[eye[4]])[0]
    # Horizontal distance
    C = detector.findDistance(face[eye[0]], face[eye[3]])[0]
    ear = (A + B) / (2.0 * C)
    return ear

while True:
    success, img = cap.read()
    if not success:
        break

    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]

        # Draw landmarks (optional)
        for id in leftEye + rightEye:
            cv2.circle(img, face[id], 3, (0, 0, 255), cv2.FILLED)

        leftEAR = findEAR(face, leftEye)
        rightEAR = findEAR(face, rightEye)

        avgEAR = (leftEAR + rightEAR) / 2.0
        ratioList.append(avgEAR)
        if len(ratioList) > 5:
            ratioList.pop(0)

        ratioAvg = sum(ratioList) / len(ratioList)

        # Threshold for blink detection
        EAR_THRESHOLD = 0.22

        if ratioAvg < EAR_THRESHOLD:
            if not eyeClosed:
                closedStartTime = time.time()
                eyeClosed = True
            duration = time.time() - closedStartTime

            if duration >= 3:
                cv2.putText(img, 'WARNING: EYES CLOSED!', (50, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            color = (0, 200, 0)
        else:
            if eyeClosed:
                blinkCounter += 1
            eyeClosed = False
            color = (255, 0, 255)

        cvzone.putTextRect(img, f'Blink Count: {blinkCounter}', (50, 100), scale=2, thickness=2, colorR=color)
        imgPlot = plotY.update(ratioAvg * 100, color)
        imgStack = cvzone.stackImages([img, imgPlot], 2, 1)
    else:
        imgStack = cvzone.stackImages([img, img], 2, 1)

    cv2.imshow("Eye Blink Detection", imgStack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
