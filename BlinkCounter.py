import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

# cap = cv2.VideoCapture('/Users/amara/SideProjects/Research/Eye_Blink_Detection/Blinking_Video.mp4')
cap = cv2.VideoCapture(2)
detector = FaceMeshDetector(maxFaces=2)
plotY = LivePlot(400, 600, [25, 40])  
ratioList=[]
blinkCounter = 0
counter = 0 
color = (255, 0 , 255)

leftEyeIdList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
# Right eye landmark indices (MediaPipe Face Mesh)
rightEyeIdList = [ 252, 253, 254, 255,  256, 339,  384 ,385, 386, 387, 388 ,446, 463]


while True:
    # if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        for id in leftEyeIdList:
            cv2.circle(img, face[id], 5, color, cv2.FILLED)
        for id in rightEyeIdList:
            cv2.circle(img, face[id], 5, color, cv2.FILLED)
        # for idx, point in enumerate(face):
        #     cv2.circle(img, point, 2, (255, 255, 255), -1)
        #     cv2.putText(img, str(idx), point, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)

        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]
        lengthHor, _ = detector.findDistance(leftUp, leftDown)
        lengthVer, _ = detector.findDistance(leftLeft, leftRight)
        cv2.line(img, leftUp, leftDown, (0, 200, 0), 3)
        cv2.line(img, leftLeft, leftRight, (0, 200, 0), 3)
        # print(lengthHor) # 36
        # print(int((lengthHor/lengthVer)*100)) # 30
        ratio = int((lengthHor/lengthVer)*100)
        ratioList.append(ratio)
        if len(ratioList) > 3: 
            ratioList.pop(0)
        
        ratioAvg = sum(ratioList)/len(ratioList)

        if ratioAvg < 31 and counter ==0:
            blinkCounter +=1
            color = (0, 200, 0)  
            counter = 1
        if counter !=0: 
            counter += 1
            if counter > 10:
                counter = 0
                color = (255, 0, 255)
        
        cvzone.putTextRect(img, f'Blink Count {blinkCounter}', (100, 100), colorR=color)

        imgPlot = plotY.update(ratioAvg, color)
        # img = cv2.resize(img, (440, 740))
        # cv2.imshow("ImagePlot", imgPlot)
        # cv2.moveWindow("ImagePlot", 100, 100)
        imgStack = cvzone.stackImages([img, imgPlot], 2, 1)
    else: 
        # img = cv2.resize(img,(440, 740))
        imgStack = cvzone.stackImages([img, img], 2, 1)
    

    # img = cv2.resize(img, (440, 740))
    cv2.imshow("image", imgStack)
    cv2.moveWindow("image", 100, 0)
    cv2.waitKey(20)