# EAR = ||P2 - P6|| + ||P3 - P5|| / 2 || P1 - P4||
# 468 landmarks in 3d 
import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
#https://learnopencv.com/driver-drowsiness-detection-using-mediapipe-in-python/

# The chosen 12 points:   P1,  P2,  P3,  P4,  P5,  P6
chosen_left_eye_idxs  = [362, 385, 387, 263, 373, 380]
chosen_right_eye_idxs = [33,  160, 158, 133, 153, 144]
all_chosen_idxs = chosen_left_eye_idxs + chosen_right_eye_idxs

image = cv2.imread(r"test-open-eyes.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert to RGB
imgH, imgW, _ = image.shape
