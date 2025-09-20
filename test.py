import cv2

url = 'http://172.20.10.3:8080/video'

# Open the video stream
cap = cv2.VideoCapture(url)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # If the frame was not retrieved properly, break the loop
    if not ret:
        print("Failed to grab frame")
        break
    
    # Display the resulting frame
    cv2.imshow('Mobile Camera', frame)
    
    # Press 'q' to exit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()