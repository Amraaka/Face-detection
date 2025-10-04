import cv2
import urllib.request
import numpy as np

url = "http://10.1.0.33:8080/stream"  # change this

stream = urllib.request.urlopen(url)
bytes_data = b''
while True:
    bytes_data += stream.read(1024)
    a = bytes_data.find(b'\xff\xd8')  # JPEG start
    b = bytes_data.find(b'\xff\xd9')  # JPEG end
    if a != -1 and b != -1:
        jpg = bytes_data[a:b+2]
        bytes_data = bytes_data[b+2:]
        img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            cv2.imshow('iPhone MJPEG', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
cv2.destroyAllWindows()
