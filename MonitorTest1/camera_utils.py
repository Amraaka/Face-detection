import cv2
import numpy as np
from typing import Union

__all__ = ["open_capture", "list_cameras", "MjpegStreamCapture"]  

def _open_url_with_opencv(url: str):
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    return None

class MjpegStreamCapture:
    def __init__(self, url: str, chunk_size: int = 16384, timeout: int = 10):
        try:
            import requests  
        except ImportError as e:
            raise RuntimeError("requests not installed. Install with: pip install requests") from e

        self._requests = requests
        self._resp = requests.get(url, stream=True, timeout=timeout)
        self._iter = self._resp.iter_content(chunk_size=chunk_size)
        self._buffer = bytearray()
        self._closed = False

    def read(self):
        if self._closed:
            return False, None
        SOI = b"\xff\xd8"
        EOI = b"\xff\xd9"
        try:
            while True:
                soi = self._buffer.find(SOI)
                eoi = self._buffer.find(EOI)
                if soi != -1 and eoi != -1 and eoi > soi:
                    jpg = bytes(self._buffer[soi:eoi + 2])
                    del self._buffer[:eoi + 2]
                    arr = np.frombuffer(jpg, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    return True, frame
                chunk = next(self._iter, None)
                if chunk is None:
                    return False, None
                self._buffer.extend(chunk)
        except Exception:
            return False, None

    def isOpened(self):
        return not self._closed

    def release(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._resp.close()
        except Exception:
            pass

def open_capture(source: Union[int, str]):
    if isinstance(source, str):
        cap = cv2.VideoCapture(source)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    # Local camera index (macOS: prefer AVFoundation)
    cap = cv2.VideoCapture(int(source), cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(int(source)) 
    return cap

def list_cameras(max_index: int = 8):
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(i)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            found.append((i, width, height))
            cap.release()
    return found