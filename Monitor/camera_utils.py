import cv2
import numpy as np
from typing import Union

__all__ = ["open_capture", "list_cameras", "MjpegStreamCapture"]  # ensure names are exported

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
    """
    Minimal MJPEG reader for URLs like http://IP:8080/video (IP Webcam).
    Provides a VideoCapture-like API: read() -> (ok, frame), release().
    """
    def __init__(self, url: str, chunk_size: int = 16384, timeout: int = 10):
        try:
            import requests  # lazy import
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
        # Find JPEG SOI/EOI markers in the buffer
        SOI = b"\xff\xd8"
        EOI = b"\xff\xd9"
        try:
            while True:
                soi = self._buffer.find(SOI)
                eoi = self._buffer.find(EOI)
                if soi != -1 and eoi != -1 and eoi > soi:
                    jpg = bytes(self._buffer[soi:eoi + 2])
                    # Trim consumed bytes
                    del self._buffer[:eoi + 2]
                    arr = np.frombuffer(jpg, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        # keep reading if decode failed
                        continue
                    return True, frame
                # Need more data
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
    """
    source: camera index (int) or network URL (str, e.g. http://IP:8080/video or rtsp://...)
    """
    if isinstance(source, str):
        # Simplified URL handling to match user's working sample
        cap = cv2.VideoCapture(source)
        # Reduce latency if supported
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    # Local camera index (macOS: prefer AVFoundation)
    cap = cv2.VideoCapture(int(source), cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(int(source))  # fallback
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