import cv2
import cvzone
import time
import argparse
import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    load_dotenv(env_path)
except ImportError:
    print("[info] python-dotenv not installed. Using system env vars only.")

from .alerts import AlertPlayer
from .phone_detection import PhoneDetector
from .face_mesh import FaceMeshService
from .eye_analysis import EyeAnalyzer
from .head_pose import HeadPoseEstimator
from .view_stack import ViewStacker
from .mouth_analysis import MouthAnalyzer
from . import camera_utils
from .stats import StatsAggregator
from .stats_store import StatsDB

# Defensive fallbacks if attributes are not exposed on some Python setups
def _fallback_open_capture(source):
    # Basic fallback without MJPEG parser
    if isinstance(source, str):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        return cap
    cap = cv2.VideoCapture(int(source), cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(int(source))
    return cap

def _fallback_list_cameras(max_index: int = 8):
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

class DriverMonitor:
    def __init__(self, cam_index=0, yolo_model_path="yolov8n.pt", warning_sound_path="warning.mp3", url=None,
                 stats_interval: float = 5.0, mongo_uri: Optional[str] = None, mongo_db: str = "driver_monitor",
                 mongo_coll: str = "stats"):
        # Open local cam or network stream
        # Use camera_utils if available, otherwise fallback
        if hasattr(camera_utils, "open_capture"):
            self.cap = camera_utils.open_capture(url if url else cam_index)
        else:
            self.cap = _fallback_open_capture(url if url else cam_index)

        self.alerts = AlertPlayer(warning_sound_path=warning_sound_path, debounce_seconds=3.0)
        self.phone = PhoneDetector(model_path=yolo_model_path)
        self.face_mesh = FaceMeshService(max_faces=1)
        self.eyes = EyeAnalyzer(blink_ratio_thresh=31, eye_closed_seconds=2)
        self.head = HeadPoseEstimator(yaw_threshold=20, pitch_threshold=20, hold_seconds=3)
        self.mouth = MouthAnalyzer()
        self.view = ViewStacker(width=400, height=600, y_range=(25, 40))
        # Stats aggregation and optional DB
        self.stats = StatsAggregator(interval_seconds=stats_interval)
        self.stats_db = None
        if mongo_uri:
            try:
                self.stats_db = StatsDB(mongo_uri, db_name=mongo_db, coll_name=mongo_coll)
                print("[info] Connected to MongoDB Atlas")
            except Exception as e:
                print(f"[warn] Mongo connection failed: {e}. Proceeding without DB.")

    def run(self):
        while True:
            ok, img = self.cap.read()
            if not ok:
                # For network streams allow retry instead of quitting
                print("[warn] Failed to grab frame; retrying...", flush=True)
                time.sleep(0.05)
                continue

            now = time.time()

            phone_detected = self.phone.detect(img)
            if phone_detected:
                self.alerts.play(now)

            img, faces = self.face_mesh.find_face(img)
            if faces:
                face = faces[0]
                ratioAvg, color, eyes_closed = self.eyes.analyze(img, face, now, on_warning=self.alerts.play)
                yaw, pitch = self.head.check(img, face, now, on_warning=self.alerts.play)
                mouth_ratio = self.mouth.analyze_mouth(face)
                
                # update stats with all behavioral data
                self.stats.update(now, yaw=yaw, pitch=pitch, phone_detected=phone_detected,
                                  blink_total=self.eyes.blinkCounter, eyes_closed=eyes_closed,
                                  mouth_open_ratio=mouth_ratio)
                imgStack = self.view.stack(img, ratioAvg, color)
            else:
                # update stats with available signals
                self.stats.update(now, yaw=None, pitch=None, phone_detected=phone_detected,
                                  blink_total=self.eyes.blinkCounter, eyes_closed=False,
                                  mouth_open_ratio=None)
                imgStack = cvzone.stackImages([img, img], 2, 1)

            # Flush to DB every interval
            if self.stats_db and self.stats.ready(now):
                doc = self.stats.flush(now)
                self.stats_db.insert(doc)

            cv2.imshow("Driver Monitoring", imgStack)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=None, help="Camera index (e.g. 0,1,2...)")
    parser.add_argument("--url", type=str, default=None, help="HTTP MJPEG or RTSP URL (e.g. http://PHONE_IP:8080/video)")
    parser.add_argument("--list-cams", action="store_true", help="List available camera indices and exit")
    parser.add_argument("--yolo", default="yolov8n.pt")
    parser.add_argument("--sound", default="warning.mp3")
    parser.add_argument("--mongo-uri", type=str, default=os.getenv("MONGODB_URI"), help="MongoDB Atlas connection string")
    parser.add_argument("--mongo-db", type=str, default=os.getenv("MONGODB_DB", "driver_monitor"))
    parser.add_argument("--mongo-coll", type=str, default=os.getenv("MONGODB_COLL", "stats"))
    parser.add_argument("--stats-interval", type=float, default=float(os.getenv("STATS_INTERVAL", "5")), help="seconds")
    args = parser.parse_args()

    if args.list_cams:
        cams = camera_utils.list_cameras() if hasattr(camera_utils, "list_cameras") else _fallback_list_cameras()
        for idx, w, h in cams:
            print(f"Camera {idx}: {w}x{h}")
        raise SystemExit(0)

    cam_index = args.cam if args.cam is not None else int(os.getenv("CAM_INDEX", "0"))

    if args.url:
        print(f"Using network stream: {args.url}")
    else:
        print(f"Using camera index: {cam_index}")

    app = DriverMonitor(
        cam_index=cam_index,
        yolo_model_path=args.yolo,
        warning_sound_path=args.sound,
        url=args.url,
        stats_interval=args.stats_interval,
        mongo_uri=args.mongo_uri,
        mongo_db=args.mongo_db,
        mongo_coll=args.mongo_coll,
    )
    app.run()

    # Example:
# .venv/bin/python -m Monitor.main --list-cams
# .venv/bin/python -m Monitor.main --cam 0
# .venv/bin/python -m Monitor.main --url http://172.20.10.3:8080/video
