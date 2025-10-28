import cv2
import cvzone
import time
import argparse
import os
from pathlib import Path
from typing import Optional
import uuid
   
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
from .emotion_detection import EmotionDetector
from .face_view import FaceOnlyView

def _fallback_open_capture(source):
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
                 mongo_coll: str = "stats", face_only_view: bool = True, enable_emotion: bool = True):
        if hasattr(camera_utils, "open_capture"):
            self.cap = camera_utils.open_capture(url if url else cam_index)
        else:
            self.cap = _fallback_open_capture(url if url else cam_index)

        if not (hasattr(self.cap, "isOpened") and self.cap.isOpened()):
            print("[warn] Primary camera open failed. Scanning for available cameras...", flush=True)
            cams = []
            if hasattr(camera_utils, "list_cameras"):
                cams = camera_utils.list_cameras()
            else:
                cams = _fallback_list_cameras()
            if cams:
                fallback_idx = cams[0][0]
                print(f"[info] Falling back to camera index: {fallback_idx}")
                if hasattr(camera_utils, "open_capture"):
                    self.cap = camera_utils.open_capture(fallback_idx)
                else:
                    self.cap = _fallback_open_capture(fallback_idx)
            else:
                print("[error] No cameras detected. If you intended to use a URL, pass --url.")

        self.alerts = AlertPlayer(warning_sound_path=warning_sound_path, debounce_seconds=3.0)
        self.phone = PhoneDetector(model_path=yolo_model_path)
        self.face_mesh = FaceMeshService(max_faces=1)
        self.eyes = EyeAnalyzer(blink_ratio_thresh=31, eye_closed_seconds=2)
        self.head = HeadPoseEstimator(yaw_threshold=20, pitch_threshold=20, hold_seconds=3)
        self.mouth = MouthAnalyzer()
        self.view = ViewStacker(width=400, height=600, y_range=(25, 40))
        
        self.enable_emotion = enable_emotion
        if enable_emotion:
            self.emotion = EmotionDetector()
        else:
            self.emotion = None
        
        self.face_only_view = face_only_view
        if face_only_view:
            self.face_view = FaceOnlyView(face_size=(480, 640))
        else:
            self.face_view = None
        self.window_title = "Driver Monitoring - Face View" if self.face_only_view else "Driver Monitoring - Full Frame"
        
        session_id = str(uuid.uuid4())
        self.stats = StatsAggregator(interval_seconds=stats_interval, session_id=session_id)
        self.stats_db = None
        if mongo_uri:
            try:
                self.stats_db = StatsDB(mongo_uri, db_name=mongo_db, coll_name=mongo_coll)
                print("[info] Connected to MongoDB Atlas")
            except Exception as e:
                print(f"[warn] Mongo connection failed: {e}. Proceeding without DB.")

    def run(self):
        cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)
        try:
            cv2.resizeWindow(self.window_title, 1280, 720)
        except Exception:
            pass

        while True:
            ok, img = self.cap.read()
            if not ok:
                print("[warn] Failed to grab frame; retrying...", flush=True)
                time.sleep(0.05)
                continue

            now = time.time()

            phone_detected = self.phone.detect(img)
            if phone_detected:
                self.alerts.play(now)

            img, faces = self.face_mesh.find_face(img)
            
            emotion = None
            emotion_confidence = 0
            emotion_box = None
            ratioAvg = 0
            color = (0, 255, 0)
            eyes_closed = False
            yaw, pitch = None, None
            mouth_ratio = None
            
            if faces:
                face = faces[0]
                
                ratioAvg, color, eyes_closed = self.eyes.analyze(img, face, now, on_warning=self.alerts.play)
                
                yaw, pitch = self.head.check(img, face, now, on_warning=self.alerts.play)
                
                mouth_ratio = self.mouth.analyze_mouth(face)
                
                if self.enable_emotion and self.emotion:
                    try:
                        face_points = face 
                        if len(face_points) > 0:
                            xs = [int(p[0]) for p in face_points]
                            ys = [int(p[1]) for p in face_points]
                            x_min, x_max = max(0, min(xs)), min(img.shape[1] - 1, max(xs))
                            y_min, y_max = max(0, min(ys)), min(img.shape[0] - 1, max(ys))
                            w = max(1, x_max - x_min)
                            h = max(1, y_max - y_min)
                            emotion_box = (x_min, y_min, w, h)
                            emotion, emotion_confidence, emotion_box = self.emotion.detect_emotion(
                                img, face_roi=emotion_box
                            )
                            
                            if emotion and emotion_box:
                                x, y, w, h = emotion_box
                                cv2.putText(img, f"{emotion} ({emotion_confidence:.1f}%)", (x, y - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    except Exception as e:
                        print(f"[warn] Emotion detection failed: {e}")
                
                self.stats.update(now, yaw=yaw, pitch=pitch, phone_detected=phone_detected,
                                  blink_total=self.eyes.blinkCounter, eyes_closed=eyes_closed,
                                  mouth_open_ratio=mouth_ratio)
                
                if self.face_only_view and self.face_view:
                    # Extract face region and create face-only view
                    face_img = self.face_view.extract_face_region(img, face, emotion_box)
                    imgStack = self.face_view.create_display(
                        face_img, emotion, emotion_confidence,
                        self.eyes.blinkCounter, eyes_closed, yaw, pitch,
                        phone_detected, mouth_ratio
                    )
                else:
                    # Full-frame view: show the original camera frame (big), with overlays drawn directly on it
                    imgStack = img
            else:
                # No face detected
                # update stats with available signals
                self.stats.update(now, yaw=None, pitch=None, phone_detected=phone_detected,
                                  blink_total=self.eyes.blinkCounter, eyes_closed=False,
                                  mouth_open_ratio=None)
                
                if self.face_only_view and self.face_view:
                    # Show full frame with info panel
                    face_img = self.face_view.extract_face_region(img)
                    imgStack = self.face_view.create_display(
                        face_img, None, 0, self.eyes.blinkCounter, False,
                        None, None, phone_detected, None
                    )
                else:
                    # Full-frame view (no stacking)
                    imgStack = img

            # Flush to DB every interval
            if self.stats_db and self.stats.ready(now):
                doc = self.stats.flush(now)
                self.stats_db.insert(doc)

            cv2.imshow(self.window_title, imgStack)
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
    parser.add_argument("--face-only", action="store_true", default=False, help="Show face-only cropped view (default: False)")
    parser.add_argument("--full-view", action="store_true", help="Force full frame view (overrides --face-only)")
    parser.add_argument("--enable-emotion", action="store_true", default=True, help="Enable emotion detection (default: True)")
    parser.add_argument("--no-emotion", action="store_true", help="Disable emotion detection")
    args = parser.parse_args()

    if args.list_cams:
        cams = camera_utils.list_cameras() if hasattr(camera_utils, "list_cameras") else _fallback_list_cameras()
        for idx, w, h in cams:
            print(f"Camera {idx}: {w}x{h}")
        raise SystemExit(0)

    cam_index = args.cam if args.cam is not None else int(os.getenv("CAM_INDEX", "1"))

    if args.url:
        print(f"Using network stream: {args.url}")
    else:
        print(f"Using camera index: {cam_index}")
    
    # Determine view mode (default: full frame)
    face_only = args.face_only and not args.full_view
    
    # Determine emotion detection
    enable_emotion = not args.no_emotion  # Emotion is enabled by default unless --no-emotion is specified
    
    print(f"View mode: {'Face-only' if face_only else 'Full frame'}")
    print(f"Emotion detection: {'Enabled' if enable_emotion else 'Disabled'}")

    app = DriverMonitor(
        cam_index=cam_index,
        yolo_model_path=args.yolo,
        warning_sound_path=args.sound,
        url=args.url,
        stats_interval=args.stats_interval,
        mongo_uri=args.mongo_uri,
        mongo_db=args.mongo_db,
        mongo_coll=args.mongo_coll,
        face_only_view=face_only,
        enable_emotion=enable_emotion,
    )
    app.run()

    # Example:
# .venv/bin/python -m Monitor.main --list-cams
# .venv/bin/python -m Monitor.main --cam 0
# .venv/bin/python -m Monitor.main --url http://172.20.10.3:8080/video
# python3 -m Monitor_emotionAdded.main --cam 1