import os
import time
from datetime import datetime

import cv2
import numpy as np
import pygame
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
from ultralytics import YOLO

# --- Optional .env support for Atlas ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- Config constants ---
EAR_THRESHOLD = 31                 # eye aspect ratio threshold (smaller => closed)
YAW_THRESHOLD = 20                 # degrees
PITCH_THRESHOLD = 20               # degrees
DROWSINESS_TIME_THRESHOLD = 2.0    # seconds eyes closed before warning
DISTRACTION_TIME_THRESHOLD = 3.0   # seconds head away before warning
WARNING_COOLDOWN = 3.0             # seconds between audio warnings
PHONE_CONFIDENCE_THRESHOLD = 0.5   # YOLO conf threshold
DB_SAVE_INTERVAL = 5.0             # seconds between periodic saves

# --- MongoDB Atlas setup ---
from pymongo import MongoClient

MONGODB_URI = os.getenv("MONGODB_URI")  # required for Atlas
MONGODB_DB = os.getenv("MONGODB_DB", "driver_monitoring")
MONGO_COLL_MONITOR = os.getenv("MONGODB_MONITORING_COLLECTION", "monitoring_data")
MONGO_COLL_SUMMARY = os.getenv("MONGODB_SUMMARY_COLLECTION", "session_summaries")

mongo_client = None
monitor_coll = None
summary_coll = None

def init_mongo():
    global mongo_client, monitor_coll, summary_coll
    if not MONGODB_URI:
        print("MongoDB: MONGODB_URI is not set. Skipping DB logging.")
        return
    try:
        mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        # Validate connection
        mongo_client.admin.command("ping")
        db = mongo_client[MONGODB_DB]
        monitor_coll = db[MONGO_COLL_MONITOR]
        summary_coll = db[MONGO_COLL_SUMMARY]
        print("MongoDB: Connected.")
    except Exception as e:
        print(f"MongoDB: Connection failed: {e}")
        mongo_client = None
        monitor_coll = None
        summary_coll = None

def save_monitoring_record(data: dict):
    if monitor_coll is None:
        return
    try:
        monitor_coll.insert_one(data)
    except Exception as e:
        print(f"MongoDB: Insert monitoring_data failed: {e}")

def save_session_summary(data: dict):
    if summary_coll is None:
        return
    try:
        summary_coll.insert_one(data)
    except Exception as e:
        print(f"MongoDB: Insert session_summaries failed: {e}")

def create_monitoring_record(session_id, ts, phone_detected, blink_count, yaw, pitch, ear, warning_type):
    return {
        "session_id": session_id,
        "timestamp": datetime.fromtimestamp(ts),
        "phone_detected": bool(phone_detected),
        "blink_count": int(blink_count),
        "head_yaw": float(yaw),
        "head_pitch": float(pitch),
        "eye_aspect_ratio": float(ear),
        "warning_type": warning_type,   # "phone" | "drowsiness" | "distraction" | None
        "created_at": datetime.utcnow(),
    }

# --- Audio (safe init) ---
warning_sound = None
try:
    pygame.mixer.init()
    try:
        warning_sound = pygame.mixer.Sound("warning.mp3")
    except Exception:
        print("Audio: warning.mp3 not found or could not be loaded. Sounds disabled.")
        warning_sound = None
except Exception as e:
    print(f"Audio: Mixer init failed ({e}). Sounds disabled.")
    warning_sound = None

def play_warning_if_ready(now, last_time):
    if warning_sound and (now - last_time) > WARNING_COOLDOWN:
        try:
            warning_sound.play()
        except Exception:
            pass
        return now
    return last_time

# --- Computer vision setup ---
# Camera: try index 1, fallback to 0
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(400, 600, [25, 40])  # For EAR visualization
model = YOLO("yolov8n.pt")            # Will auto-download if needed

# --- Runtime state ---
ratioList = []
blinkCounter = 0
counter = 0
color = (255, 0, 255)

head_turned = False
turned_start_time = 0.0
last_warning_time = 0.0

leftEyeIdList = [22, 23, 24, 25, 26, 110, 157, 158, 159, 160, 161, 130, 243]
rightEyeIdList = [252, 253, 254, 255, 256, 339, 384, 385, 386, 387, 388, 446, 463]

eye_closed_start_time = None
eye_closed_warning_given = False

# Session/DB
session_start_time = time.time()
session_id = int(session_start_time)
total_phone_detections = 0
total_drowsiness_warnings = 0
total_distraction_warnings = 0
last_db_save = time.time()

init_mongo()

try:
    while True:
        ok, img = cap.read()
        if not ok:
            print("Camera frame not available. Exiting.")
            break

        current_time = time.time()
        warning_type = None

        # --- YOLO phone detection ---
        phone_detected = False
        results = model(img, stream=True, verbose=False)
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = str(model.names[cls])
                phone_labels = ["cell phone", "phone", "mobile phone", "smartphone"]
                if any(p in label.lower() for p in phone_labels) and conf >= PHONE_CONFIDENCE_THRESHOLD:
                    phone_detected = True
                    total_phone_detections += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if phone_detected:
            warning_type = "phone"
            cv2.putText(img, "WARNING: PHONE DETECTED!", (50, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            last_warning_time = play_warning_if_ready(current_time, last_warning_time)

        # --- Face mesh / EAR ---
        img, faces = detector.findFaceMesh(img, draw=False)

        # Defaults when no face
        yaw, pitch, ear_avg = 0.0, 0.0, 0.0

        if faces:
            face = faces[0]

            leftUp = face[159]
            leftDown = face[23]
            leftLeft = face[130]
            leftRight = face[243]
            lengthVer, _ = detector.findDistance(leftUp, leftDown)
            lengthHor, _ = detector.findDistance(leftLeft, leftRight)
            cv2.line(img, leftUp, leftDown, (0, 200, 0), 3)
            cv2.line(img, leftLeft, leftRight, (0, 200, 0), 3)

            ratio = (lengthVer / max(1e-6, lengthHor)) * 100.0
            ratioList.append(ratio)
            if len(ratioList) > 3:
                ratioList.pop(0)
            ear_avg = sum(ratioList) / len(ratioList)

            # Blink/drowsiness
            if ear_avg < EAR_THRESHOLD:
                if counter == 0:
                    blinkCounter += 1
                    color = (0, 200, 0)
                    counter = 1
                if eye_closed_start_time is None:
                    eye_closed_start_time = current_time
                    eye_closed_warning_given = False
                elif not eye_closed_warning_given and (current_time - eye_closed_start_time) > DROWSINESS_TIME_THRESHOLD:
                    warning_type = "drowsiness"
                    total_drowsiness_warnings += 1
                    cv2.putText(img, "WARNING: EYES CLOSED!", (50, 350),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    last_warning_time = play_warning_if_ready(current_time, last_warning_time)
                    eye_closed_warning_given = True
            else:
                eye_closed_start_time = None
                eye_closed_warning_given = False

            if counter != 0:
                counter += 1
                if counter > 10:
                    counter = 0
                    color = (255, 0, 255)

            cvzone.putTextRect(img, f"Blink Count: {blinkCounter}", (50, 100), scale=2, thickness=2, colorR=color)

            # --- Head pose (PnP) ---
            image_points = np.array([
                face[1],    # Nose tip
                face[152],  # Chin
                face[263],  # Left eye left corner
                face[33],   # Right eye right corner
                face[287],  # Left mouth corner
                face[57]    # Right mouth corner
            ], dtype="double")

            model_points = np.array([
                (0.0, 0.0, 0.0),
                (0.0, -63.6, -12.5),
                (-43.3, 32.7, -26.0),
                (43.3, 32.7, -26.0),
                (-28.9, -28.9, -24.1),
                (28.9, -28.9, -24.1)
            ])

            size = img.shape
            focal_length = size[1]
            center = (size[1] / 2, size[0] / 2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype="double"
            )
            dist_coeffs = np.zeros((4, 1))

            ok_solve, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )

            if ok_solve:
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                proj_matrix = np.hstack((rotation_matrix, translation_vector))
                _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)
                pitch, yaw, roll = [float(angle[0]) for angle in eulerAngles]

                cv2.putText(img, f"Yaw: {int(yaw)}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(img, f"Pitch: {int(pitch)}", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                # Distraction detection
                if abs(yaw) > YAW_THRESHOLD or abs(pitch) > PITCH_THRESHOLD:
                    if not head_turned:
                        turned_start_time = current_time
                        head_turned = True
                    if (current_time - turned_start_time) > DISTRACTION_TIME_THRESHOLD:
                        warning_type = "distraction"
                        total_distraction_warnings += 1
                        cv2.putText(img, "WARNING: LOOKING AWAY!", (50, 250),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                        last_warning_time = play_warning_if_ready(current_time, last_warning_time)
                else:
                    head_turned = False

            imgPlot = plotY.update(ear_avg, color)
            imgStack = cvzone.stackImages([img, imgPlot], 2, 1)
        else:
            imgStack = cvzone.stackImages([img, img], 2, 1)

        # HUD stats
        cv2.putText(imgStack, f"Session: {session_id}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(imgStack, f"Phone: {total_phone_detections}", (240, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(imgStack, f"Drowsy: {total_drowsiness_warnings}", (410, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(imgStack, f"Distract: {total_distraction_warnings}", (610, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- Periodic DB save ---
        if (current_time - last_db_save) >= DB_SAVE_INTERVAL:
            rec = create_monitoring_record(
                session_id=session_id,
                ts=current_time,
                phone_detected=phone_detected,
                blink_count=blinkCounter,
                yaw=yaw,
                pitch=pitch,
                ear=ear_avg,
                warning_type=warning_type
            )
            save_monitoring_record(rec)
            last_db_save = current_time

        # Display
        cv2.imshow("Driver Monitoring", imgStack)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # Session summary
    session_summary = {
        "session_id": session_id,
        "session_start": datetime.fromtimestamp(session_start_time),
        "session_end": datetime.utcnow(),
        "duration_seconds": float(time.time() - session_start_time),
        "total_blinks": int(blinkCounter),
        "total_phone_detections": int(total_phone_detections),
        "total_drowsiness_warnings": int(total_drowsiness_warnings),
        "total_distraction_warnings": int(total_distraction_warnings),
        "record_type": "session_summary",
        "created_at": datetime.utcnow(),
    }
    save_session_summary(session_summary)

    # Cleanup
    try:
        cap.release()
    except Exception:
        pass
    cv2.destroyAllWindows()
    if mongo_client:
        mongo_client.close()
    print("Exited.")