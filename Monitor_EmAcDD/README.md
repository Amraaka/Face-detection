# Driver Monitor with Emotion & Activity Detection

Enhanced driver monitoring system with emotion detection and dangerous activity recognition.

## Features

### Core Detection Layers
1. **Eye Analysis**: Blink detection and drowsiness monitoring
2. **Head Pose Estimation**: Looking left/right/up/down tracking
3. **Phone Detection**: Mobile phone usage detection
4. **Mouth Analysis**: Yawn detection
5. **Emotion Detection**: Real-time emotion recognition (Angry, Happy, Neutral)
6. **Activity Detection**: Dangerous driver actions (NEW!)
   - Drinking
   - Talking on phone
   - Yawning
   - Other activities

## Activity Detection

The activity detection layer uses a Vision Transformer (ViT) model to identify dangerous driver behaviors:

### Detected Activities
- **drinking**: Driver consuming beverages while driving
- **talking_phone**: Driver using phone (talking)
- **yawning**: Driver yawning (fatigue indicator)
- **other_activities**: General distracting activities

### Model
- Model: `Ganaa614/vit-tiny-patch16-224activity_recognition_4feats`
- Input: Full frame RGB image
- Output: Activity label + confidence score

## Database Storage

All detected activities are stored in MongoDB with the following structure:

```javascript
{
  "session_id": "uuid",
  "window_start_dt": "2025-10-28T12:00:00Z",
  "window_end_dt": "2025-10-28T12:00:05Z",
  "duration_sec": 5,
  
  // Activity data
  "activity_counts": {
    "drinking": 2,
    "other_activities": 0,
    "talking_phone": 1,
    "yawning": 3
  },
  "activity_top_label": "yawning",        // Most frequent activity
  "activity_top_conf_avg": 0.8543,        // Average confidence
  "activity_last": "drinking",             // Last detected activity
  
  // Other detection data...
  "emotion_counts": { ... },
  "blinks_count": 10,
  // ...
}
```

## Usage

### Basic Usage
```bash
# Enable all features (default)
python3 -m Monitor_EmAc.main --cam 1

# Disable activity detection
python3 -m Monitor_EmAc.main --cam 1 --no-activity

# Disable emotion detection
python3 -m Monitor_EmAc.main --cam 1 --no-emotion

# Enable both with MongoDB storage
python3 -m Monitor_EmAc.main --cam 1 \
  --mongo-uri "mongodb+srv://..." \
  --mongo-db "driver_monitor" \
  --mongo-coll "stats"
```

### Command Line Arguments
```
--cam INDEX               Camera index (default: 1)
--url URL                 HTTP MJPEG or RTSP URL
--enable-emotion          Enable emotion detection (default: True)
--no-emotion             Disable emotion detection
--enable-activity        Enable activity detection (default: True)
--no-activity            Disable activity detection
--face-only              Show face-only cropped view
--full-view              Force full frame view
--stats-interval SECS    Stats aggregation interval (default: 5.0)
--mongo-uri URI          MongoDB connection string
--mongo-db DB            MongoDB database name
--mongo-coll COLL        MongoDB collection name
```

## API Endpoints

### Get Activity Data
```http
GET /api/stats/activities?lastHours=24&interval=1m
```

Response:
```json
{
  "range": {
    "from": null,
    "to": "2025-10-28T12:00:00Z",
    "interval": { "unit": "sec", "binSize": 60 }
  },
  "timeseries": [
    {
      "ts": "2025-10-28T11:00:00Z",
      "activities": {
        "drinking": 5,
        "other_activities": 2,
        "talking_phone": 3,
        "yawning": 8
      },
      "top_label": "yawning",
      "avg_confidence": 0.8234
    }
  ]
}
```

### Get Emotion Data
```http
GET /api/stats/emotions?lastHours=24&interval=1m
```

### Get All Stats
```http
GET /api/stats?lastHours=24&interval=5m
```

## Architecture

### Detection Pipeline
```
Frame Input
    ↓
Phone Detection (YOLOv8) ─→ Stats
    ↓
Face Mesh Detection
    ↓
├─→ Eye Analysis ─→ Stats
├─→ Head Pose ─→ Stats
├─→ Mouth Analysis ─→ Stats
├─→ Emotion Detection (CNN) ─→ Stats
└─→ Activity Detection (ViT) ─→ Stats
    ↓
Stats Aggregation (every 5s)
    ↓
MongoDB Storage
```

### Files
- `main.py`: Main application loop
- `activity_detection.py`: Activity detection module (NEW)
- `emotion_detection.py`: Emotion detection module
- `eye_analysis.py`: Blink and drowsiness detection
- `head_pose.py`: Head pose estimation
- `mouth_analysis.py`: Yawn detection
- `phone_detection.py`: Phone usage detection
- `stats.py`: Statistics aggregation
- `stats_store.py`: MongoDB storage

## Dependencies

Required packages:
- opencv-python
- torch
- transformers
- pillow
- mediapipe
- ultralytics
- pymongo

Install via:
```bash
pip install opencv-python torch transformers pillow mediapipe ultralytics pymongo
```

## Performance Notes

- Activity detection runs on full frames at ~5-10 FPS on CPU
- Emotion detection runs on face crops at ~15-20 FPS on CPU
- For better performance, use GPU: set `device="cuda"` in constructors
- Activity model is lightweight (ViT-tiny) optimized for edge devices

## Database Schema

See `backend/src/models/DriverStat.js` for complete schema including:
- `activity_counts`: Map of activity labels to counts
- `activity_top_label`: Most frequent activity in window
- `activity_top_conf_avg`: Average confidence score
- `activity_last`: Last detected activity label
