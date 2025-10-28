# MonitorTest2 - Enhanced Driver Monitoring with Emotion Detection

## New Features

### 1. Emotion Detection
- Integrated CNN-based emotion detection from emotionCNN.py
- Detects: Angry, Happy, Neutral emotions
- Shows confidence percentage for each prediction
- Uses pre-trained model from HuggingFace

### 2. Face-Only View (Default)
- Shows only the driver's face in a large, clear window
- Includes info panel with real-time statistics:
  - Current emotion and confidence
  - Eye status (open/closed)
  - Blink count
  - Head pose (yaw/pitch angles)
  - Mouth status (yawning detection)
  - Phone detection warnings

## Usage

### Basic Usage (Face-Only View with Emotion Detection)
```bash
# Activate the environment
source face_detection_env/bin/activate

# Run with default settings (face-only view, emotion enabled)
python -m MonitorTest2.main --cam 0
```

### List Available Cameras
```bash
python -m MonitorTest2.main --list-cams
```

### Run with Full Frame View
```bash
# Disable face-only view to see traditional stacked view
python -m MonitorTest2.main --cam 0 --full-view
```

### Disable Emotion Detection
```bash
# Run without emotion detection for better performance
python -m MonitorTest2.main --cam 0 --no-emotion
```

### Use Network Stream
```bash
# Connect to phone camera or IP camera
python -m MonitorTest2.main --url http://192.168.1.100:8080/video
```

### Complete Example with All Options
```bash
python -m MonitorTest2.main \
    --cam 0 \
    --yolo yolov8n.pt \
    --sound warning.mp3 \
    --stats-interval 5 \
    --mongo-uri "mongodb+srv://..." \
    --mongo-db driver_monitor \
    --mongo-coll stats
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--cam` | Camera index (0, 1, 2...) | 1 |
| `--url` | Network stream URL | None |
| `--list-cams` | List available cameras and exit | - |
| `--yolo` | Path to YOLO model for phone detection | yolov8n.pt |
| `--sound` | Path to warning sound file | warning.mp3 |
| `--face-only` | Enable face-only view (default) | True |
| `--full-view` | Show full frame instead of face-only | False |
| `--enable-emotion` | Enable emotion detection (default) | True |
| `--no-emotion` | Disable emotion detection | False |
| `--stats-interval` | Statistics aggregation interval (seconds) | 5 |
| `--mongo-uri` | MongoDB connection string | None |
| `--mongo-db` | MongoDB database name | driver_monitor |
| `--mongo-coll` | MongoDB collection name | stats |

## Display Information

### Face-Only View Shows:
- **Large face image** (640x480 pixels)
- **Emotion**: Current detected emotion with color coding
  - 🟢 Green = Happy
  - 🟡 Yellow = Neutral
  - 🔴 Red = Angry
- **Confidence**: Emotion prediction confidence (0-100%)
- **Eyes**: Status (OPEN/CLOSED) with color indication
- **Blinks**: Total blink count
- **Head Pose**: Yaw and pitch angles in degrees
- **Mouth**: Yawning detection status
- **Phone**: Warning if phone is detected

### Full View Shows:
- Original frame with overlays
- Eye ratio plot
- Traditional monitoring interface

## Controls

- **Press 'q'** to quit the application

## Performance Notes

### With Emotion Detection
- First run downloads model from HuggingFace (~50MB)
- TensorFlow model loading takes ~5-10 seconds on first import
- Real-time emotion detection runs at ~20-30 FPS on modern hardware

### Without Emotion Detection
- Faster startup time
- Higher FPS (~30-60 FPS)
- Lower CPU/GPU usage

## Requirements

All requirements are in `requirements-updated.txt`:
- tensorflow==2.18.0
- mediapipe==0.10.14
- opencv-python==4.10.0.84
- cvzone==1.6.1
- ultralytics==8.3.0
- huggingface-hub==0.26.2
- And more...

## Troubleshooting

### Emotion model fails to load
```
[warn] Failed to load emotion model: ...
```
**Solution**: Check internet connection. Model downloads from HuggingFace on first run.

### Camera not found
```
[error] No cameras detected
```
**Solution**: Run `--list-cams` to see available cameras, then use the correct index.

### Slow performance
**Solution**: Disable emotion detection with `--no-emotion` flag.

### Import errors
**Solution**: Make sure you activated the virtual environment:
```bash
source face_detection_env/bin/activate
```

## Architecture

### New Modules
1. **emotion_detection.py**: EmotionDetector class
   - Loads CNN model from HuggingFace
   - Detects emotions from face images
   - Returns emotion label, confidence, and bounding box

2. **face_view.py**: FaceOnlyView class
   - Extracts and crops face region
   - Creates info panel with statistics
   - Combines face and info for display

### Integration Flow
```
Camera → Face Detection → Emotion Detection → Face-Only View
                       ↓
              Eye/Head/Mouth Analysis
                       ↓
              Stats Aggregation → MongoDB
```

## Examples

### Quick Test
```bash
# Activate environment
source face_detection_env/bin/activate

# Quick test with default camera
python -m MonitorTest2.main --cam 0
```

### Production Mode
```bash
# With MongoDB logging
python -m MonitorTest2.main \
    --cam 0 \
    --mongo-uri "$MONGODB_URI" \
    --stats-interval 3
```

### Development Mode
```bash
# Without emotion detection for faster iteration
python -m MonitorTest2.main --cam 0 --no-emotion --full-view
```
