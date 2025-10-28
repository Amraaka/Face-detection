# Environment Setup Guide

## Problem
Version conflicts between `jax`, `mediapipe`, and `tensorflow` when running:
- `emotionCNN.py` (requires TensorFlow)
- `MonitorTest1` modules (require mediapipe via cvzone)

## Solution

### Automatic Setup (Recommended)

Run the setup script:
```bash
cd /Users/amara/SideProjects/Research/Face-detection
./setup_env.sh
```

This will:
1. Remove any old environment
2. Create a new `face_detection_env` virtual environment with Python 3.10
3. Install all compatible packages from `requirements-updated.txt`

### Manual Setup

If you prefer manual setup:

```bash
# Create virtual environment
python3.10 -m venv face_detection_env

# Activate it
source face_detection_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements-updated.txt
```

## Usage

### Activate the environment
```bash
source face_detection_env/bin/activate
```

### Run emotionCNN.py
```bash
python emotionCNN.py
```

### Run MonitorTest1
```bash
python -m MonitorTest1.main --cam 1
```

Or list cameras first:
```bash
python -m MonitorTest1.main --list-cams
```

### Deactivate
```bash
deactivate
```

## Package Versions (Resolved)

- **tensorflow**: 2.18.0 (latest stable, compatible with macOS)
- **mediapipe**: 0.10.14 (has its own jax dependencies)
- **opencv-python**: 4.10.0.84
- **cvzone**: 1.6.1
- **ultralytics**: 8.3.0 (YOLOv8)
- **numpy**: <2.0.0 (for compatibility)

## Notes

- Uses Python 3.10 for best compatibility
- Avoids explicit jax installation to prevent conflicts
- TensorFlow 2.18.0 and mediapipe 0.10.14 have compatible dependency trees
- If you still encounter issues, you may need to run them in separate environments

## Troubleshooting

### If Python 3.10 is not installed:
```bash
brew install python@3.10
```

### If you get permission errors:
```bash
chmod +x setup_env.sh
```

### Camera not found errors:
Try different camera indices:
```bash
python -m MonitorTest1.main --list-cams
python -m MonitorTest1.main --cam 0  # or 1, 2, etc.
```
