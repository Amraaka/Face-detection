#!/usr/bin/env python3
"""
Verify that all required packages are installed and compatible
for emotionCNN.py and MonitorTest1
"""

import sys

def check_import(module_name, import_statement=None):
    """Try importing a module and report status"""
    try:
        if import_statement:
            exec(import_statement)
        else:
            __import__(module_name)
        print(f"✅ {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {module_name}: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {module_name}: {e}")
        return False

def main():
    print("🔍 Verifying Python environment for Face-detection project\n")
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version < (3, 10):
        print("⚠️  Warning: Python 3.10+ is recommended")
    print()
    
    # Core packages
    print("Core ML Frameworks:")
    check_import("tensorflow")
    check_import("mediapipe")
    check_import("jax")
    check_import("jaxlib")
    print()
    
    # Computer vision
    print("Computer Vision:")
    check_import("cv2", "import cv2")
    check_import("cvzone")
    check_import("numpy")
    print()
    
    # Object detection
    print("Object Detection:")
    check_import("ultralytics", "from ultralytics import YOLO")
    print()
    
    # Database and utilities
    print("Database & Utilities:")
    check_import("pymongo")
    check_import("dotenv", "from dotenv import load_dotenv")
    check_import("requests")
    check_import("pygame")
    print()
    
    # HuggingFace
    print("HuggingFace:")
    check_import("huggingface_hub", "from huggingface_hub import snapshot_download")
    print()
    
    # Check version compatibility
    print("Package Versions:")
    try:
        import tensorflow as tf
        print(f"  tensorflow: {tf.__version__}")
    except:
        pass
    
    try:
        import mediapipe as mp
        print(f"  mediapipe: {mp.__version__}")
    except:
        pass
    
    try:
        import cv2
        print(f"  opencv-python: {cv2.__version__}")
    except:
        pass
    
    try:
        import jax
        print(f"  jax: {jax.__version__}")
    except:
        pass
    
    try:
        import numpy as np
        print(f"  numpy: {np.__version__}")
    except:
        pass
    
    print("\n✅ Verification complete!")

if __name__ == "__main__":
    main()
