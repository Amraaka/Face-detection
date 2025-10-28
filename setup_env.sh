#!/bin/bash
# Script to recreate Python environment for Face-detection project
# Resolves version conflicts between jax, mediapipe, and tensorflow

set -e  # Exit on error

echo "🔧 Setting up Python environment for Face-detection project..."

# Configuration
ENV_NAME="face_detection_env"
PYTHON_VERSION="python3.10"

# Check if Python 3.10 is available
if ! command -v $PYTHON_VERSION &> /dev/null; then
    echo "❌ Python 3.10 not found. Please install Python 3.10 first."
    echo "   You can install it via Homebrew: brew install python@3.10"
    exit 1
fi

# Remove old environment if it exists
if [ -d "$ENV_NAME" ]; then
    echo "🗑️  Removing old environment: $ENV_NAME"
    rm -rf "$ENV_NAME"
fi

# Create new virtual environment
echo "📦 Creating new virtual environment: $ENV_NAME"
$PYTHON_VERSION -m venv "$ENV_NAME"

# Activate the environment
echo "✅ Activating environment..."
source "$ENV_NAME/bin/activate"

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install requirements
echo "📥 Installing packages from requirements-updated.txt..."
pip install -r requirements-updated.txt

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "   source $ENV_NAME/bin/activate"
echo ""
echo "To test emotionCNN.py:"
echo "   python emotionCNN.py"
echo ""
echo "To test MonitorTest1:"
echo "   python -m MonitorTest1.main --cam 1"
echo ""
echo "To deactivate:"
echo "   deactivate"
