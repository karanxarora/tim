#!/bin/bash
# Quick fix for ARM32 setup issues
# Run this if you're having package installation problems

echo "=========================================="
echo "ARM32 Setup Quick Fix"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Please run this script from the EdgeVLM directory"
    exit 1
fi

# Activate virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "❌ Virtual environment not found. Run setup script first."
    exit 1
fi

# Install only essential packages that work on ARM32
echo "Installing essential packages..."

# Core packages (these have ARM32 wheels)
pip install --upgrade \
    numpy \
    pillow \
    requests \
    pyyaml \
    fastapi \
    uvicorn \
    pydantic \
    python-multipart \
    aiofiles \
    psutil \
    py-cpuinfo \
    python-json-logger \
    tqdm \
    huggingface-hub \
    sentencepiece \
    protobuf

# Install OpenCV
echo "Installing OpenCV..."
pip install opencv-python

# Install llama-cpp-python
echo "Installing llama-cpp-python..."
CMAKE_ARGS="-DLLAMA_NATIVE=ON -DLLAMA_NEON=ON" pip install llama-cpp-python

# Test the installation
echo "Testing installation..."
python -c "
import sys
print('Python version:', sys.version)

try:
    import numpy
    print('✅ numpy')
except ImportError:
    print('❌ numpy')

try:
    import cv2
    print('✅ opencv-python')
except ImportError:
    print('❌ opencv-python')

try:
    import llama_cpp
    print('✅ llama-cpp-python')
except ImportError:
    print('❌ llama-cpp-python')

try:
    import fastapi
    print('✅ fastapi')
except ImportError:
    print('❌ fastapi')

try:
    from api import app
    print('✅ EdgeVLM API')
except ImportError as e:
    print('❌ EdgeVLM API:', e)
"

echo "=========================================="
echo "Quick fix complete!"
echo "=========================================="
echo ""
echo "If all packages show ✅, you can proceed with:"
echo "1. python download_models_arm32.py"
echo "2. python main.py --log-level DEBUG"
echo ""
echo "If some packages show ❌, try:"
echo "1. pip install --upgrade pip"
echo "2. pip install --no-cache-dir <package-name>"
echo "3. Or use the minimal setup script: ./setup-arm32-minimal.sh"
