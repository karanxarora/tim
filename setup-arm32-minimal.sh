#!/bin/bash
# EdgeVLM Minimal Setup for ARM32 (armv7l)
# Only installs essential packages that work on ARM32

set -e

echo "=========================================="
echo "EdgeVLM Minimal ARM32 Setup"
echo "=========================================="

# Detect system architecture
ARCH=$(uname -m)
echo "Detected architecture: $ARCH"

if [[ "$ARCH" != "armv7l" ]]; then
    echo "Warning: This script is optimized for ARM32 (armv7l)"
    echo "Current architecture: $ARCH"
fi

# Update system packages
echo "Updating system packages..."
sudo apt update
sudo apt upgrade -y

# Install minimal system dependencies
echo "Installing minimal system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    cmake \
    pkg-config \
    libjpeg-dev \
    libtiff5-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    wget \
    curl \
    git \
    htop \
    vim

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install only essential packages that work on ARM32
echo "Installing essential ARM32-compatible packages..."

# Core packages (these have ARM32 wheels)
pip install \
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

# Install OpenCV (has ARM32 wheels)
echo "Installing OpenCV..."
pip install opencv-python

# Install llama-cpp-python with ARM32 optimizations
echo "Installing llama-cpp-python for ARM32..."
CMAKE_ARGS="-DLLAMA_NATIVE=ON -DLLAMA_NEON=ON" pip install llama-cpp-python

# Create necessary directories
echo "Creating project directories..."
mkdir -p models
mkdir -p logs
mkdir -p benchmarks
mkdir -p uploads
mkdir -p cache
mkdir -p examples

# Set up system optimizations for Raspberry Pi
echo "Setting up system optimizations..."

# Increase GPU memory split (for better performance)
echo "Setting GPU memory split..."
sudo raspi-config nonint do_memory_split 128

# Enable camera interface
echo "Enabling camera interface..."
sudo raspi-config nonint do_camera 0

# Set up swap file for better memory management
echo "Setting up swap file..."
sudo dphys-swapfile swapoff
sudo sed -i 's/CONF_SWAPSIZE=100/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Optimize system for EdgeVLM
echo "Optimizing system settings..."
echo "vm.swappiness=10" | sudo tee -a /etc/sysctl.conf
echo "vm.vfs_cache_pressure=50" | sudo tee -a /etc/sysctl.conf

# Create systemd service for EdgeVLM
echo "Creating systemd service..."
sudo tee /etc/systemd/system/edgevlm.service > /dev/null <<EOF
[Unit]
Description=EdgeVLM Vision-Language Model Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
ExecStart=$(pwd)/venv/bin/python main.py --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable the service
sudo systemctl daemon-reload
sudo systemctl enable edgevlm.service

echo "=========================================="
echo "EdgeVLM Minimal ARM32 Setup Complete!"
echo "=========================================="
echo ""
echo "✅ Essential packages installed"
echo "❌ PyTorch skipped (not available for ARM32)"
echo "❌ SciPy skipped (build issues on ARM32)"
echo "❌ Matplotlib skipped (build issues on ARM32)"
echo ""
echo "The system will use llama-cpp-python for inference"
echo ""
echo "Next steps:"
echo "1. Test the system: python test_arm32_setup.py"
echo "2. Download models: python download_models_arm32.py"
echo "3. Test the system: python main.py --log-level DEBUG"
echo "4. Start service: sudo systemctl start edgevlm"
echo ""
echo "For better performance, consider using 64-bit Raspberry Pi OS"
echo "System optimized for ARM32 (armv7l) architecture!"
