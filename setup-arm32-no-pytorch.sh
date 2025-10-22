#!/bin/bash
# EdgeVLM Setup Script for Raspberry Pi ARM32 (armv7l) - PyTorch Alternative
# Optimized for Raspberry Pi 4 with 4GB+ RAM

set -e

echo "=========================================="
echo "EdgeVLM ARM32 Setup for Raspberry Pi"
echo "PyTorch Alternative Version"
echo "=========================================="

# Detect system architecture
ARCH=$(uname -m)
echo "Detected architecture: $ARCH"

if [[ "$ARCH" != "armv7l" ]]; then
    echo "Warning: This script is optimized for ARM32 (armv7l)"
    echo "Current architecture: $ARCH"
    echo "Continue anyway? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Setup cancelled"
        exit 1
    fi
fi

# Update system packages
echo "Updating system packages..."
sudo apt update
sudo apt upgrade -y

# Install system dependencies
echo "Installing system dependencies..."
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

# Install ARM32 optimized packages
echo "Installing ARM32 optimized packages..."

# Install numpy first (ARM32 compatible)
pip install numpy==1.24.3

# Install PyTorch alternative for ARM32
echo "Installing PyTorch alternative for ARM32..."
echo "PyTorch wheels are not available for ARM32 from PyPI"
echo "Installing alternative packages..."

# Install alternative packages that work on ARM32
pip install \
    scipy \
    scikit-learn \
    scikit-image \
    matplotlib

# Install OpenCV for ARM32
echo "Installing OpenCV for ARM32..."
pip install opencv-python==4.8.1.78

# Install other core packages
echo "Installing core packages..."
pip install \
    pillow==10.1.0 \
    requests==2.31.0 \
    pyyaml==6.0.1 \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pydantic==2.5.0 \
    python-multipart==0.0.6 \
    aiofiles==23.2.1 \
    psutil==5.9.6 \
    py-cpuinfo==9.0.0 \
    python-json-logger==2.0.7 \
    tqdm==4.66.1 \
    huggingface-hub==0.19.4

# Install llama-cpp-python with ARM32 optimizations
echo "Installing llama-cpp-python for ARM32..."
CMAKE_ARGS="-DLLAMA_NATIVE=ON -DLLAMA_NEON=ON" pip install llama-cpp-python==0.2.90

# Install transformers and other ML packages (without PyTorch dependency)
echo "Installing ML packages..."
pip install \
    transformers==4.36.0 \
    accelerate==0.25.0 \
    sentencepiece==0.1.99 \
    protobuf==4.25.1

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
echo "EdgeVLM ARM32 Setup Complete!"
echo "=========================================="
echo ""
echo "Note: PyTorch was not installed due to ARM32 compatibility issues"
echo "The system will use llama-cpp-python for inference instead"
echo ""
echo "Next steps:"
echo "1. Download models: python download_models_arm32.py"
echo "2. Test the system: python main.py --log-level DEBUG"
echo "3. Start service: sudo systemctl start edgevlm"
echo "4. Check status: sudo systemctl status edgevlm"
echo ""
echo "For ngrok integration:"
echo "1. Install ngrok: curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && echo 'deb https://ngrok-agent.s3.amazonaws.com buster main' | sudo tee /etc/apt/sources.list.d/ngrok.list && sudo apt update && sudo apt install ngrok"
echo "2. Authenticate: ngrok config add-authtoken <your-token>"
echo "3. Run: python ngrok_integration.py"
echo ""
echo "System optimized for ARM32 (armv7l) architecture!"
