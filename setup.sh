#!/bin/bash
# EdgeVLM Setup Script for Raspberry Pi / ARM64
# Tested on Ubuntu 22.04 LTS and Raspberry Pi OS 64-bit

set -e

echo "======================================"
echo "EdgeVLM Setup for Raspberry Pi"
echo "======================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on ARM64
ARCH=$(uname -m)
if [[ "$ARCH" != "aarch64" && "$ARCH" != "arm64" ]]; then
    echo -e "${YELLOW}Warning: Not running on ARM64 architecture. System detected: $ARCH${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check available RAM
TOTAL_RAM=$(free -m | awk '/^Mem:/{print $2}')
if [ "$TOTAL_RAM" -lt 6144 ]; then
    echo -e "${YELLOW}Warning: Less than 6GB RAM available ($TOTAL_RAM MB). Performance may be degraded.${NC}"
fi

# Create directory structure
echo -e "${GREEN}Creating directory structure...${NC}"
mkdir -p models
mkdir -p logs
mkdir -p benchmarks
mkdir -p cache
mkdir -p uploads

# Update system packages
echo -e "${GREEN}Updating system packages...${NC}"
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv python3-dev
sudo apt-get install -y cmake build-essential pkg-config
sudo apt-get install -y libopenblas-dev liblapack-dev
sudo apt-get install -y libopencv-dev
sudo apt-get install -y wget curl git

# Create virtual environment
echo -e "${GREEN}Creating Python virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch CPU-only for ARM
echo -e "${GREEN}Installing PyTorch (CPU-only for ARM)...${NC}"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install llama-cpp-python with ARM optimizations
echo -e "${GREEN}Installing llama-cpp-python with ARM NEON optimizations...${NC}"
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python --no-cache-dir

# Install remaining requirements
echo -e "${GREEN}Installing Python dependencies...${NC}"
pip install -r requirements.txt

# Download quantized models
echo -e "${GREEN}Downloading quantized models...${NC}"

# Function to download with progress
download_model() {
    local url=$1
    local output=$2
    
    if [ -f "$output" ]; then
        echo -e "${YELLOW}Model already exists: $output${NC}"
    else
        echo -e "${GREEN}Downloading $output...${NC}"
        wget --progress=bar:force:noscroll "$url" -O "$output" || {
            echo -e "${RED}Failed to download $output${NC}"
            return 1
        }
    fi
}

# Note: These are placeholder URLs - you'll need to replace with actual model locations
# MobileVLM-V2 quantized (example - actual model needs to be built/quantized)
echo -e "${YELLOW}Note: Model downloads require manual setup or HuggingFace authentication${NC}"
echo -e "${YELLOW}Please download the following models manually:${NC}"
echo "1. MobileVLM-V2 (1.7B, Q4_K_M quantized) -> models/mobilevlm_v2_1.7b_q4.gguf"
echo "2. TinyLlama (1.1B, Q4_K_M quantized) -> models/tinyllama_1.1b_q4.gguf"
echo "3. Vision encoder weights -> models/vision_encoder.pth"
echo ""
echo -e "${GREEN}Automated download script for supported models:${NC}"

# Download TinyLlama if available
if [ ! -f "models/tinyllama_1.1b_q4.gguf" ]; then
    echo "Attempting to download TinyLlama quantized model..."
    # TinyLlama Q4_K_M from TheBloke or similar
    wget -q --show-progress "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
        -O "models/tinyllama_1.1b_q4.gguf" || echo -e "${YELLOW}Manual download required${NC}"
fi

# Set environment variables
echo -e "${GREEN}Setting up environment variables...${NC}"
cat > .env << EOF
# EdgeVLM Environment Variables
EDGEVLM_CONFIG=config.yaml
EDGEVLM_MODELS_DIR=models
EDGEVLM_LOGS_DIR=logs
EDGEVLM_CACHE_DIR=cache
PYTHONUNBUFFERED=1
EOF

# Create systemd service (optional)
echo -e "${GREEN}Creating systemd service...${NC}"
cat > edgevlm.service << EOF
[Unit]
Description=EdgeVLM API Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment="PATH=$(pwd)/venv/bin"
ExecStart=$(pwd)/venv/bin/python main.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

echo -e "${GREEN}To install as a system service, run:${NC}"
echo "  sudo cp edgevlm.service /etc/systemd/system/"
echo "  sudo systemctl enable edgevlm"
echo "  sudo systemctl start edgevlm"

# Performance tuning for Raspberry Pi
echo -e "${GREEN}Applying Raspberry Pi performance tunings...${NC}"
cat > performance_tuning.sh << 'EOF'
#!/bin/bash
# Performance tuning for Raspberry Pi

# Increase swap size (if needed)
sudo dphys-swapfile swapoff || true
sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile 2>/dev/null || true
sudo dphys-swapfile setup || true
sudo dphys-swapfile swapon || true

# CPU governor to performance mode
echo "Setting CPU governor to performance mode..."
for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
    echo performance | sudo tee $cpu/cpufreq/scaling_governor > /dev/null 2>&1 || true
done

# Increase file descriptor limits
ulimit -n 4096

echo "Performance tuning applied!"
EOF

chmod +x performance_tuning.sh

echo ""
echo -e "${GREEN}======================================"
echo "Setup Complete!"
echo "======================================${NC}"
echo ""
echo "Next steps:"
echo "1. Download required models (see instructions above)"
echo "2. Activate virtual environment: source venv/bin/activate"
echo "3. Run performance tuning: ./performance_tuning.sh"
echo "4. Start the API server: python main.py"
echo ""
echo "For optimal performance on Raspberry Pi:"
echo "  - Ensure adequate cooling"
echo "  - Use a high-quality power supply (5V 3A minimum)"
echo "  - Monitor temperature: vcgencmd measure_temp"
echo ""
echo -e "${YELLOW}Documentation available in docs/ directory${NC}"

