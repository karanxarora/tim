# EdgeVLM Deployment Guide for Raspberry Pi ARM32

This guide will help you deploy EdgeVLM on a Raspberry Pi running ARM32 (armv7l) architecture.

## Prerequisites

- Raspberry Pi 4 (4GB+ RAM recommended)
- Raspberry Pi OS (64-bit or 32-bit)
- MicroSD card (32GB+ recommended)
- Internet connection
- Camera module (optional, for camera features)

## Quick Start

### 1. Prepare Raspberry Pi

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install git
sudo apt install git -y

# Clone the repository
git clone <your-repo-url>
cd tim
```

### 2. Run ARM32 Setup Script

```bash
# Make setup script executable
chmod +x setup-arm32.sh

# Run the setup script
./setup-arm32.sh
```

This script will:
- Install all system dependencies
- Create a Python virtual environment
- Install ARM32-optimized packages
- Set up system optimizations
- Create a systemd service

### 3. Download Models

```bash
# Activate virtual environment
source venv/bin/activate

# Download ARM32-optimized models
python download_models_arm32.py
```

### 4. Test the System

```bash
# Test basic functionality
python main.py --log-level DEBUG

# Test ngrok integration (optional)
python ngrok_integration.py
```

### 5. Start as Service

```bash
# Start the service
sudo systemctl start edgevlm

# Check status
sudo systemctl status edgevlm

# Enable auto-start
sudo systemctl enable edgevlm
```

## ARM32 Optimizations

### Memory Management
- Reduced model sizes (Q4 quantization)
- Optimized cache settings
- Swap file configuration
- Memory-mapped model loading

### Performance Tuning
- ARM NEON SIMD instructions
- Optimized thread count
- Reduced input resolution
- FP16 disabled (ARM32 compatibility)

### System Configuration
- GPU memory split: 128MB
- Camera interface enabled
- Swap file: 2GB
- Kernel optimizations

## Model Recommendations

### For 4GB Pi:
- **Primary**: MobileVLM V2 1.7B Q4 (1.2GB)
- **Draft**: TinyLlama 1.1B Q4 (700MB)
- **Total**: ~2GB

### For 8GB Pi:
- **Primary**: MobileVLM V2 1.7B Q4 (1.2GB)
- **Draft**: TinyLlama 1.1B Q4 (700MB)
- **Backup**: MobileVLM V2 1.7B Q2 (600MB)
- **Total**: ~2.5GB

## Performance Expectations

### Latency (Raspberry Pi 4):
- Image captioning: 3-8 seconds
- VQA: 4-10 seconds
- Camera processing: 15 FPS (reduced resolution)

### Memory Usage:
- Base system: ~1GB
- Model loading: ~2GB
- Processing: ~500MB
- **Total**: ~3.5GB (4GB Pi)

## Troubleshooting

### Common Issues:

1. **Out of Memory**:
   ```bash
   # Increase swap
   sudo dphys-swapfile swapoff
   sudo sed -i 's/CONF_SWAPSIZE=2048/CONF_SWAPSIZE=4096/' /etc/dphys-swapfile
   sudo dphys-swapfile setup
   sudo dphys-swapfile swapon
   ```

2. **Slow Performance**:
   ```bash
   # Check CPU temperature
   vcgencmd measure_temp
   
   # Check memory usage
   free -h
   
   # Check GPU memory
   vcgencmd get_mem gpu
   ```

3. **Camera Issues**:
   ```bash
   # Enable camera
   sudo raspi-config
   # Navigate to: Interface Options > Camera > Enable
   
   # Test camera
   libcamera-hello --list-cameras
   ```

### Logs and Monitoring:

```bash
# View service logs
sudo journalctl -u edgevlm -f

# View application logs
tail -f logs/edgevlm.log

# Monitor system resources
htop
```

## Remote Access Setup

### Using Ngrok:

1. **Install ngrok**:
   ```bash
   curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
   echo 'deb https://ngrok-agent.s3.amazonaws.com buster main' | sudo tee /etc/apt/sources.list.d/ngrok.list
   sudo apt update && sudo apt install ngrok
   ```

2. **Authenticate**:
   ```bash
   ngrok config add-authtoken <your-token>
   ```

3. **Start remote access**:
   ```bash
   python ngrok_integration.py
   ```

### Using SSH Tunneling:

```bash
# From your local machine
ssh -L 8000:localhost:8000 pi@<pi-ip-address>
```

## Configuration Files

- `config.yaml`: Main configuration (ARM32 optimized)
- `requirements-arm32.txt`: ARM32-compatible packages
- `setup-arm32.sh`: Automated setup script
- `download_models_arm32.py`: Model downloader

## API Endpoints

Once running, the API will be available at:
- `http://localhost:8000` (local)
- `http://<pi-ip>:8000` (network)
- `https://<ngrok-url>` (remote via ngrok)

### Key Endpoints:
- `GET /health` - Health check
- `POST /process` - Process image
- `POST /register` - Register camera
- `GET /docs` - API documentation

## Monitoring and Maintenance

### Regular Maintenance:
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Clean logs
sudo journalctl --vacuum-time=7d

# Check disk space
df -h

# Restart service
sudo systemctl restart edgevlm
```

### Performance Monitoring:
```bash
# CPU temperature
watch -n 1 vcgencmd measure_temp

# Memory usage
watch -n 1 free -h

# Service status
watch -n 5 sudo systemctl status edgevlm
```

## Support

For issues specific to ARM32 deployment:
1. Check the logs: `sudo journalctl -u edgevlm -f`
2. Verify system resources: `htop`
3. Test individual components: `python -c "from api import app; print('OK')"`
4. Check configuration: `python -c "import yaml; print(yaml.safe_load(open('config.yaml')))"`

## Performance Tips

1. **Use a fast microSD card** (Class 10 or better)
2. **Enable GPU memory split** for better performance
3. **Use wired ethernet** for stable network connections
4. **Monitor temperature** to prevent throttling
5. **Close unnecessary services** to free up resources

---

**Note**: This deployment is optimized for Raspberry Pi 4 with 4GB+ RAM. For older Pi models, consider using the Q2 quantized models for better performance.
