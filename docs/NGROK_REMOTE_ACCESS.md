# EdgeVLM Remote Access with Ngrok

This guide shows how to expose EdgeVLM services remotely using ngrok so cameras from anywhere can connect.

## Overview

With ngrok integration, you can:
- **Expose EdgeVLM services** to the internet
- **Connect cameras from anywhere** (not just local network)
- **Maintain secure connections** with API keys
- **Monitor remote cameras** in real-time

## Quick Start

### 1. Setup Ngrok (One-time)

```bash
# Install ngrok (if not already installed)
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo 'deb https://ngrok-agent.s3.amazonaws.com buster main' | sudo tee /etc/apt/sources.list.d/ngrok.list
sudo apt update && sudo apt install ngrok

# Authenticate ngrok (get token from https://dashboard.ngrok.com/get-started/your-authtoken)
ngrok config add-authtoken YOUR_TOKEN
```

### 2. Start Remote Services

```bash
# Start all services with ngrok tunnels
./setup_remote.sh
```

This will:
- Start EdgeVLM APIs (ports 8000, 8002)
- Create ngrok tunnels
- Display public URLs
- Save configuration

### 3. Connect Remote Cameras

```bash
# Use the ngrok URL from setup_remote.sh output
python examples/remote_camera_client.py \
  --ngrok-url "https://abc123.ngrok.io" \
  --name "Office Camera" \
  --location "Office Building" \
  --type usb \
  --device 0
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Remote Cameras                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Office     │  │   Home       │  │   Mobile     │     │
│  │   Camera     │  │   Camera     │  │   App        │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                 │                 │              │
│         └─────────────────┼─────────────────┘              │
└───────────────────────────┼─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                        Internet                              │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                    Ngrok Tunnels                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  https://abc123.ngrok.io -> localhost:8002          │   │
│  │  https://def456.ngrok.io -> localhost:8000          │   │
│  └─────────────────────┬───────────────────────────────┘   │
└─────────────────────────┼───────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                EdgeVLM Services (Raspberry Pi)               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Camera Registration API (port 8002)                │   │
│  │  Main API (port 8000)                              │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Manual Setup

### 1. Start EdgeVLM Services

```bash
# Terminal 1: Start main API
python main.py --port 8000

# Terminal 2: Start camera registration API
python api_camera_registration.py --port 8002
```

### 2. Start Ngrok Tunnels

```bash
# Terminal 3: Expose camera registration API
ngrok http 8002

# Terminal 4: Expose main API (optional)
ngrok http 8000
```

### 3. Get Tunnel URLs

```bash
# Get tunnel URLs from ngrok API
curl http://localhost:4040/api/tunnels | python3 -c "
import sys, json
data = json.load(sys.stdin)
for tunnel in data['tunnels']:
    print(f'{tunnel[\"name\"]}: {tunnel[\"public_url\"]} -> {tunnel[\"config\"][\"addr\"]}')
"
```

## Camera Client Usage

### Basic Remote Camera

```python
from examples.remote_camera_client import RemoteCameraClient

# Create remote camera client
camera = RemoteCameraClient(
    ngrok_url="https://abc123.ngrok.io",
    camera_name="Office Camera",
    location="Office Building",
    source_type="usb",
    device_id=0
)

# Start camera (registers and processes frames)
camera.start()
```

### RTSP Camera

```python
camera = RemoteCameraClient(
    ngrok_url="https://abc123.ngrok.io",
    camera_name="Security Camera",
    location="Front Door",
    source_type="rtsp",
    url="rtsp://192.168.1.100:554/stream1",
    username="admin",
    password="password"
)

camera.start()
```

### HTTP Camera

```python
camera = RemoteCameraClient(
    ngrok_url="https://abc123.ngrok.io",
    camera_name="Web Camera",
    location="Living Room",
    source_type="http",
    url="http://192.168.1.101:8080/video"
)

camera.start()
```

## Command Line Usage

### USB Camera

```bash
python examples/remote_camera_client.py \
  --ngrok-url "https://abc123.ngrok.io" \
  --name "Office Camera" \
  --location "Office Building" \
  --type usb \
  --device 0
```

### RTSP Camera

```bash
python examples/remote_camera_client.py \
  --ngrok-url "https://abc123.ngrok.io" \
  --name "Security Camera" \
  --location "Front Door" \
  --type rtsp \
  --url "rtsp://192.168.1.100:554/stream1"
```

### HTTP Camera

```bash
python examples/remote_camera_client.py \
  --ngrok-url "https://abc123.ngrok.io" \
  --name "Web Camera" \
  --location "Living Room" \
  --type http \
  --url "http://192.168.1.101:8080/video"
```

## Monitoring

### Check Service Status

```bash
# Check EdgeVLM services
curl https://abc123.ngrok.io/health
curl https://def456.ngrok.io/health

# Check registered cameras
curl https://abc123.ngrok.io/cameras
```

### Monitor Script

```bash
# Run monitoring script
./monitor_services.sh
```

Output:
```
EdgeVLM Remote Services Monitor
================================
Service Status:
Main API (8000): healthy
Registration API (8002): healthy

Ngrok Tunnels:
command_line: https://abc123.ngrok.io -> localhost:8002
command_line: https://def456.ngrok.io -> localhost:8000

Registered Cameras:
Office Camera: active (uptime: 1234.5s)
Security Camera: active (uptime: 567.8s)
```

## Security Considerations

### API Key Security

```python
# Store API keys securely
import os
from cryptography.fernet import Fernet

class SecureRemoteClient:
    def __init__(self):
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)
    
    def store_credentials(self, camera_id, api_key):
        credentials = f"{camera_id}:{api_key}"
        encrypted = self.cipher.encrypt(credentials.encode())
        
        with open('.camera_credentials', 'wb') as f:
            f.write(encrypted)
```

### HTTPS in Production

```bash
# Use custom domains with SSL
ngrok http 8002 --hostname=your-domain.com
```

### Firewall Configuration

```bash
# Only allow necessary ports
sudo ufw allow 22    # SSH
sudo ufw allow 8000  # Main API (if needed locally)
sudo ufw allow 8002  # Registration API (if needed locally)
sudo ufw enable
```

## Troubleshooting

### Common Issues

#### 1. Ngrok Authentication Failed

```bash
# Check authentication
ngrok config check

# Re-authenticate
ngrok config add-authtoken YOUR_TOKEN
```

#### 2. Tunnel Not Created

```bash
# Check ngrok status
curl http://localhost:4040/api/tunnels

# Restart ngrok
pkill ngrok
ngrok http 8002
```

#### 3. Camera Registration Failed

```bash
# Check service health
curl https://your-ngrok-url.ngrok.io/health

# Check logs
tail -f logs/registration_api.log
```

#### 4. Connection Timeout

```bash
# Check network connectivity
ping your-ngrok-url.ngrok.io

# Check firewall
sudo ufw status
```

### Debug Mode

```bash
# Start with debug logging
python examples/remote_camera_client.py \
  --ngrok-url "https://abc123.ngrok.io" \
  --name "Debug Camera" \
  --location "Debug Location" \
  --type usb \
  --device 0 \
  --debug
```

## Production Deployment

### Systemd Service

```ini
[Unit]
Description=EdgeVLM Remote Services
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/edgevlm
ExecStart=/home/pi/edgevlm/setup_remote.sh
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Custom Domains

```bash
# Use custom domain with ngrok
ngrok http 8002 --hostname=your-domain.com

# Or use ngrok config file
cat > ~/.config/ngrok/ngrok.yml << EOF
version: "2"
authtoken: YOUR_TOKEN
tunnels:
  edgevlm:
    proto: http
    addr: 8002
    hostname: your-domain.com
EOF

ngrok start edgevlm
```

### Load Balancing

```nginx
upstream edgevlm_remote {
    server localhost:8002;
}

server {
    listen 443 ssl;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://edgevlm_remote;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Examples

### Multiple Cameras

```bash
# Camera 1: Office
python examples/remote_camera_client.py \
  --ngrok-url "https://abc123.ngrok.io" \
  --name "Office Camera" \
  --location "Office" \
  --type usb --device 0 &

# Camera 2: Home
python examples/remote_camera_client.py \
  --ngrok-url "https://abc123.ngrok.io" \
  --name "Home Camera" \
  --location "Home" \
  --type rtsp \
  --url "rtsp://192.168.1.100:554/stream1" &

# Camera 3: Mobile
python examples/remote_camera_client.py \
  --ngrok-url "https://abc123.ngrok.io" \
  --name "Mobile Camera" \
  --location "Mobile" \
  --type http \
  --url "http://192.168.1.101:8080/video" &
```

### Monitoring Dashboard

```python
import requests
import time

def monitor_remote_cameras(ngrok_url):
    while True:
        try:
            # Get camera status
            response = requests.get(f"{ngrok_url}/cameras")
            cameras = response.json()
            
            print(f"\nRemote Cameras Status ({time.strftime('%H:%M:%S')}):")
            print("=" * 50)
            
            for camera_id, info in cameras.items():
                status = info['status']
                uptime = info['uptime_seconds']
                frames = info['frame_count']
                
                print(f"{info['name']}: {status}")
                print(f"  Location: {info['location']}")
                print(f"  Uptime: {uptime:.1f}s")
                print(f"  Frames: {frames}")
                print()
            
            time.sleep(30)  # Update every 30 seconds
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10)

# Usage
monitor_remote_cameras("https://abc123.ngrok.io")
```

## Next Steps

1. **Test with local camera** first
2. **Add remote cameras** one by one
3. **Monitor performance** and adjust settings
4. **Implement security** measures
5. **Scale to multiple locations**

For more details, see the main documentation in `README.md` and `docs/CAMERA_REGISTRATION.md`.
