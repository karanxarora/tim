# Remote Camera Integration Guide

This guide explains how to integrate EdgeVLM with remote cameras for real-world deployment scenarios.

## Overview

EdgeVLM Remote Camera API extends the base system to support various camera protocols commonly used in surveillance, IoT, and edge AI applications:

- **RTSP**: Real-Time Streaming Protocol (most IP cameras)
- **HTTP**: HTTP-based video streams
- **MJPEG**: Motion JPEG streams
- **USB**: Local USB cameras
- **File**: Video files (for testing)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Remote Cameras                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   IP Camera  │  │  Web Camera  │  │  USB Camera  │     │
│  │   (RTSP)     │  │   (HTTP)     │  │              │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                 │                 │              │
│         └─────────────────┼─────────────────┘              │
└───────────────────────────┼─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                EdgeVLM Remote API                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            Camera Manager                           │   │
│  │  • Multi-protocol support                          │   │
│  │  • Connection management                           │   │
│  │  • Frame buffering                                 │   │
│  └─────────────────────┬───────────────────────────────┘   │
│                        │                                   │
│  ┌─────────────────────▼───────────────────────────────┐   │
│  │            EdgeVLM Pipeline                        │   │
│  │  • Vision processing                               │   │
│  │  • Inference engine                                │   │
│  │  • Optimizations                                   │   │
│  └─────────────────────┬───────────────────────────────┘   │
│                        │                                   │
│  ┌─────────────────────▼───────────────────────────────┐   │
│  │            REST API                                 │   │
│  │  • Camera management                               │   │
│  │  • Frame processing                                │   │
│  │  • Live streaming                                  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Start Remote Camera API

```bash
# Start the remote camera API (different port from main API)
python api_remote.py --port 8001

# Or with auto-reload for development
python api_remote.py --port 8001 --reload
```

### 2. Add a Camera

```bash
# Add RTSP camera
curl -X POST http://localhost:8001/cameras \
  -H "Content-Type: application/json" \
  -d '{
    "name": "front_door",
    "source_type": "rtsp",
    "url": "rtsp://192.168.1.100:554/stream1",
    "username": "admin",
    "password": "password",
    "resolution": [640, 480],
    "fps": 15
  }'
```

### 3. Start Camera

```bash
curl -X POST http://localhost:8001/cameras/front_door/start
```

### 4. Process Frames

```bash
# Generate caption
curl -X POST http://localhost:8001/process \
  -H "Content-Type: application/json" \
  -d '{
    "camera_name": "front_door",
    "task": "caption"
  }'

# Answer question
curl -X POST http://localhost:8001/process \
  -H "Content-Type: application/json" \
  -d '{
    "camera_name": "front_door",
    "task": "vqa",
    "question": "Is there a person in the image?"
  }'
```

## Camera Configuration

### RTSP Cameras (Most Common)

```python
config = {
    "name": "ip_camera",
    "source_type": "rtsp",
    "url": "rtsp://192.168.1.100:554/stream1",
    "username": "admin",
    "password": "password",
    "resolution": [640, 480],
    "fps": 15,
    "timeout": 10
}
```

**Common RTSP URLs**:
- `rtsp://ip:554/stream1` - Standard
- `rtsp://ip:554/live/ch0` - Some DVRs
- `rtsp://ip:554/cam/realmonitor?channel=1&subtype=0` - Hikvision
- `rtsp://ip:554/axis-media/media.amp` - Axis cameras

### HTTP Cameras

```python
config = {
    "name": "web_camera",
    "source_type": "http",
    "url": "http://192.168.1.101:8080/video",
    "resolution": [640, 480],
    "fps": 15
}
```

### MJPEG Streams

```python
config = {
    "name": "mjpeg_camera",
    "source_type": "mjpeg",
    "url": "http://192.168.1.102:8080/mjpeg",
    "resolution": [640, 480],
    "fps": 10
}
```

### USB Cameras

```python
config = {
    "name": "usb_cam",
    "source_type": "usb",
    "device_id": 0,
    "resolution": [640, 480],
    "fps": 30
}
```

## API Endpoints

### Camera Management

#### Add Camera
```http
POST /cameras
Content-Type: application/json

{
  "name": "camera_name",
  "source_type": "rtsp|http|mjpeg|usb|file",
  "url": "camera_url",
  "username": "optional_username",
  "password": "optional_password",
  "resolution": [640, 480],
  "fps": 15
}
```

#### List Cameras
```http
GET /cameras
```

Response:
```json
{
  "camera1": {
    "name": "camera1",
    "running": true,
    "connected": true,
    "frame_count": 1234,
    "config": {
      "source_type": "rtsp",
      "resolution": [640, 480],
      "fps": 15
    }
  }
}
```

#### Start Camera
```http
POST /cameras/{camera_name}/start
```

#### Stop Camera
```http
POST /cameras/{camera_name}/stop
```

#### Test Camera
```http
POST /cameras/{camera_name}/test
```

### Frame Processing

#### Process Frame
```http
POST /process
Content-Type: application/json

{
  "camera_name": "camera_name",
  "task": "caption|vqa",
  "question": "optional_question",
  "max_length": 128
}
```

#### Process Latest Frame (GET)
```http
GET /process/latest?camera_name=camera1&task=caption&question=What%20is%20this?
```

### Live Streaming

#### Stream Processing
```http
GET /stream/{camera_name}?task=caption&interval=5.0
```

Returns Server-Sent Events (SSE) stream:
```
data: {"camera_name": "camera1", "result": {"caption": "A person walking..."}, ...}

data: {"camera_name": "camera1", "result": {"caption": "A car in the driveway"}, ...}
```

## Python Client Examples

### Basic Usage

```python
import requests
import time

# Client class
class CameraClient:
    def __init__(self, base_url="http://localhost:8001"):
        self.base_url = base_url
    
    def add_camera(self, name, source_type, url, **kwargs):
        config = {
            "name": name,
            "source_type": source_type,
            "url": url,
            **kwargs
        }
        response = requests.post(f"{self.base_url}/cameras", json=config)
        return response.status_code == 200
    
    def start_camera(self, name):
        response = requests.post(f"{self.base_url}/cameras/{name}/start")
        return response.status_code == 200
    
    def process_frame(self, camera_name, task="caption", question=None):
        data = {
            "camera_name": camera_name,
            "task": task,
            "question": question
        }
        response = requests.post(f"{self.base_url}/process", json=data)
        return response.json()

# Usage
client = CameraClient()

# Add and start camera
client.add_camera("door_cam", "rtsp", "rtsp://192.168.1.100:554/stream1")
client.start_camera("door_cam")

# Process frames
for i in range(10):
    result = client.process_frame("door_cam", "caption")
    print(f"Frame {i}: {result['result']['caption']}")
    time.sleep(5)
```

### Advanced Usage

```python
import asyncio
import aiohttp

class AsyncCameraClient:
    def __init__(self, base_url="http://localhost:8001"):
        self.base_url = base_url
    
    async def process_stream(self, camera_name, task="caption", interval=5.0):
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/stream/{camera_name}"
            params = {"task": task, "interval": interval}
            
            async with session.get(url, params=params) as response:
                async for line in response.content:
                    if line.startswith(b'data: '):
                        data = json.loads(line[6:])
                        yield data

# Usage
async def main():
    client = AsyncCameraClient()
    
    async for result in client.process_stream("door_cam", "caption"):
        print(f"Caption: {result['result']['caption']}")

asyncio.run(main())
```

## Integration Patterns

### 1. Surveillance System

```python
# Monitor multiple cameras
cameras = ["front_door", "back_door", "garage"]

for camera in cameras:
    # Process each camera
    result = client.process_frame(camera, "vqa", "Is there a person?")
    
    if "person" in result['result']['answer'].lower():
        # Alert system
        send_alert(f"Person detected on {camera}")
```

### 2. IoT Integration

```python
# Process camera data and send to IoT platform
def process_and_upload():
    result = client.process_frame("sensor_cam", "caption")
    
    # Send to IoT platform
    iot_data = {
        "timestamp": result['frame_timestamp'],
        "description": result['result']['caption'],
        "location": "entrance"
    }
    
    requests.post("https://iot-platform.com/api/data", json=iot_data)
```

### 3. Real-time Dashboard

```python
# WebSocket integration for real-time updates
import websocket

def on_message(ws, message):
    data = json.loads(message)
    update_dashboard(data)

# Connect to camera stream
ws = websocket.WebSocketApp(
    "ws://localhost:8001/stream/door_cam",
    on_message=on_message
)
ws.run_forever()
```

## Performance Considerations

### Network Optimization

1. **Use appropriate resolution**:
   ```python
   # Lower resolution for better performance
   "resolution": [320, 240]  # Instead of [640, 480]
   ```

2. **Adjust frame rate**:
   ```python
   # Lower FPS for less network load
   "fps": 10  # Instead of 30
   ```

3. **Buffer management**:
   ```python
   # Smaller buffer for lower latency
   "buffer_size": 5  # Instead of 10
   ```

### Memory Management

1. **Process frames selectively**:
   ```python
   # Process every 5th frame
   if frame_count % 5 == 0:
       result = client.process_frame(camera_name)
   ```

2. **Clear cache periodically**:
   ```python
   # Clear cache every 100 frames
   if frame_count % 100 == 0:
       requests.post(f"{base_url}/clear-cache")
   ```

### Error Handling

```python
def robust_process_frame(camera_name, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = client.process_frame(camera_name)
            return result
        except requests.Timeout:
            print(f"Timeout on attempt {attempt + 1}")
            time.sleep(1)
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            time.sleep(1)
    
    return None
```

## Troubleshooting

### Common Issues

#### 1. Camera Connection Failed

**Symptoms**: Camera shows as not connected

**Solutions**:
- Check camera URL and credentials
- Verify network connectivity
- Test with VLC or similar player
- Check camera supports the protocol

#### 2. No Frames Available

**Symptoms**: "No frame available" error

**Solutions**:
- Ensure camera is started
- Check camera is actually streaming
- Increase timeout in config
- Verify camera resolution/fps settings

#### 3. High Latency

**Symptoms**: Slow processing

**Solutions**:
- Reduce camera resolution
- Lower frame rate
- Enable early exit
- Use speculative decoding
- Check network bandwidth

#### 4. Memory Issues

**Symptoms**: Out of memory errors

**Solutions**:
- Process frames less frequently
- Clear cache more often
- Reduce buffer size
- Use lower resolution

### Debugging

```python
# Check camera status
status = client.list_cameras()
print(json.dumps(status, indent=2))

# Test camera connection
test_result = client.test_camera("camera_name")
print(test_result)

# Monitor health
health = client.get_health()
print(f"Active cameras: {health['active_cameras']}")
```

## Security Considerations

### Network Security

1. **Use HTTPS/WSS** for production:
   ```python
   # Use secure protocols
   base_url = "https://your-pi-ip:8001"
   ```

2. **Authentication**:
   ```python
   # Add API key authentication
   headers = {"Authorization": "Bearer your-api-key"}
   ```

3. **Camera credentials**:
   ```python
   # Store credentials securely
   config = {
       "username": os.getenv("CAMERA_USERNAME"),
       "password": os.getenv("CAMERA_PASSWORD")
   }
   ```

### Access Control

1. **Firewall rules**:
   ```bash
   # Only allow specific IPs
   ufw allow from 192.168.1.0/24 to any port 8001
   ```

2. **VPN access** for remote cameras

## Production Deployment

### Systemd Service

```ini
[Unit]
Description=EdgeVLM Remote Camera API
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/edgevlm
Environment="PATH=/home/pi/edgevlm/venv/bin"
ExecStart=/home/pi/edgevlm/venv/bin/python api_remote.py --port 8001
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

EXPOSE 8001

CMD ["python", "api_remote.py", "--port", "8001"]
```

### Load Balancing

```nginx
upstream edgevlm_cameras {
    server 192.168.1.10:8001;
    server 192.168.1.11:8001;
    server 192.168.1.12:8001;
}

server {
    listen 80;
    location / {
        proxy_pass http://edgevlm_cameras;
    }
}
```

## Examples

See `examples/remote_camera_demo.py` for complete working examples including:
- RTSP camera integration
- HTTP camera setup
- USB camera usage
- Multiple camera management
- Health monitoring
- Error handling

## Next Steps

1. **Start with USB camera** for testing
2. **Add IP camera** for real deployment
3. **Implement monitoring** for production
4. **Add authentication** for security
5. **Scale horizontally** with multiple Pis

For more details, see the main documentation in `README.md` and `docs/API_REFERENCE.md`.
