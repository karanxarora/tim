# Camera Registration API Guide

This guide explains how cameras self-register with EdgeVLM and establish persistent connections for real-time processing.

## Overview

The Camera Registration API implements a **camera-initiated connection model** where:

1. **Cameras register themselves** with EdgeVLM service
2. **Persistent connections** are maintained via heartbeats
3. **Frames are uploaded** for processing on-demand
4. **Commands are sent** from EdgeVLM to cameras for control

This is ideal for IoT/surveillance deployments where cameras are distributed and need to self-register.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Camera Devices                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   IP Camera  │  │  USB Camera  │  │  Mobile App  │     │
│  │   (RTSP)     │  │              │  │              │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                 │                 │              │
│         └─────────────────┼─────────────────┘              │
└───────────────────────────┼─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│              EdgeVLM Camera Registration API                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            Registration Service                     │   │
│  │  • Camera registration                             │   │
│  │  • API key management                              │   │
│  │  • Endpoint provisioning                           │   │
│  └─────────────────────┬───────────────────────────────┘   │
│                        │                                   │
│  ┌─────────────────────▼───────────────────────────────┐   │
│  │            Connection Manager                       │   │
│  │  • Heartbeat monitoring                            │   │
│  │  • Session management                              │   │
│  │  • Command distribution                            │   │
│  └─────────────────────┬───────────────────────────────┘   │
│                        │                                   │
│  ┌─────────────────────▼───────────────────────────────┐   │
│  │            EdgeVLM Pipeline                        │   │
│  │  • Frame processing                                │   │
│  │  • Inference engine                                │   │
│  │  • Optimizations                                   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Start Camera Registration API

```bash
# Start the camera registration API (port 8002)
python api_camera_registration.py --port 8002

# Or with auto-reload for development
python api_camera_registration.py --port 8002 --reload
```

### 2. Camera Registration Flow

```python
# Step 1: Camera registers with EdgeVLM
registration_data = {
    "name": "Front Door Camera",
    "location": "Main Entrance",
    "source_type": "rtsp",
    "url": "rtsp://192.168.1.100:554/stream1",
    "username": "admin",
    "password": "password",
    "resolution": [640, 480],
    "fps": 15,
    "capabilities": ["caption", "vqa"]
}

response = requests.post("http://localhost:8002/register", json=registration_data)
result = response.json()

camera_id = result['camera_id']
api_key = result['api_key']
endpoints = result['endpoints']
```

### 3. Camera Heartbeat

```python
# Step 2: Camera sends periodic heartbeats
heartbeat_data = {
    "camera_id": camera_id,
    "api_key": api_key,
    "status": "active",
    "frame_count": 1234
}

response = requests.post(endpoints['heartbeat'], json=heartbeat_data)
commands = response.json()['commands']  # Process any commands from server
```

### 4. Frame Upload

```python
# Step 3: Camera uploads frames for processing
frame_data = {
    "camera_id": camera_id,
    "api_key": api_key,
    "frame_data": base64_encoded_image,
    "timestamp": time.time(),
    "frame_number": 1234,
    "task": "caption"
}

response = requests.post(endpoints['upload_frame'], json=frame_data)
result = response.json()  # Contains processing results
```

## API Endpoints

### Registration

#### Register Camera
```http
POST /register
Content-Type: application/json

{
  "name": "Camera Name",
  "location": "Camera Location",
  "source_type": "rtsp|http|usb|file",
  "url": "camera_url",
  "username": "optional_username",
  "password": "optional_password",
  "resolution": [640, 480],
  "fps": 15,
  "capabilities": ["caption", "vqa"],
  "metadata": {}
}
```

**Response**:
```json
{
  "camera_id": "uuid",
  "api_key": "api_key",
  "status": "registered",
  "message": "Camera registered successfully",
  "endpoints": {
    "heartbeat": "http://localhost:8002/cameras/{camera_id}/heartbeat",
    "upload_frame": "http://localhost:8002/cameras/{camera_id}/upload",
    "status": "http://localhost:8002/cameras/{camera_id}/status"
  },
  "config": {
    "resolution": [640, 480],
    "fps": 15,
    "capabilities": ["caption", "vqa"]
  }
}
```

### Camera Communication

#### Send Heartbeat
```http
POST /cameras/{camera_id}/heartbeat
Content-Type: application/json

{
  "camera_id": "camera_id",
  "api_key": "api_key",
  "status": "active",
  "frame_count": 1234,
  "error_message": "optional_error"
}
```

**Response**:
```json
{
  "status": "ok",
  "message": "Heartbeat received",
  "commands": [
    {
      "type": "adjust_fps",
      "value": 10,
      "reason": "High CPU usage"
    },
    {
      "type": "set_task",
      "value": "caption"
    }
  ]
}
```

#### Upload Frame
```http
POST /cameras/{camera_id}/upload
Content-Type: application/json

{
  "camera_id": "camera_id",
  "api_key": "api_key",
  "frame_data": "base64_encoded_image",
  "timestamp": 1234567890.123,
  "frame_number": 1234,
  "task": "caption|vqa",
  "question": "optional_question"
}
```

**Response**:
```json
{
  "camera_id": "camera_id",
  "task": "caption",
  "result": {
    "caption": "A person walking in the hallway",
    "latency": 2.345,
    "tokens_per_second": 18.5
  },
  "processing_time": 2.456,
  "timestamp": "2025-10-22T10:30:45.123456"
}
```

### Management

#### List Cameras
```http
GET /cameras
```

#### Get Camera Status
```http
GET /cameras/{camera_id}/status?api_key=optional_api_key
```

#### Start Camera Processing
```http
POST /cameras/{camera_id}/start?api_key=optional_api_key
```

#### Stop Camera Processing
```http
POST /cameras/{camera_id}/stop?api_key=optional_api_key
```

## Camera Client Implementation

### Basic Camera Client

```python
import requests
import time
import base64
import cv2
import numpy as np
from datetime import datetime

class CameraClient:
    def __init__(self, name, location, source_type, **kwargs):
        self.name = name
        self.location = location
        self.source_type = source_type
        self.api_base_url = "http://localhost:8002"
        self.camera_id = None
        self.api_key = None
        self.endpoints = None
        self.capture = None
        self.is_running = False
    
    def register(self):
        """Register camera with EdgeVLM"""
        data = {
            "name": self.name,
            "location": self.location,
            "source_type": self.source_type,
            "resolution": [640, 480],
            "fps": 15,
            "capabilities": ["caption", "vqa"]
        }
        
        response = requests.post(f"{self.api_base_url}/register", json=data)
        if response.status_code == 200:
            result = response.json()
            self.camera_id = result['camera_id']
            self.api_key = result['api_key']
            self.endpoints = result['endpoints']
            return True
        return False
    
    def send_heartbeat(self):
        """Send heartbeat to EdgeVLM"""
        data = {
            "camera_id": self.camera_id,
            "api_key": self.api_key,
            "status": "active",
            "frame_count": self.frame_count
        }
        
        response = requests.post(self.endpoints['heartbeat'], json=data)
        if response.status_code == 200:
            commands = response.json().get('commands', [])
            self.process_commands(commands)
            return True
        return False
    
    def upload_frame(self, frame, task="caption", question=None):
        """Upload frame for processing"""
        # Encode frame as base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_data = base64.b64encode(buffer).decode('utf-8')
        
        data = {
            "camera_id": self.camera_id,
            "api_key": self.api_key,
            "frame_data": frame_data,
            "timestamp": time.time(),
            "frame_number": self.frame_count,
            "task": task,
            "question": question
        }
        
        response = requests.post(self.endpoints['upload_frame'], json=data)
        if response.status_code == 200:
            return response.json()
        return None
    
    def process_commands(self, commands):
        """Process commands from EdgeVLM"""
        for command in commands:
            if command['type'] == 'adjust_fps':
                self.capture.set(cv2.CAP_PROP_FPS, command['value'])
            elif command['type'] == 'set_task':
                self.current_task = command['value']
    
    def run(self):
        """Main camera loop"""
        # Register camera
        if not self.register():
            print("Failed to register camera")
            return
        
        # Setup camera
        self.capture = cv2.VideoCapture(0)  # USB camera
        if not self.capture.isOpened():
            print("Failed to open camera")
            return
        
        self.is_running = True
        self.frame_count = 0
        
        try:
            while self.is_running:
                # Capture frame
                ret, frame = self.capture.read()
                if not ret:
                    continue
                
                # Send heartbeat every 30 seconds
                if self.frame_count % 450 == 0:  # 30 seconds at 15fps
                    self.send_heartbeat()
                
                # Upload frame every 5 seconds
                if self.frame_count % 75 == 0:  # 5 seconds at 15fps
                    result = self.upload_frame(frame, "caption")
                    if result:
                        print(f"Caption: {result['result']['caption']}")
                
                self.frame_count += 1
                time.sleep(1/15)  # 15 FPS
                
        except KeyboardInterrupt:
            print("Stopping camera...")
        finally:
            self.capture.release()

# Usage
camera = CameraClient("Office Camera", "Office Room", "usb")
camera.run()
```

### Advanced Camera Client

```python
import asyncio
import aiohttp
import threading

class AsyncCameraClient:
    def __init__(self, name, location, source_type, **kwargs):
        self.name = name
        self.location = location
        self.source_type = source_type
        self.api_base_url = "http://localhost:8002"
        self.camera_id = None
        self.api_key = None
        self.session = None
        self.is_running = False
    
    async def register(self):
        """Async camera registration"""
        data = {
            "name": self.name,
            "location": self.location,
            "source_type": self.source_type,
            "resolution": [640, 480],
            "fps": 15,
            "capabilities": ["caption", "vqa"]
        }
        
        async with self.session.post(f"{self.api_base_url}/register", json=data) as response:
            if response.status == 200:
                result = await response.json()
                self.camera_id = result['camera_id']
                self.api_key = result['api_key']
                return True
            return False
    
    async def send_heartbeat(self):
        """Async heartbeat"""
        data = {
            "camera_id": self.camera_id,
            "api_key": self.api_key,
            "status": "active"
        }
        
        async with self.session.post(
            f"{self.api_base_url}/cameras/{self.camera_id}/heartbeat",
            json=data
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result.get('commands', [])
            return []
    
    async def upload_frame(self, frame_data, task="caption"):
        """Async frame upload"""
        data = {
            "camera_id": self.camera_id,
            "api_key": self.api_key,
            "frame_data": frame_data,
            "timestamp": time.time(),
            "task": task
        }
        
        async with self.session.post(
            f"{self.api_base_url}/cameras/{self.camera_id}/upload",
            json=data
        ) as response:
            if response.status == 200:
                return await response.json()
            return None
    
    async def run(self):
        """Async main loop"""
        async with aiohttp.ClientSession() as session:
            self.session = session
            
            # Register camera
            if not await self.register():
                print("Failed to register camera")
                return
            
            self.is_running = True
            heartbeat_task = asyncio.create_task(self.heartbeat_loop())
            
            try:
                # Main processing loop
                while self.is_running:
                    # Capture and process frame
                    # ... camera capture code ...
                    
                    await asyncio.sleep(1/15)  # 15 FPS
                    
            finally:
                self.is_running = False
                heartbeat_task.cancel()
    
    async def heartbeat_loop(self):
        """Background heartbeat loop"""
        while self.is_running:
            await self.send_heartbeat()
            await asyncio.sleep(30)  # Every 30 seconds

# Usage
async def main():
    camera = AsyncCameraClient("Office Camera", "Office Room", "usb")
    await camera.run()

asyncio.run(main())
```

## Command System

EdgeVLM can send commands to cameras for control:

### Available Commands

1. **adjust_fps**: Change camera frame rate
   ```json
   {
     "type": "adjust_fps",
     "value": 10,
     "reason": "High CPU usage"
   }
   ```

2. **set_task**: Set processing task
   ```json
   {
     "type": "set_task",
     "value": "vqa"
   }
   ```

3. **change_resolution**: Adjust camera resolution
   ```json
   {
     "type": "change_resolution",
     "value": [320, 240]
   }
   ```

4. **set_question**: Set VQA question
   ```json
   {
     "type": "set_question",
     "value": "Is there a person in the image?"
   }
   ```

### Command Processing

```python
def process_commands(self, commands):
    """Process commands from EdgeVLM"""
    for command in commands:
        cmd_type = command['type']
        value = command['value']
        
        if cmd_type == 'adjust_fps':
            self.capture.set(cv2.CAP_PROP_FPS, value)
            print(f"FPS adjusted to {value}")
            
        elif cmd_type == 'set_task':
            self.current_task = value
            print(f"Task set to {value}")
            
        elif cmd_type == 'change_resolution':
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, value[0])
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, value[1])
            print(f"Resolution changed to {value}")
            
        elif cmd_type == 'set_question':
            self.current_question = value
            print(f"Question set to {value}")
```

## Error Handling

### Connection Errors

```python
def robust_heartbeat(self, max_retries=3):
    """Robust heartbeat with retry"""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                self.endpoints['heartbeat'],
                json=self.heartbeat_data,
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
        except requests.Timeout:
            print(f"Heartbeat timeout, attempt {attempt + 1}")
        except requests.ConnectionError:
            print(f"Connection error, attempt {attempt + 1}")
        
        time.sleep(2 ** attempt)  # Exponential backoff
    
    return None
```

### Reconnection Logic

```python
def auto_reconnect(self):
    """Auto-reconnect on connection loss"""
    while self.is_running:
        try:
            # Test connection
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            if response.status_code == 200:
                print("Connection restored")
                return True
        except:
            pass
        
        print("Connection lost, retrying in 30 seconds...")
        time.sleep(30)
    
    return False
```

## Security

### API Key Management

```python
# Store API key securely
import os
from cryptography.fernet import Fernet

class SecureCameraClient:
    def __init__(self):
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)
    
    def store_credentials(self, camera_id, api_key):
        """Store credentials encrypted"""
        credentials = f"{camera_id}:{api_key}"
        encrypted = self.cipher.encrypt(credentials.encode())
        
        with open('.camera_credentials', 'wb') as f:
            f.write(encrypted)
    
    def load_credentials(self):
        """Load credentials from file"""
        try:
            with open('.camera_credentials', 'rb') as f:
                encrypted = f.read()
            
            decrypted = self.cipher.decrypt(encrypted)
            camera_id, api_key = decrypted.decode().split(':')
            return camera_id, api_key
        except:
            return None, None
```

### Network Security

```python
# Use HTTPS in production
api_base_url = "https://your-pi-ip:8002"

# Add certificate verification
import ssl
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE  # For self-signed certs
```

## Monitoring

### Camera Status Dashboard

```python
def monitor_cameras():
    """Monitor all registered cameras"""
    response = requests.get("http://localhost:8002/cameras")
    cameras = response.json()
    
    print("Camera Status Dashboard")
    print("=" * 50)
    
    for camera_id, camera_info in cameras.items():
        status = camera_info['status']
        uptime = camera_info['uptime_seconds']
        frame_count = camera_info['frame_count']
        
        print(f"{camera_info['name']}: {status}")
        print(f"  Uptime: {uptime:.1f}s")
        print(f"  Frames: {frame_count}")
        print(f"  Last heartbeat: {camera_info['last_heartbeat']}")
        print()
```

### Health Monitoring

```python
def health_check():
    """Check system health"""
    response = requests.get("http://localhost:8002/health")
    health = response.json()
    
    print(f"System Status: {health['status']}")
    print(f"Registered Cameras: {health['registered_cameras']}")
    print(f"Active Cameras: {health['active_cameras']}")
    print(f"CPU: {health['system_info']['cpu_percent']:.1f}%")
    print(f"Memory: {health['system_info']['memory_percent']:.1f}%")
```

## Production Deployment

### Systemd Service

```ini
[Unit]
Description=EdgeVLM Camera Registration API
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/edgevlm
Environment="PATH=/home/pi/edgevlm/venv/bin"
ExecStart=/home/pi/edgevlm/venv/bin/python api_camera_registration.py --port 8002
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Load Balancing

```nginx
upstream edgevlm_cameras {
    server 192.168.1.10:8002;
    server 192.168.1.11:8002;
    server 192.168.1.12:8002;
}

server {
    listen 443 ssl;
    server_name camera-api.yourdomain.com;
    
    location / {
        proxy_pass http://edgevlm_cameras;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Examples

See `examples/camera_client.py` and `examples/camera_registration_demo.py` for complete working examples.

## Next Steps

1. **Start with USB camera** for testing
2. **Add IP camera** for real deployment
3. **Implement security** (HTTPS, authentication)
4. **Add monitoring** for production
5. **Scale horizontally** with multiple Pis

For more details, see the main documentation in `README.md` and `docs/API_REFERENCE.md`.
