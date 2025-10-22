"""
Camera Client Example
Shows how cameras should register and communicate with EdgeVLM
"""

import requests
import time
import base64
import json
import cv2
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any
import threading
import logging


class CameraClient:
    """
    Example camera client that registers with EdgeVLM and sends frames
    """
    
    def __init__(
        self,
        camera_name: str,
        location: str,
        source_type: str,
        source_url: str = None,
        device_id: int = None,
        api_base_url: str = "http://localhost:8002"
    ):
        self.camera_name = camera_name
        self.location = location
        self.source_type = source_type
        self.source_url = source_url
        self.device_id = device_id
        self.api_base_url = api_base_url
        
        self.camera_id = None
        self.api_key = None
        self.endpoints = None
        self.capture = None
        self.is_running = False
        self.frame_count = 0
        self.last_task = "caption"
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"Camera-{camera_name}")
    
    def register(self) -> bool:
        """Register camera with EdgeVLM service"""
        registration_data = {
            "name": self.camera_name,
            "location": self.location,
            "source_type": self.source_type,
            "url": self.source_url,
            "device_id": self.device_id,
            "resolution": [640, 480],
            "fps": 15,
            "capabilities": ["caption", "vqa"],
            "metadata": {
                "client_version": "1.0.0",
                "platform": "raspberry_pi"
            }
        }
        
        try:
            response = requests.post(
                f"{self.api_base_url}/register",
                json=registration_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                self.camera_id = result['camera_id']
                self.api_key = result['api_key']
                self.endpoints = result['endpoints']
                
                self.logger.info(f"Successfully registered camera: {self.camera_id}")
                self.logger.info(f"API Key: {self.api_key}")
                self.logger.info(f"Endpoints: {self.endpoints}")
                return True
            else:
                self.logger.error(f"Registration failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Registration error: {e}")
            return False
    
    def setup_camera(self) -> bool:
        """Setup camera capture based on source type"""
        try:
            if self.source_type == "rtsp":
                self.capture = cv2.VideoCapture(self.source_url)
            elif self.source_type == "http":
                self.capture = cv2.VideoCapture(self.source_url)
            elif self.source_type == "usb":
                self.capture = cv2.VideoCapture(self.device_id or 0)
            elif self.source_type == "file":
                self.capture = cv2.VideoCapture(self.source_url)
            else:
                self.logger.error(f"Unsupported source type: {self.source_type}")
                return False
            
            if not self.capture.isOpened():
                self.logger.error(f"Failed to open camera source")
                return False
            
            # Set camera properties
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.capture.set(cv2.CAP_PROP_FPS, 15)
            
            self.logger.info(f"Camera setup successful: {self.source_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Camera setup error: {e}")
            return False
    
    def send_heartbeat(self) -> bool:
        """Send heartbeat to EdgeVLM service"""
        if not self.camera_id or not self.api_key:
            return False
        
        heartbeat_data = {
            "camera_id": self.camera_id,
            "api_key": self.api_key,
            "status": "active",
            "frame_count": self.frame_count
        }
        
        try:
            response = requests.post(
                self.endpoints['heartbeat'],
                json=heartbeat_data,
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Process commands from server
                for command in result.get('commands', []):
                    self.process_command(command)
                
                return True
            else:
                self.logger.warning(f"Heartbeat failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.warning(f"Heartbeat error: {e}")
            return False
    
    def process_command(self, command: Dict[str, Any]):
        """Process command from EdgeVLM service"""
        command_type = command.get('type')
        
        if command_type == 'adjust_fps':
            new_fps = command.get('value', 15)
            self.capture.set(cv2.CAP_PROP_FPS, new_fps)
            self.logger.info(f"Adjusted FPS to {new_fps}")
            
        elif command_type == 'set_task':
            self.last_task = command.get('value', 'caption')
            self.logger.info(f"Set task to {self.last_task}")
            
        elif command_type == 'change_resolution':
            resolution = command.get('value', [640, 480])
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            self.logger.info(f"Changed resolution to {resolution}")
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a frame from camera"""
        if not self.capture or not self.capture.isOpened():
            return None
        
        ret, frame = self.capture.read()
        if not ret:
            return None
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb
    
    def encode_frame(self, frame: np.ndarray) -> str:
        """Encode frame as base64"""
        # Convert numpy array to PIL Image
        from PIL import Image
        image = Image.fromarray(frame)
        
        # Convert to base64
        import io
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        image_data = buffer.getvalue()
        
        return base64.b64encode(image_data).decode('utf-8')
    
    def upload_frame(self, frame: np.ndarray, task: str = "caption", question: str = None) -> Optional[Dict[str, Any]]:
        """Upload frame to EdgeVLM service"""
        if not self.camera_id or not self.api_key:
            return None
        
        # Encode frame
        frame_data = self.encode_frame(frame)
        
        upload_data = {
            "camera_id": self.camera_id,
            "api_key": self.api_key,
            "frame_data": frame_data,
            "timestamp": time.time(),
            "frame_number": self.frame_count,
            "task": task,
            "question": question
        }
        
        try:
            response = requests.post(
                self.endpoints['upload_frame'],
                json=upload_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                self.frame_count += 1
                return result
            else:
                self.logger.warning(f"Frame upload failed: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.warning(f"Frame upload error: {e}")
            return None
    
    def run_heartbeat_loop(self):
        """Run heartbeat in background thread"""
        while self.is_running:
            self.send_heartbeat()
            time.sleep(30)  # Send heartbeat every 30 seconds
    
    def run_camera_loop(self):
        """Main camera processing loop"""
        self.logger.info("Starting camera processing loop...")
        
        while self.is_running:
            try:
                # Capture frame
                frame = self.capture_frame()
                if frame is None:
                    self.logger.warning("Failed to capture frame")
                    time.sleep(1)
                    continue
                
                # Upload frame for processing
                result = self.upload_frame(frame, self.last_task)
                
                if result:
                    # Log result
                    if self.last_task == "caption":
                        caption = result['result']['caption']
                        latency = result['processing_time']
                        self.logger.info(f"Caption: {caption} ({latency:.2f}s)")
                    elif self.last_task == "vqa":
                        answer = result['result']['answer']
                        latency = result['processing_time']
                        self.logger.info(f"Answer: {answer} ({latency:.2f}s)")
                
                # Control frame rate
                time.sleep(1.0 / 15)  # 15 FPS
                
            except KeyboardInterrupt:
                self.logger.info("Camera loop interrupted")
                break
            except Exception as e:
                self.logger.error(f"Camera loop error: {e}")
                time.sleep(1)
    
    def start(self):
        """Start camera client"""
        self.logger.info(f"Starting camera client: {self.camera_name}")
        
        # Register camera
        if not self.register():
            self.logger.error("Failed to register camera")
            return False
        
        # Setup camera
        if not self.setup_camera():
            self.logger.error("Failed to setup camera")
            return False
        
        # Start processing
        self.is_running = True
        
        # Start heartbeat thread
        heartbeat_thread = threading.Thread(target=self.run_heartbeat_loop, daemon=True)
        heartbeat_thread.start()
        
        # Start camera loop
        try:
            self.run_camera_loop()
        finally:
            self.stop()
    
    def stop(self):
        """Stop camera client"""
        self.logger.info("Stopping camera client...")
        self.is_running = False
        
        if self.capture:
            self.capture.release()
        
        self.logger.info("Camera client stopped")


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="EdgeVLM Camera Client")
    parser.add_argument("--name", type=str, default="test_camera", help="Camera name")
    parser.add_argument("--location", type=str, default="test_location", help="Camera location")
    parser.add_argument("--type", type=str, default="usb", choices=["rtsp", "http", "usb", "file"], help="Camera type")
    parser.add_argument("--url", type=str, help="Camera URL (for rtsp/http/file)")
    parser.add_argument("--device", type=int, default=0, help="USB device ID")
    parser.add_argument("--api-url", type=str, default="http://localhost:8002", help="EdgeVLM API URL")
    
    args = parser.parse_args()
    
    # Create camera client
    client = CameraClient(
        camera_name=args.name,
        location=args.location,
        source_type=args.type,
        source_url=args.url,
        device_id=args.device,
        api_base_url=args.api_url
    )
    
    try:
        client.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        client.stop()


if __name__ == "__main__":
    main()
