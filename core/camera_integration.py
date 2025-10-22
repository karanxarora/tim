"""
Remote Camera Integration for EdgeVLM
Supports various camera protocols and streaming sources
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, Union, Callable
from dataclasses import dataclass
import requests
import cv2
import numpy as np
from urllib.parse import urlparse
import threading
from queue import Queue, Empty
import base64
from io import BytesIO
from PIL import Image


@dataclass
class CameraConfig:
    """Configuration for camera sources"""
    source_type: str  # "rtsp", "http", "mjpeg", "usb", "file"
    url: Optional[str] = None
    device_id: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    resolution: tuple = (640, 480)
    fps: int = 30
    timeout: int = 10
    retry_attempts: int = 3
    buffer_size: int = 10


class RemoteCameraCapture:
    """
    Unified camera capture supporting multiple protocols
    """
    
    def __init__(
        self,
        config: CameraConfig,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.capture = None
        self.is_running = False
        self.frame_queue = Queue(maxsize=config.buffer_size)
        self.capture_thread = None
        self.last_frame_time = 0
        self.frame_count = 0
        
    def _build_camera_url(self) -> str:
        """Build camera URL with authentication if needed"""
        if self.config.source_type in ["rtsp", "http"]:
            url = self.config.url
            if self.config.username and self.config.password:
                parsed = urlparse(url)
                url = f"{parsed.scheme}://{self.config.username}:{self.config.password}@{parsed.netloc}{parsed.path}"
            return url
        return self.config.url or ""
    
    def _initialize_capture(self) -> bool:
        """Initialize camera capture based on source type"""
        try:
            if self.config.source_type == "rtsp":
                return self._init_rtsp()
            elif self.config.source_type == "http":
                return self._init_http()
            elif self.config.source_type == "mjpeg":
                return self._init_mjpeg()
            elif self.config.source_type == "usb":
                return self._init_usb()
            elif self.config.source_type == "file":
                return self._init_file()
            else:
                self.logger.error(f"Unsupported camera type: {self.config.source_type}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {e}")
            return False
    
    def _init_rtsp(self) -> bool:
        """Initialize RTSP camera"""
        url = self._build_camera_url()
        self.capture = cv2.VideoCapture(url)
        
        if not self.capture.isOpened():
            self.logger.error(f"Failed to open RTSP stream: {url}")
            return False
        
        # Set buffer size to reduce latency
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution[0])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution[1])
        self.capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        
        self.logger.info(f"RTSP camera initialized: {url}")
        return True
    
    def _init_http(self) -> bool:
        """Initialize HTTP camera (IP camera)"""
        url = self._build_camera_url()
        self.capture = cv2.VideoCapture(url)
        
        if not self.capture.isOpened():
            self.logger.error(f"Failed to open HTTP stream: {url}")
            return False
        
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution[0])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution[1])
        
        self.logger.info(f"HTTP camera initialized: {url}")
        return True
    
    def _init_mjpeg(self) -> bool:
        """Initialize MJPEG stream"""
        url = self._build_camera_url()
        self.capture = cv2.VideoCapture(url)
        
        if not self.capture.isOpened():
            self.logger.error(f"Failed to open MJPEG stream: {url}")
            return False
        
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.logger.info(f"MJPEG camera initialized: {url}")
        return True
    
    def _init_usb(self) -> bool:
        """Initialize USB camera"""
        self.capture = cv2.VideoCapture(self.config.device_id or 0)
        
        if not self.capture.isOpened():
            self.logger.error(f"Failed to open USB camera: {self.config.device_id}")
            return False
        
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution[0])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution[1])
        self.capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        
        self.logger.info(f"USB camera initialized: device {self.config.device_id}")
        return True
    
    def _init_file(self) -> bool:
        """Initialize file-based camera (for testing)"""
        if not self.config.url:
            self.logger.error("File camera requires URL")
            return False
        
        self.capture = cv2.VideoCapture(self.config.url)
        
        if not self.capture.isOpened():
            self.logger.error(f"Failed to open file: {self.config.url}")
            return False
        
        self.logger.info(f"File camera initialized: {self.config.url}")
        return True
    
    def _capture_loop(self):
        """Background thread for continuous frame capture"""
        retry_count = 0
        
        while self.is_running:
            try:
                ret, frame = self.capture.read()
                
                if not ret:
                    self.logger.warning("Failed to read frame from camera")
                    retry_count += 1
                    
                    if retry_count >= self.config.retry_attempts:
                        self.logger.error("Max retry attempts reached, stopping capture")
                        break
                    
                    time.sleep(1)
                    continue
                
                # Reset retry count on successful read
                retry_count = 0
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Add frame to queue (non-blocking)
                try:
                    self.frame_queue.put_nowait({
                        'frame': frame_rgb,
                        'timestamp': time.time(),
                        'frame_number': self.frame_count
                    })
                    self.frame_count += 1
                except:
                    # Queue full, drop oldest frame
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait({
                            'frame': frame_rgb,
                            'timestamp': time.time(),
                            'frame_number': self.frame_count
                        })
                        self.frame_count += 1
                    except:
                        pass
                
                # Control frame rate
                if self.config.fps > 0:
                    time.sleep(1.0 / self.config.fps)
                    
            except Exception as e:
                self.logger.error(f"Error in capture loop: {e}")
                time.sleep(1)
        
        self.logger.info("Capture loop stopped")
    
    def start(self) -> bool:
        """Start camera capture"""
        if self.is_running:
            self.logger.warning("Camera already running")
            return True
        
        if not self._initialize_capture():
            return False
        
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        self.logger.info("Camera capture started")
        return True
    
    def stop(self):
        """Stop camera capture"""
        self.is_running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)
        
        if self.capture:
            self.capture.release()
        
        self.logger.info("Camera capture stopped")
    
    def get_latest_frame(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Get the latest frame from the camera"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def get_frame_count(self) -> int:
        """Get total number of frames captured"""
        return self.frame_count
    
    def is_connected(self) -> bool:
        """Check if camera is connected and running"""
        return self.is_running and self.capture and self.capture.isOpened()


class CameraManager:
    """
    Manages multiple camera sources
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.cameras: Dict[str, RemoteCameraCapture] = {}
        self.active_camera: Optional[str] = None
    
    def add_camera(self, name: str, config: CameraConfig) -> bool:
        """Add a camera source"""
        camera = RemoteCameraCapture(config, self.logger)
        self.cameras[name] = camera
        self.logger.info(f"Added camera: {name} ({config.source_type})")
        return True
    
    def start_camera(self, name: str) -> bool:
        """Start a specific camera"""
        if name not in self.cameras:
            self.logger.error(f"Camera not found: {name}")
            return False
        
        success = self.cameras[name].start()
        if success:
            self.active_camera = name
            self.logger.info(f"Started camera: {name}")
        return success
    
    def stop_camera(self, name: str):
        """Stop a specific camera"""
        if name in self.cameras:
            self.cameras[name].stop()
            if self.active_camera == name:
                self.active_camera = None
            self.logger.info(f"Stopped camera: {name}")
    
    def get_frame(self, camera_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get frame from active or specified camera"""
        camera_name = camera_name or self.active_camera
        
        if not camera_name or camera_name not in self.cameras:
            return None
        
        return self.cameras[camera_name].get_latest_frame()
    
    def list_cameras(self) -> Dict[str, Dict[str, Any]]:
        """List all cameras and their status"""
        status = {}
        for name, camera in self.cameras.items():
            status[name] = {
                'running': camera.is_running,
                'connected': camera.is_connected(),
                'frame_count': camera.get_frame_count(),
                'config': {
                    'source_type': camera.config.source_type,
                    'resolution': camera.config.resolution,
                    'fps': camera.config.fps
                }
            }
        return status
    
    def stop_all(self):
        """Stop all cameras"""
        for name in list(self.cameras.keys()):
            self.stop_camera(name)


class RemoteImageProcessor:
    """
    Process images from remote sources for EdgeVLM
    """
    
    def __init__(
        self,
        pipeline,  # EdgeVLM pipeline instance
        logger: Optional[logging.Logger] = None
    ):
        self.pipeline = pipeline
        self.logger = logger or logging.getLogger(__name__)
        self.camera_manager = CameraManager(logger)
    
    def process_camera_frame(
        self,
        camera_name: str,
        task: str = "caption",
        question: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Process a frame from a camera"""
        frame_data = self.camera_manager.get_frame(camera_name)
        
        if not frame_data:
            self.logger.warning(f"No frame available from camera: {camera_name}")
            return None
        
        frame = frame_data['frame']
        timestamp = frame_data['timestamp']
        
        try:
            if task == "caption":
                result = self.pipeline.generate_caption(
                    image_input=frame,
                    max_length=128
                )
            elif task == "vqa":
                if not question:
                    question = "What is in this image?"
                result = self.pipeline.answer_question(
                    image_input=frame,
                    question=question,
                    max_length=64
                )
            else:
                self.logger.error(f"Unknown task: {task}")
                return None
            
            # Add camera metadata
            result['camera_name'] = camera_name
            result['frame_timestamp'] = timestamp
            result['frame_number'] = frame_data['frame_number']
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process frame: {e}")
            return None
    
    def add_camera(self, name: str, config: CameraConfig) -> bool:
        """Add a camera to the processor"""
        return self.camera_manager.add_camera(name, config)
    
    def start_camera(self, name: str) -> bool:
        """Start a camera"""
        return self.camera_manager.start_camera(name)
    
    def stop_camera(self, name: str):
        """Stop a camera"""
        self.camera_manager.stop_camera(name)
    
    def get_camera_status(self) -> Dict[str, Any]:
        """Get status of all cameras"""
        return self.camera_manager.list_cameras()


def create_camera_configs() -> Dict[str, CameraConfig]:
    """Create common camera configurations"""
    configs = {
        # RTSP camera (most common for IP cameras)
        "rtsp_camera": CameraConfig(
            source_type="rtsp",
            url="rtsp://192.168.1.100:554/stream1",
            username="admin",
            password="password",
            resolution=(640, 480),
            fps=15
        ),
        
        # HTTP camera (some IP cameras)
        "http_camera": CameraConfig(
            source_type="http",
            url="http://192.168.1.101:8080/video",
            resolution=(640, 480),
            fps=15
        ),
        
        # MJPEG stream
        "mjpeg_camera": CameraConfig(
            source_type="mjpeg",
            url="http://192.168.1.102:8080/mjpeg",
            resolution=(640, 480),
            fps=10
        ),
        
        # USB camera (local)
        "usb_camera": CameraConfig(
            source_type="usb",
            device_id=0,
            resolution=(640, 480),
            fps=30
        ),
        
        # File camera (for testing)
        "file_camera": CameraConfig(
            source_type="file",
            url="test_video.mp4",
            resolution=(640, 480),
            fps=30
        )
    }
    
    return configs
