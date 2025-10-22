"""
Remote Camera Integration Demo
Demonstrates how to use EdgeVLM with remote cameras
"""

import requests
import time
import json
from typing import Dict, Any


class RemoteCameraClient:
    """Client for interacting with EdgeVLM Remote Camera API"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
    
    def add_rtsp_camera(self, name: str, url: str, username: str = None, password: str = None) -> bool:
        """Add an RTSP camera"""
        config = {
            "name": name,
            "source_type": "rtsp",
            "url": url,
            "username": username,
            "password": password,
            "resolution": [640, 480],
            "fps": 15
        }
        
        response = requests.post(f"{self.base_url}/cameras", json=config)
        return response.status_code == 200
    
    def add_http_camera(self, name: str, url: str) -> bool:
        """Add an HTTP camera"""
        config = {
            "name": name,
            "source_type": "http",
            "url": url,
            "resolution": [640, 480],
            "fps": 15
        }
        
        response = requests.post(f"{self.base_url}/cameras", json=config)
        return response.status_code == 200
    
    def add_usb_camera(self, name: str, device_id: int = 0) -> bool:
        """Add a USB camera"""
        config = {
            "name": name,
            "source_type": "usb",
            "device_id": device_id,
            "resolution": [640, 480],
            "fps": 30
        }
        
        response = requests.post(f"{self.base_url}/cameras", json=config)
        return response.status_code == 200
    
    def start_camera(self, name: str) -> bool:
        """Start a camera"""
        response = requests.post(f"{self.base_url}/cameras/{name}/start")
        return response.status_code == 200
    
    def stop_camera(self, name: str) -> bool:
        """Stop a camera"""
        response = requests.post(f"{self.base_url}/cameras/{name}/stop")
        return response.status_code == 200
    
    def list_cameras(self) -> Dict[str, Any]:
        """List all cameras"""
        response = requests.get(f"{self.base_url}/cameras")
        return response.json()
    
    def process_frame(self, camera_name: str, task: str = "caption", question: str = None) -> Dict[str, Any]:
        """Process a frame from a camera"""
        data = {
            "camera_name": camera_name,
            "task": task,
            "question": question
        }
        
        response = requests.post(f"{self.base_url}/process", json=data)
        return response.json()
    
    def test_camera(self, camera_name: str) -> Dict[str, Any]:
        """Test camera connection"""
        response = requests.post(f"{self.base_url}/cameras/{camera_name}/test")
        return response.json()
    
    def get_health(self) -> Dict[str, Any]:
        """Get system health"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()


def demo_rtsp_camera():
    """Demo with RTSP camera"""
    print("=" * 60)
    print("RTSP Camera Demo")
    print("=" * 60)
    
    client = RemoteCameraClient()
    
    # Add RTSP camera
    print("Adding RTSP camera...")
    success = client.add_rtsp_camera(
        name="ip_camera",
        url="rtsp://192.168.1.100:554/stream1",
        username="admin",
        password="password"
    )
    
    if not success:
        print("❌ Failed to add RTSP camera")
        return
    
    print("✓ RTSP camera added")
    
    # Start camera
    print("Starting camera...")
    success = client.start_camera("ip_camera")
    
    if not success:
        print("❌ Failed to start camera")
        return
    
    print("✓ Camera started")
    
    # Test camera
    print("Testing camera connection...")
    test_result = client.test_camera("ip_camera")
    print(f"Test result: {test_result}")
    
    # Process frames
    print("\nProcessing frames...")
    for i in range(5):
        print(f"\nFrame {i+1}:")
        
        # Caption
        result = client.process_frame("ip_camera", "caption")
        if "result" in result:
            caption = result["result"]["caption"]
            latency = result["processing_time"]
            print(f"  Caption: {caption}")
            print(f"  Latency: {latency:.2f}s")
        else:
            print(f"  Error: {result}")
        
        time.sleep(2)
    
    # Stop camera
    print("\nStopping camera...")
    client.stop_camera("ip_camera")
    print("✓ Camera stopped")


def demo_http_camera():
    """Demo with HTTP camera"""
    print("=" * 60)
    print("HTTP Camera Demo")
    print("=" * 60)
    
    client = RemoteCameraClient()
    
    # Add HTTP camera
    print("Adding HTTP camera...")
    success = client.add_http_camera(
        name="web_camera",
        url="http://192.168.1.101:8080/video"
    )
    
    if not success:
        print("❌ Failed to add HTTP camera")
        return
    
    print("✓ HTTP camera added")
    
    # Start camera
    print("Starting camera...")
    success = client.start_camera("web_camera")
    
    if not success:
        print("❌ Failed to start camera")
        return
    
    print("✓ Camera started")
    
    # VQA demo
    print("\nVQA Demo:")
    questions = [
        "What is in this image?",
        "What colors do you see?",
        "Is this indoors or outdoors?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        result = client.process_frame("web_camera", "vqa", question)
        
        if "result" in result:
            answer = result["result"]["answer"]
            latency = result["processing_time"]
            print(f"A: {answer}")
            print(f"  Latency: {latency:.2f}s")
        else:
            print(f"  Error: {result}")
        
        time.sleep(3)
    
    # Stop camera
    print("\nStopping camera...")
    client.stop_camera("web_camera")
    print("✓ Camera stopped")


def demo_usb_camera():
    """Demo with USB camera"""
    print("=" * 60)
    print("USB Camera Demo")
    print("=" * 60)
    
    client = RemoteCameraClient()
    
    # Add USB camera
    print("Adding USB camera...")
    success = client.add_usb_camera(
        name="usb_cam",
        device_id=0
    )
    
    if not success:
        print("❌ Failed to add USB camera")
        return
    
    print("✓ USB camera added")
    
    # Start camera
    print("Starting camera...")
    success = client.start_camera("usb_cam")
    
    if not success:
        print("❌ Failed to start camera")
        return
    
    print("✓ Camera started")
    
    # Continuous processing
    print("\nContinuous processing (press Ctrl+C to stop)...")
    try:
        frame_count = 0
        while True:
            result = client.process_frame("usb_cam", "caption")
            
            if "result" in result:
                frame_count += 1
                caption = result["result"]["caption"]
                latency = result["processing_time"]
                print(f"Frame {frame_count}: {caption} ({latency:.2f}s)")
            else:
                print(f"Error: {result}")
            
            time.sleep(5)  # Process every 5 seconds
            
    except KeyboardInterrupt:
        print("\nStopping...")
    
    # Stop camera
    print("Stopping camera...")
    client.stop_camera("usb_cam")
    print("✓ Camera stopped")


def demo_multiple_cameras():
    """Demo with multiple cameras"""
    print("=" * 60)
    print("Multiple Cameras Demo")
    print("=" * 60)
    
    client = RemoteCameraClient()
    
    # Add multiple cameras
    cameras = [
        {"name": "camera1", "type": "rtsp", "url": "rtsp://192.168.1.100:554/stream1"},
        {"name": "camera2", "type": "http", "url": "http://192.168.1.101:8080/video"},
        {"name": "camera3", "type": "usb", "device_id": 0}
    ]
    
    print("Adding cameras...")
    for cam in cameras:
        if cam["type"] == "rtsp":
            success = client.add_rtsp_camera(cam["name"], cam["url"])
        elif cam["type"] == "http":
            success = client.add_http_camera(cam["name"], cam["url"])
        elif cam["type"] == "usb":
            success = client.add_usb_camera(cam["name"], cam["device_id"])
        
        if success:
            print(f"✓ Added {cam['name']}")
        else:
            print(f"❌ Failed to add {cam['name']}")
    
    # List cameras
    print("\nCamera status:")
    cameras = client.list_cameras()
    for name, status in cameras.items():
        print(f"  {name}: {'Running' if status['running'] else 'Stopped'}")
    
    # Start all cameras
    print("\nStarting cameras...")
    for name in cameras.keys():
        success = client.start_camera(name)
        if success:
            print(f"✓ Started {name}")
        else:
            print(f"❌ Failed to start {name}")
    
    # Process from different cameras
    print("\nProcessing from different cameras...")
    for i in range(3):
        for name in cameras.keys():
            print(f"\nProcessing from {name}:")
            result = client.process_frame(name, "caption")
            
            if "result" in result:
                caption = result["result"]["caption"]
                print(f"  {caption}")
            else:
                print(f"  Error: {result}")
        
        time.sleep(2)
    
    # Stop all cameras
    print("\nStopping all cameras...")
    for name in cameras.keys():
        client.stop_camera(name)
        print(f"✓ Stopped {name}")


def demo_health_monitoring():
    """Demo health monitoring"""
    print("=" * 60)
    print("Health Monitoring Demo")
    print("=" * 60)
    
    client = RemoteCameraClient()
    
    # Get health status
    health = client.get_health()
    
    print("System Health:")
    print(f"  Status: {health['status']}")
    print(f"  Pipeline loaded: {health['pipeline_loaded']}")
    print(f"  Cameras configured: {health['cameras_configured']}")
    print(f"  Active cameras: {health['active_cameras']}")
    
    print("\nSystem Info:")
    sys_info = health['system_info']
    print(f"  CPU: {sys_info['cpu_percent']:.1f}%")
    print(f"  Memory: {sys_info['memory_percent']:.1f}%")
    if sys_info['temperature_celsius']:
        print(f"  Temperature: {sys_info['temperature_celsius']:.1f}°C")
    
    print("\nCamera Status:")
    for name, status in health['camera_status'].items():
        print(f"  {name}:")
        print(f"    Running: {status['running']}")
        print(f"    Connected: {status['connected']}")
        print(f"    Frame count: {status['frame_count']}")


def main():
    """Run all demos"""
    print("EdgeVLM Remote Camera Integration Demo")
    print("=" * 60)
    print("Make sure the remote camera API is running:")
    print("  python api_remote.py --port 8001")
    print("=" * 60)
    
    try:
        # Test connection
        client = RemoteCameraClient()
        health = client.get_health()
        print(f"✓ Connected to API: {health['status']}")
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("Please start the remote camera API first:")
        print("  python api_remote.py --port 8001")
        return
    
    # Run demos
    demos = [
        ("Health Monitoring", demo_health_monitoring),
        ("RTSP Camera", demo_rtsp_camera),
        ("HTTP Camera", demo_http_camera),
        ("USB Camera", demo_usb_camera),
        ("Multiple Cameras", demo_multiple_cameras)
    ]
    
    for name, demo_func in demos:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")
        
        try:
            demo_func()
        except KeyboardInterrupt:
            print(f"\n{name} interrupted by user")
            break
        except Exception as e:
            print(f"❌ {name} failed: {e}")
        
        print(f"\n{name} completed")
        input("Press Enter to continue to next demo...")
    
    print("\nAll demos completed!")


if __name__ == "__main__":
    main()
