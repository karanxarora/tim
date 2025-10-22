"""
Camera Registration Demo
Demonstrates the camera registration workflow
"""

import requests
import time
import json
from typing import Dict, Any


class EdgeVLMAdminClient:
    """Admin client for managing registered cameras"""
    
    def __init__(self, api_base_url: str = "http://localhost:8002"):
        self.api_base_url = api_base_url
    
    def get_health(self) -> Dict[str, Any]:
        """Get system health"""
        response = requests.get(f"{self.api_base_url}/health")
        return response.json()
    
    def list_cameras(self) -> Dict[str, Any]:
        """List all registered cameras"""
        response = requests.get(f"{self.api_base_url}/cameras")
        return response.json()
    
    def get_camera_status(self, camera_id: str) -> Dict[str, Any]:
        """Get specific camera status"""
        response = requests.get(f"{self.api_base_url}/cameras/{camera_id}/status")
        return response.json()
    
    def start_camera(self, camera_id: str) -> Dict[str, Any]:
        """Start camera processing"""
        response = requests.post(f"{self.api_base_url}/cameras/{camera_id}/start")
        return response.json()
    
    def stop_camera(self, camera_id: str) -> Dict[str, Any]:
        """Stop camera processing"""
        response = requests.post(f"{self.api_base_url}/cameras/{camera_id}/stop")
        return response.json()


def demo_camera_registration():
    """Demo camera registration process"""
    print("=" * 60)
    print("Camera Registration Demo")
    print("=" * 60)
    
    admin = EdgeVLMAdminClient()
    
    # Check system health
    print("1. Checking system health...")
    health = admin.get_health()
    print(f"   Status: {health['status']}")
    print(f"   Registered cameras: {health['registered_cameras']}")
    print(f"   Active cameras: {health['active_cameras']}")
    print()
    
    # Simulate camera registration
    print("2. Simulating camera registration...")
    
    # Camera 1: RTSP Camera
    camera1_data = {
        "name": "Front Door Camera",
        "location": "Main Entrance",
        "source_type": "rtsp",
        "url": "rtsp://192.168.1.100:554/stream1",
        "username": "admin",
        "password": "password",
        "resolution": [640, 480],
        "fps": 15,
        "capabilities": ["caption", "vqa"],
        "metadata": {
            "model": "Hikvision DS-2CD2143G0-I",
            "firmware": "V5.6.0",
            "client_version": "1.0.0"
        }
    }
    
    response = requests.post(f"{admin.api_base_url}/register", json=camera1_data)
    if response.status_code == 200:
        camera1 = response.json()
        print(f"   ✓ Camera 1 registered: {camera1['camera_id']}")
        print(f"   API Key: {camera1['api_key']}")
        print(f"   Endpoints: {list(camera1['endpoints'].keys())}")
    else:
        print(f"   ✗ Camera 1 registration failed: {response.status_code}")
        return
    
    print()
    
    # Camera 2: USB Camera
    camera2_data = {
        "name": "Office Camera",
        "location": "Office Room",
        "source_type": "usb",
        "device_id": 0,
        "resolution": [640, 480],
        "fps": 30,
        "capabilities": ["caption"],
        "metadata": {
            "model": "Logitech C920",
            "client_version": "1.0.0"
        }
    }
    
    response = requests.post(f"{admin.api_base_url}/register", json=camera2_data)
    if response.status_code == 200:
        camera2 = response.json()
        print(f"   ✓ Camera 2 registered: {camera2['camera_id']}")
        print(f"   API Key: {camera2['api_key']}")
    else:
        print(f"   ✗ Camera 2 registration failed: {response.status_code}")
        return
    
    print()
    
    # List all cameras
    print("3. Listing all registered cameras...")
    cameras = admin.list_cameras()
    
    for camera_id, camera_info in cameras.items():
        print(f"   Camera: {camera_info['name']}")
        print(f"     ID: {camera_id}")
        print(f"     Location: {camera_info['location']}")
        print(f"     Status: {camera_info['status']}")
        print(f"     Registered: {camera_info['registered_at']}")
        print(f"     Last Heartbeat: {camera_info['last_heartbeat']}")
        print(f"     Frame Count: {camera_info['frame_count']}")
        print(f"     Uptime: {camera_info['uptime_seconds']:.1f}s")
        print()
    
    # Simulate camera heartbeats
    print("4. Simulating camera heartbeats...")
    
    for i in range(3):
        print(f"   Heartbeat {i+1}:")
        
        # Camera 1 heartbeat
        heartbeat1 = {
            "camera_id": camera1['camera_id'],
            "api_key": camera1['api_key'],
            "status": "active",
            "frame_count": i * 10
        }
        
        response = requests.post(
            f"{admin.api_base_url}/cameras/{camera1['camera_id']}/heartbeat",
            json=heartbeat1
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"     Camera 1: {result['status']} - {len(result['commands'])} commands")
        else:
            print(f"     Camera 1: Failed ({response.status_code})")
        
        # Camera 2 heartbeat
        heartbeat2 = {
            "camera_id": camera2['camera_id'],
            "api_key": camera2['api_key'],
            "status": "active",
            "frame_count": i * 15
        }
        
        response = requests.post(
            f"{admin.api_base_url}/cameras/{camera2['camera_id']}/heartbeat",
            json=heartbeat2
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"     Camera 2: {result['status']} - {len(result['commands'])} commands")
        else:
            print(f"     Camera 2: Failed ({response.status_code})")
        
        time.sleep(2)
    
    print()
    
    # Simulate frame upload
    print("5. Simulating frame upload...")
    
    # Create a test image (in real scenario, this would be from camera)
    import numpy as np
    from PIL import Image
    import base64
    import io
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    pil_image = Image.fromarray(test_image)
    
    # Convert to base64
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG')
    image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Upload frame
    frame_data = {
        "camera_id": camera1['camera_id'],
        "api_key": camera1['api_key'],
        "frame_data": image_data,
        "timestamp": time.time(),
        "frame_number": 1,
        "task": "caption"
    }
    
    response = requests.post(
        f"{admin.api_base_url}/cameras/{camera1['camera_id']}/upload",
        json=frame_data
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Frame processed successfully")
        print(f"   Task: {result['task']}")
        print(f"   Processing time: {result['processing_time']:.2f}s")
        print(f"   Result: {result['result']}")
    else:
        print(f"   ✗ Frame upload failed: {response.status_code}")
        print(f"   Error: {response.text}")
    
    print()
    
    # Final status
    print("6. Final camera status...")
    cameras = admin.list_cameras()
    
    for camera_id, camera_info in cameras.items():
        print(f"   {camera_info['name']}: {camera_info['status']} "
              f"({camera_info['frame_count']} frames, "
              f"{camera_info['uptime_seconds']:.1f}s uptime)")


def demo_camera_management():
    """Demo camera management operations"""
    print("=" * 60)
    print("Camera Management Demo")
    print("=" * 60)
    
    admin = EdgeVLMAdminClient()
    
    # List cameras
    print("1. Current cameras:")
    cameras = admin.list_cameras()
    
    if not cameras:
        print("   No cameras registered")
        return
    
    for camera_id, camera_info in cameras.items():
        print(f"   {camera_info['name']} ({camera_id}): {camera_info['status']}")
    
    print()
    
    # Start/stop cameras
    for camera_id in cameras.keys():
        print(f"2. Managing camera {camera_id}...")
        
        # Start camera
        result = admin.start_camera(camera_id)
        print(f"   Start: {result['status']} - {result['message']}")
        
        time.sleep(2)
        
        # Stop camera
        result = admin.stop_camera(camera_id)
        print(f"   Stop: {result['status']} - {result['message']}")
        
        print()


def demo_error_handling():
    """Demo error handling scenarios"""
    print("=" * 60)
    print("Error Handling Demo")
    print("=" * 60)
    
    admin = EdgeVLMAdminClient()
    
    # Test invalid camera ID
    print("1. Testing invalid camera ID...")
    try:
        response = requests.get(f"{admin.api_base_url}/cameras/invalid_id/status")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print()
    
    # Test invalid API key
    print("2. Testing invalid API key...")
    try:
        heartbeat_data = {
            "camera_id": "test_id",
            "api_key": "invalid_key",
            "status": "active"
        }
        response = requests.post(f"{admin.api_base_url}/cameras/test_id/heartbeat", json=heartbeat_data)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print()
    
    # Test malformed request
    print("3. Testing malformed request...")
    try:
        response = requests.post(f"{admin.api_base_url}/register", json={"invalid": "data"})
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")


def main():
    """Run all demos"""
    print("EdgeVLM Camera Registration Demo")
    print("=" * 60)
    print("Make sure the camera registration API is running:")
    print("  python api_camera_registration.py --port 8002")
    print("=" * 60)
    
    try:
        # Test connection
        admin = EdgeVLMAdminClient()
        health = admin.get_health()
        print(f"✓ Connected to API: {health['status']}")
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("Please start the camera registration API first:")
        print("  python api_camera_registration.py --port 8002")
        return
    
    # Run demos
    demos = [
        ("Camera Registration", demo_camera_registration),
        ("Camera Management", demo_camera_management),
        ("Error Handling", demo_error_handling)
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
