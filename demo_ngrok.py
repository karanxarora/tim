#!/usr/bin/env python3
"""
Ngrok Demo Script
Demonstrates EdgeVLM remote access with ngrok
"""

import requests
import time
import json
import subprocess
import sys
from pathlib import Path


def check_ngrok_installed():
    """Check if ngrok is installed"""
    try:
        result = subprocess.run(['ngrok', 'version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Ngrok installed: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ Ngrok not installed!")
    print("Install with:")
    print("  curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null")
    print("  echo 'deb https://ngrok-agent.s3.amazonaws.com buster main' | sudo tee /etc/apt/sources.list.d/ngrok.list")
    print("  sudo apt update && sudo apt install ngrok")
    return False


def check_ngrok_auth():
    """Check if ngrok is authenticated"""
    try:
        result = subprocess.run(['ngrok', 'config', 'check'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ngrok authenticated")
            return True
    except:
        pass
    
    print("⚠️  Ngrok authentication required")
    print("Run: ngrok config add-authtoken YOUR_TOKEN")
    print("Get token from: https://dashboard.ngrok.com/get-started/your-authtoken")
    return False


def start_services():
    """Start EdgeVLM services"""
    print("\n🚀 Starting EdgeVLM services...")
    
    # Start main API
    print("Starting main API (port 8000)...")
    main_process = subprocess.Popen([
        sys.executable, 'main.py', '--port', '8000'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Start camera registration API
    print("Starting camera registration API (port 8002)...")
    reg_process = subprocess.Popen([
        sys.executable, 'api_camera_registration.py', '--port', '8002'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for services to start
    print("Waiting for services to start...")
    time.sleep(5)
    
    # Check if services are running
    try:
        main_health = requests.get('http://localhost:8000/health', timeout=5)
        reg_health = requests.get('http://localhost:8002/health', timeout=5)
        
        if main_health.status_code == 200 and reg_health.status_code == 200:
            print("✅ All services started successfully")
            return main_process, reg_process
        else:
            print("❌ Services failed to start")
            return None, None
    except:
        print("❌ Services failed to start")
        return None, None


def start_ngrok_tunnels():
    """Start ngrok tunnels"""
    print("\n🌐 Starting ngrok tunnels...")
    
    # Start ngrok for camera registration API
    print("Starting ngrok tunnel for camera registration API...")
    ngrok_reg = subprocess.Popen([
        'ngrok', 'http', '8002', '--log=stdout'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Start ngrok for main API
    print("Starting ngrok tunnel for main API...")
    ngrok_main = subprocess.Popen([
        'ngrok', 'http', '8000', '--log=stdout'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for ngrok to start
    print("Waiting for ngrok tunnels to establish...")
    time.sleep(10)
    
    return ngrok_reg, ngrok_main


def get_tunnel_urls():
    """Get tunnel URLs from ngrok API"""
    print("\n🔗 Getting tunnel URLs...")
    
    try:
        response = requests.get('http://localhost:4040/api/tunnels', timeout=5)
        if response.status_code == 200:
            data = response.json()
            
            urls = {}
            for tunnel in data.get('tunnels', []):
                addr = tunnel.get('config', {}).get('addr')
                public_url = tunnel.get('public_url')
                
                if addr == 'localhost:8002':
                    urls['registration'] = public_url
                elif addr == 'localhost:8000':
                    urls['main'] = public_url
            
            return urls
        else:
            print("❌ Failed to get tunnel URLs")
            return None
    except Exception as e:
        print(f"❌ Error getting tunnel URLs: {e}")
        return None


def test_remote_connection(registration_url):
    """Test remote connection"""
    print(f"\n🧪 Testing remote connection...")
    print(f"Registration URL: {registration_url}")
    
    try:
        # Test health endpoint
        health_response = requests.get(f"{registration_url}/health", timeout=10)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"✅ Health check passed: {health_data['status']}")
            print(f"   Registered cameras: {health_data['registered_cameras']}")
            print(f"   Active cameras: {health_data['active_cameras']}")
        else:
            print(f"❌ Health check failed: {health_response.status_code}")
            return False
        
        # Test registration endpoint
        test_registration = {
            "name": "Test Camera",
            "location": "Test Location",
            "source_type": "usb",
            "device_id": 0,
            "resolution": [640, 480],
            "fps": 15,
            "capabilities": ["caption", "vqa"],
            "metadata": {
                "test": True,
                "demo": True
            }
        }
        
        reg_response = requests.post(f"{registration_url}/register", json=test_registration, timeout=10)
        if reg_response.status_code == 200:
            reg_data = reg_response.json()
            print(f"✅ Camera registration successful")
            print(f"   Camera ID: {reg_data['camera_id']}")
            print(f"   API Key: {reg_data['api_key'][:8]}...")
            return True
        else:
            print(f"❌ Camera registration failed: {reg_response.status_code}")
            print(f"   Error: {reg_response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False


def generate_camera_client_command(registration_url):
    """Generate camera client command"""
    print(f"\n📝 Camera Client Command:")
    print("=" * 60)
    
    command = f"""python examples/remote_camera_client.py \\
  --ngrok-url "{registration_url}" \\
  --name "Remote Camera" \\
  --location "Remote Location" \\
  --type usb \\
  --device 0"""
    
    print(command)
    print("=" * 60)


def save_config(urls):
    """Save configuration"""
    config = {
        "services": {
            "camera_registration": {
                "url": urls.get('registration'),
                "port": 8002,
                "description": "Camera Registration API"
            },
            "main_api": {
                "url": urls.get('main'),
                "port": 8000,
                "description": "Main EdgeVLM API"
            }
        },
        "timestamp": time.time(),
        "ngrok_status": True
    }
    
    with open('remote_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n💾 Configuration saved to: remote_config.json")


def main():
    """Main demo function"""
    print("=" * 60)
    print("EdgeVLM Ngrok Remote Access Demo")
    print("=" * 60)
    
    # Check prerequisites
    if not check_ngrok_installed():
        return
    
    if not check_ngrok_auth():
        return
    
    # Start services
    main_process, reg_process = start_services()
    if not main_process or not reg_process:
        return
    
    # Start ngrok tunnels
    ngrok_reg, ngrok_main = start_ngrok_tunnels()
    
    # Get tunnel URLs
    urls = get_tunnel_urls()
    if not urls:
        print("❌ Failed to get tunnel URLs")
        return
    
    # Display results
    print("\n" + "=" * 60)
    print("EdgeVLM Services Exposed Remotely!")
    print("=" * 60)
    
    if 'registration' in urls:
        print(f"📹 Camera Registration API: {urls['registration']}")
    
    if 'main' in urls:
        print(f"🖼️  Main API: {urls['main']}")
    
    # Test connection
    if 'registration' in urls:
        if test_remote_connection(urls['registration']):
            generate_camera_client_command(urls['registration'])
    
    # Save configuration
    save_config(urls)
    
    print(f"\n🎉 Demo Complete!")
    print(f"\nNext steps:")
    print(f"1. Test camera registration: curl {urls.get('registration', 'N/A')}/health")
    print(f"2. Run camera client: python examples/remote_camera_client.py --ngrok-url \"{urls.get('registration', 'N/A')}\" --name \"Test Camera\" --location \"Test\" --type usb --device 0")
    print(f"3. View ngrok dashboard: http://localhost:4040")
    print(f"\nPress Ctrl+C to stop all services...")
    
    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping all services...")
        
        # Stop processes
        for process in [main_process, reg_process, ngrok_reg, ngrok_main]:
            if process:
                process.terminate()
                process.wait()
        
        print("✅ All services stopped")


if __name__ == "__main__":
    main()
