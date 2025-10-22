"""
Ngrok Integration for EdgeVLM
Exposes EdgeVLM services remotely for camera connections
"""

import os
import subprocess
import time
import requests
import json
import logging
from typing import Optional, Dict, Any
from pathlib import Path


class NgrokManager:
    """
    Manages ngrok tunnels for EdgeVLM services
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.tunnels: Dict[str, Dict[str, Any]] = {}
        self.ngrok_process = None
        self.ngrok_api_url = "http://localhost:4040"
        
    def start_ngrok(self, port: int, subdomain: Optional[str] = None) -> Optional[str]:
        """
        Start ngrok tunnel for a specific port
        
        Args:
            port: Local port to expose
            subdomain: Optional custom subdomain
            
        Returns:
            Public URL if successful, None otherwise
        """
        try:
            # Build ngrok command
            cmd = ["ngrok", "http", str(port)]
            
            if subdomain:
                cmd.extend(["--subdomain", subdomain])
            
            # Start ngrok process
            self.logger.info(f"Starting ngrok tunnel for port {port}")
            self.ngrok_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for ngrok to start
            time.sleep(3)
            
            # Get tunnel URL
            tunnel_url = self.get_tunnel_url(port)
            
            if tunnel_url:
                self.tunnels[str(port)] = {
                    "url": tunnel_url,
                    "port": port,
                    "subdomain": subdomain,
                    "status": "active"
                }
                self.logger.info(f"Ngrok tunnel started: {tunnel_url}")
                return tunnel_url
            else:
                self.logger.error(f"Failed to get tunnel URL for port {port}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to start ngrok tunnel: {e}")
            return None
    
    def get_tunnel_url(self, port: int) -> Optional[str]:
        """
        Get tunnel URL from ngrok API
        
        Args:
            port: Local port
            
        Returns:
            Public URL if found
        """
        try:
            response = requests.get(f"{self.ngrok_api_url}/api/tunnels", timeout=5)
            if response.status_code == 200:
                data = response.json()
                
                for tunnel in data.get('tunnels', []):
                    if tunnel.get('config', {}).get('addr') == f"localhost:{port}":
                        return tunnel.get('public_url')
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get tunnel URL: {e}")
            return None
    
    def get_all_tunnels(self) -> Dict[str, Any]:
        """
        Get all active tunnels
        
        Returns:
            Dictionary of tunnel information
        """
        try:
            response = requests.get(f"{self.ngrok_api_url}/api/tunnels", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {}
            
        except Exception as e:
            self.logger.error(f"Failed to get tunnels: {e}")
            return {}
    
    def stop_ngrok(self):
        """Stop ngrok process"""
        if self.ngrok_process:
            self.ngrok_process.terminate()
            self.ngrok_process.wait()
            self.ngrok_process = None
            self.logger.info("Ngrok process stopped")
    
    def is_ngrok_running(self) -> bool:
        """Check if ngrok is running"""
        try:
            response = requests.get(f"{self.ngrok_api_url}/api/tunnels", timeout=2)
            return response.status_code == 200
        except:
            return False


class EdgeVLMRemoteExposer:
    """
    Exposes EdgeVLM services remotely via ngrok
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.ngrok_manager = NgrokManager(logger)
        self.services = {}
        
    def expose_main_api(self, port: int = 8000, subdomain: Optional[str] = None) -> Optional[str]:
        """Expose main EdgeVLM API"""
        url = self.ngrok_manager.start_ngrok(port, subdomain)
        if url:
            self.services['main_api'] = {
                'url': url,
                'port': port,
                'subdomain': subdomain,
                'description': 'Main EdgeVLM API (local image processing)'
            }
            self.logger.info(f"Main API exposed at: {url}")
        return url
    
    def expose_camera_registration_api(self, port: int = 8002, subdomain: Optional[str] = None) -> Optional[str]:
        """Expose camera registration API"""
        url = self.ngrok_manager.start_ngrok(port, subdomain)
        if url:
            self.services['camera_registration'] = {
                'url': url,
                'port': port,
                'subdomain': subdomain,
                'description': 'Camera Registration API (self-registration)'
            }
            self.logger.info(f"Camera Registration API exposed at: {url}")
        return url
    
    def expose_remote_camera_api(self, port: int = 8001, subdomain: Optional[str] = None) -> Optional[str]:
        """Expose remote camera API"""
        url = self.ngrok_manager.start_ngrok(port, subdomain)
        if url:
            self.services['remote_camera'] = {
                'url': url,
                'port': port,
                'subdomain': subdomain,
                'description': 'Remote Camera API (manual management)'
            }
            self.logger.info(f"Remote Camera API exposed at: {url}")
        return url
    
    def expose_all_services(self, subdomains: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Expose all EdgeVLM services
        
        Args:
            subdomains: Optional custom subdomains for each service
            
        Returns:
            Dictionary of service URLs
        """
        if subdomains is None:
            subdomains = {}
        
        urls = {}
        
        # Expose main API
        main_url = self.expose_main_api(8000, subdomains.get('main'))
        if main_url:
            urls['main_api'] = main_url
        
        # Expose camera registration API (recommended)
        reg_url = self.expose_camera_registration_api(8002, subdomains.get('registration'))
        if reg_url:
            urls['camera_registration'] = reg_url
        
        # Expose remote camera API (alternative)
        remote_url = self.expose_remote_camera_api(8001, subdomains.get('remote'))
        if remote_url:
            urls['remote_camera'] = remote_url
        
        return urls
    
    def get_service_urls(self) -> Dict[str, str]:
        """Get all exposed service URLs"""
        return {name: info['url'] for name, info in self.services.items()}
    
    def generate_camera_config(self, service_type: str = 'registration') -> Dict[str, Any]:
        """
        Generate camera configuration for remote connection
        
        Args:
            service_type: 'registration' or 'remote'
            
        Returns:
            Camera configuration
        """
        if service_type not in self.services:
            raise ValueError(f"Service {service_type} not exposed")
        
        service_info = self.services[service_type]
        base_url = service_info['url']
        
        if service_type == 'registration':
            return {
                'api_base_url': base_url,
                'registration_endpoint': f"{base_url}/register",
                'heartbeat_endpoint': f"{base_url}/cameras/{{camera_id}}/heartbeat",
                'upload_endpoint': f"{base_url}/cameras/{{camera_id}}/upload",
                'status_endpoint': f"{base_url}/cameras/{{camera_id}}/status"
            }
        elif service_type == 'remote':
            return {
                'api_base_url': base_url,
                'cameras_endpoint': f"{base_url}/cameras",
                'process_endpoint': f"{base_url}/process",
                'health_endpoint': f"{base_url}/health"
            }
    
    def save_config(self, filename: str = "remote_config.json"):
        """Save remote configuration to file"""
        config = {
            'services': self.services,
            'urls': self.get_service_urls(),
            'camera_configs': {
                'registration': self.generate_camera_config('registration'),
                'remote': self.generate_camera_config('remote')
            },
            'timestamp': time.time(),
            'ngrok_status': self.ngrok_manager.is_ngrok_running()
        }
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Configuration saved to {filename}")
    
    def stop_all(self):
        """Stop all ngrok tunnels"""
        self.ngrok_manager.stop_ngrok()
        self.services = {}


def main():
    """Main function to expose EdgeVLM services"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Expose EdgeVLM services via ngrok")
    parser.add_argument("--main-port", type=int, default=8000, help="Main API port")
    parser.add_argument("--registration-port", type=int, default=8002, help="Registration API port")
    parser.add_argument("--remote-port", type=int, default=8001, help="Remote camera API port")
    parser.add_argument("--main-subdomain", type=str, help="Custom subdomain for main API")
    parser.add_argument("--registration-subdomain", type=str, help="Custom subdomain for registration API")
    parser.add_argument("--remote-subdomain", type=str, help="Custom subdomain for remote API")
    parser.add_argument("--config-file", type=str, default="remote_config.json", help="Config file to save")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("EdgeVLM-Remote")
    
    # Create exposer
    exposer = EdgeVLMRemoteExposer(logger)
    
    try:
        # Expose all services
        subdomains = {
            'main': args.main_subdomain,
            'registration': args.registration_subdomain,
            'remote': args.remote_subdomain
        }
        
        urls = exposer.expose_all_services(subdomains)
        
        print("\n" + "="*60)
        print("EdgeVLM Services Exposed Remotely")
        print("="*60)
        
        for service_name, url in urls.items():
            print(f"{service_name.replace('_', ' ').title()}: {url}")
        
        print("\nCamera Registration (Recommended):")
        print(f"  Registration: {urls.get('camera_registration', 'Not exposed')}")
        
        print("\nManual Camera Management (Alternative):")
        print(f"  Remote API: {urls.get('remote_camera', 'Not exposed')}")
        
        print("\nLocal Image Processing:")
        print(f"  Main API: {urls.get('main_api', 'Not exposed')}")
        
        # Save configuration
        exposer.save_config(args.config_file)
        
        print(f"\nConfiguration saved to: {args.config_file}")
        print("\nPress Ctrl+C to stop all tunnels...")
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping ngrok tunnels...")
        exposer.stop_all()
        print("All tunnels stopped")


if __name__ == "__main__":
    main()
