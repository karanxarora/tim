#!/usr/bin/env python3
"""
Test script for ARM32 setup without PyTorch
"""

import sys
import os
import platform
from pathlib import Path

def test_architecture():
    """Test system architecture"""
    print("=" * 60)
    print("ARM32 Setup Test")
    print("=" * 60)
    
    arch = platform.machine()
    print(f"Architecture: {arch}")
    
    if arch == "armv7l":
        print("✅ ARM32 detected - PyTorch not available")
        return True
    elif arch == "aarch64":
        print("✅ ARM64 detected - PyTorch available")
        return True
    else:
        print(f"ℹ️  {arch} detected - PyTorch may be available")
        return True

def test_imports():
    """Test available imports"""
    print("\nTesting imports...")
    
    # Test core packages
    try:
        import numpy
        print("✅ numpy imported")
    except ImportError as e:
        print(f"❌ numpy failed: {e}")
        return False
    
    try:
        import cv2
        print("✅ opencv-python imported")
    except ImportError as e:
        print(f"❌ opencv-python failed: {e}")
        return False
    
    try:
        import llama_cpp
        print("✅ llama-cpp-python imported")
    except ImportError as e:
        print(f"❌ llama-cpp-python failed: {e}")
        return False
    
    # Test PyTorch (should fail on ARM32)
    try:
        import torch
        print("✅ PyTorch imported (ARM64 system)")
    except ImportError:
        print("ℹ️  PyTorch not available (expected on ARM32)")
    
    # Test alternative packages
    try:
        import scipy
        print("✅ scipy imported")
    except ImportError as e:
        print(f"❌ scipy failed: {e}")
    
    try:
        import sklearn
        print("✅ scikit-learn imported")
    except ImportError as e:
        print(f"❌ scikit-learn failed: {e}")
    
    return True

def test_edgevlm_modules():
    """Test EdgeVLM modules"""
    print("\nTesting EdgeVLM modules...")
    
    try:
        from api import app
        print("✅ API module imported")
    except ImportError as e:
        print(f"❌ API module failed: {e}")
        return False
    
    try:
        from pipeline import EdgeVLMPipeline
        print("✅ Pipeline module imported")
    except ImportError as e:
        print(f"❌ Pipeline module failed: {e}")
        return False
    
    try:
        from ngrok_integration import EdgeVLMRemoteExposer
        print("✅ Ngrok integration imported")
    except ImportError as e:
        print(f"❌ Ngrok integration failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration"""
    print("\nTesting configuration...")
    
    try:
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print("✅ Config loaded")
        print(f"   - Target device: {config.get('hardware', {}).get('target_device', 'unknown')}")
        print(f"   - Max RAM: {config.get('hardware', {}).get('max_ram_mb', 'unknown')}MB")
        print(f"   - Inference engine: {config.get('inference', {}).get('engine', 'unknown')}")
        
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing ARM32 setup...")
    
    tests = [
        test_architecture,
        test_imports,
        test_edgevlm_modules,
        test_config
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ ARM32 setup is working!")
        print("\nNext steps:")
        print("1. Download models: python download_models_arm32.py")
        print("2. Test the system: python main.py --log-level DEBUG")
        print("3. Start service: sudo systemctl start edgevlm")
    else:
        print("❌ Some tests failed")
        print("\nTroubleshooting:")
        print("1. Check if all packages are installed")
        print("2. Verify virtual environment is activated")
        print("3. Check system architecture")
    
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
