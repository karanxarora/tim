"""
Test Script for EdgeVLM API
Quick validation of API functionality
"""

import requests
import sys
import time
from pathlib import Path


def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Health check passed")
            print(f"  Status: {data['status']}")
            print(f"  Pipeline loaded: {data['pipeline_loaded']}")
            print(f"  CPU: {data['system_info']['cpu_percent']:.1f}%")
            print(f"  Memory: {data['system_info']['memory_percent']:.1f}%")
            if data['system_info']['temperature_celsius']:
                print(f"  Temperature: {data['system_info']['temperature_celsius']:.1f}°C")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False


def test_metrics():
    """Test metrics endpoint"""
    print("\nTesting /metrics endpoint...")
    try:
        response = requests.get("http://localhost:8000/metrics", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Metrics retrieved")
            print(f"  Total inferences: {data['inference_metrics']['total_inferences']}")
            if data['inference_metrics']['total_inferences'] > 0:
                print(f"  Avg latency: {data['inference_metrics']['avg_latency']:.3f}s")
                print(f"  Early exit rate: {data['inference_metrics']['early_exit_rate']:.1%}")
            return True
        else:
            print(f"✗ Metrics failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Metrics error: {e}")
        return False


def test_caption(image_path):
    """Test caption endpoint"""
    print(f"\nTesting /caption endpoint with {image_path}...")
    
    if not Path(image_path).exists():
        print(f"✗ Image not found: {image_path}")
        print("  Please provide a valid image path")
        return False
    
    try:
        start = time.time()
        with open(image_path, 'rb') as f:
            response = requests.post(
                "http://localhost:8000/caption",
                files={'image': f},
                data={'max_length': 128},
                timeout=30
            )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Caption generated in {elapsed:.2f}s")
            print(f"  Caption: {data['caption']}")
            print(f"  Latency: {data['latency']:.3f}s")
            print(f"  Tokens/sec: {data['tokens_per_second']:.1f}")
            print(f"  Early exit: {data['early_exit']}")
            if data['exit_layer']:
                print(f"  Exit layer: {data['exit_layer']}")
            return True
        else:
            print(f"✗ Caption failed: {response.status_code}")
            print(f"  Error: {response.json().get('detail', 'Unknown')}")
            return False
    except Exception as e:
        print(f"✗ Caption error: {e}")
        return False


def test_vqa(image_path, question="What is in this image?"):
    """Test VQA endpoint"""
    print(f"\nTesting /vqa endpoint...")
    print(f"  Question: {question}")
    
    if not Path(image_path).exists():
        print(f"✗ Image not found: {image_path}")
        return False
    
    try:
        start = time.time()
        with open(image_path, 'rb') as f:
            response = requests.post(
                "http://localhost:8000/vqa",
                files={'image': f},
                data={
                    'question': question,
                    'max_length': 64
                },
                timeout=30
            )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Answer generated in {elapsed:.2f}s")
            print(f"  Answer: {data['answer']}")
            print(f"  Latency: {data['latency']:.3f}s")
            print(f"  Tokens/sec: {data['tokens_per_second']:.1f}")
            return True
        else:
            print(f"✗ VQA failed: {response.status_code}")
            print(f"  Error: {response.json().get('detail', 'Unknown')}")
            return False
    except Exception as e:
        print(f"✗ VQA error: {e}")
        return False


def test_clear_cache():
    """Test cache clearing"""
    print("\nTesting /clear-cache endpoint...")
    try:
        response = requests.post("http://localhost:8000/clear-cache", timeout=5)
        if response.status_code == 200:
            print(f"✓ Cache cleared successfully")
            return True
        else:
            print(f"✗ Cache clear failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Cache clear error: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("EdgeVLM API Test Suite")
    print("=" * 60)
    
    # Parse arguments
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    question = sys.argv[2] if len(sys.argv) > 2 else "What is in this image?"
    
    results = []
    
    # Test health
    results.append(("Health", test_health()))
    
    # Test metrics
    results.append(("Metrics", test_metrics()))
    
    # Test caption (if image provided)
    if image_path:
        results.append(("Caption", test_caption(image_path)))
        results.append(("VQA", test_vqa(image_path, question)))
        results.append(("Clear Cache", test_clear_cache()))
    else:
        print("\n⚠ No image provided, skipping caption/VQA tests")
        print("  Usage: python test_api.py <image_path> [question]")
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

