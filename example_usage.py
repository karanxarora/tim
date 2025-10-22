"""
Example Usage of EdgeVLM API
Demonstrates common use cases and patterns
"""

import requests
import time
from pathlib import Path


# Configuration
API_BASE_URL = "http://localhost:8000"
IMAGE_PATH = "test_image.jpg"  # Replace with your image


def example_basic_caption():
    """Example 1: Basic image captioning"""
    print("=" * 60)
    print("Example 1: Basic Image Captioning")
    print("=" * 60)
    
    with open(IMAGE_PATH, 'rb') as f:
        response = requests.post(
            f"{API_BASE_URL}/caption",
            files={'image': f},
            data={'max_length': 128}
        )
    
    result = response.json()
    
    print(f"Caption: {result['caption']}")
    print(f"Latency: {result['latency']:.2f}s")
    print(f"Tokens/sec: {result['tokens_per_second']:.1f}")
    print()


def example_vqa():
    """Example 2: Visual Question Answering"""
    print("=" * 60)
    print("Example 2: Visual Question Answering")
    print("=" * 60)
    
    questions = [
        "What is the main subject of this image?",
        "What colors are present?",
        "Is this indoors or outdoors?",
    ]
    
    for question in questions:
        with open(IMAGE_PATH, 'rb') as f:
            response = requests.post(
                f"{API_BASE_URL}/vqa",
                files={'image': f},
                data={'question': question, 'max_length': 64}
            )
        
        result = response.json()
        print(f"Q: {question}")
        print(f"A: {result['answer']}")
        print(f"   (Latency: {result['latency']:.2f}s)")
        print()


def example_batch_processing():
    """Example 3: Batch processing multiple images"""
    print("=" * 60)
    print("Example 3: Batch Processing")
    print("=" * 60)
    
    # List of images to process
    image_dir = Path("test_images")
    if not image_dir.exists():
        print(f"Directory {image_dir} not found, using single image")
        images = [IMAGE_PATH]
    else:
        images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    
    print(f"Processing {len(images)} images...\n")
    
    results = []
    total_time = 0
    
    for i, image_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] Processing {image_path.name}...")
        
        start = time.time()
        with open(image_path, 'rb') as f:
            response = requests.post(
                f"{API_BASE_URL}/caption",
                files={'image': f}
            )
        elapsed = time.time() - start
        
        result = response.json()
        results.append(result)
        total_time += elapsed
        
        print(f"  Caption: {result['caption'][:60]}...")
        print(f"  Time: {elapsed:.2f}s")
        print()
    
    # Summary
    print(f"Batch processing complete!")
    print(f"  Total images: {len(images)}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Avg time per image: {total_time/len(images):.2f}s")
    print()


def example_monitoring():
    """Example 4: Monitoring system metrics"""
    print("=" * 60)
    print("Example 4: System Monitoring")
    print("=" * 60)
    
    # Get health status
    health = requests.get(f"{API_BASE_URL}/health").json()
    
    print("System Health:")
    print(f"  Status: {health['status']}")
    print(f"  CPU: {health['system_info']['cpu_percent']:.1f}%")
    print(f"  Memory: {health['system_info']['memory_percent']:.1f}%")
    if health['system_info']['temperature_celsius']:
        print(f"  Temperature: {health['system_info']['temperature_celsius']:.1f}°C")
    print()
    
    # Get performance metrics
    metrics = requests.get(f"{API_BASE_URL}/metrics").json()
    
    print("Performance Metrics:")
    inf_metrics = metrics['inference_metrics']
    print(f"  Total inferences: {inf_metrics['total_inferences']}")
    
    if inf_metrics['total_inferences'] > 0:
        print(f"  Avg latency: {inf_metrics['avg_latency']:.3f}s")
        print(f"  P95 latency: {inf_metrics['p95_latency']:.3f}s")
        print(f"  Avg tokens/sec: {inf_metrics['avg_tokens_per_second']:.1f}")
        print(f"  Early exit rate: {inf_metrics['early_exit_rate']:.1%}")
        print(f"  Max memory: {inf_metrics['max_memory_mb']:.0f}MB")
    print()


def example_performance_comparison():
    """Example 5: Compare with/without optimizations"""
    print("=" * 60)
    print("Example 5: Performance Comparison")
    print("=" * 60)
    
    print("Running inference to gather metrics...")
    
    # Run a few inferences
    num_runs = 5
    latencies = []
    
    for i in range(num_runs):
        with open(IMAGE_PATH, 'rb') as f:
            response = requests.post(
                f"{API_BASE_URL}/caption",
                files={'image': f}
            )
        result = response.json()
        latencies.append(result['latency'])
        
        print(f"  Run {i+1}: {result['latency']:.2f}s", end="")
        if result['early_exit']:
            print(f" (early exit at layer {result['exit_layer']})")
        else:
            print()
    
    # Statistics
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    
    print(f"\nStatistics:")
    print(f"  Average: {avg_latency:.2f}s")
    print(f"  Min: {min_latency:.2f}s")
    print(f"  Max: {max_latency:.2f}s")
    print(f"  Std dev: {(sum((x - avg_latency)**2 for x in latencies) / len(latencies))**0.5:.2f}s")
    print()


def example_error_handling():
    """Example 6: Proper error handling"""
    print("=" * 60)
    print("Example 6: Error Handling")
    print("=" * 60)
    
    def safe_caption(image_path, max_retries=3):
        """Caption with retry logic"""
        for attempt in range(max_retries):
            try:
                with open(image_path, 'rb') as f:
                    response = requests.post(
                        f"{API_BASE_URL}/caption",
                        files={'image': f},
                        timeout=30
                    )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 503:
                    print(f"  Service unavailable, retrying... ({attempt+1}/{max_retries})")
                    time.sleep(2)
                else:
                    error = response.json().get('detail', 'Unknown error')
                    print(f"  Error: {error}")
                    return None
                    
            except requests.Timeout:
                print(f"  Timeout, retrying... ({attempt+1}/{max_retries})")
                time.sleep(2)
            except Exception as e:
                print(f"  Exception: {e}")
                return None
        
        print(f"  Failed after {max_retries} attempts")
        return None
    
    result = safe_caption(IMAGE_PATH)
    if result:
        print(f"✓ Success: {result['caption']}")
    else:
        print(f"✗ Failed to generate caption")
    print()


def example_streaming_workflow():
    """Example 7: Continuous processing workflow"""
    print("=" * 60)
    print("Example 7: Streaming Workflow (simulated)")
    print("=" * 60)
    
    print("Simulating real-time image processing...")
    print("(In production, this would process camera frames)")
    print()
    
    # Simulate processing frames
    frame_count = 5
    process_every_n = 1  # Process every frame (adjust for performance)
    
    for frame_num in range(frame_count):
        if frame_num % process_every_n == 0:
            print(f"Frame {frame_num}: Processing...")
            
            with open(IMAGE_PATH, 'rb') as f:
                response = requests.post(
                    f"{API_BASE_URL}/caption",
                    files={'image': f},
                    data={'max_length': 64}  # Shorter for real-time
                )
            
            result = response.json()
            print(f"  Caption: {result['caption'][:50]}...")
            print(f"  Latency: {result['latency']:.2f}s")
            
            # Check if we can maintain real-time (e.g., 1 fps)
            if result['latency'] > 1.0:
                print(f"  ⚠ Warning: Latency exceeds 1s target")
        else:
            print(f"Frame {frame_num}: Skipped")
        
        time.sleep(0.5)  # Simulate frame rate
    
    print("\nStreaming simulation complete")
    print()


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("EdgeVLM API Usage Examples")
    print("=" * 60 + "\n")
    
    # Check if API is available
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if response.status_code != 200:
            print("❌ API is not available. Please start the server first:")
            print("   python main.py")
            return
    except:
        print("❌ Cannot connect to API. Please start the server first:")
        print("   python main.py")
        return
    
    # Check if image exists
    if not Path(IMAGE_PATH).exists():
        print(f"⚠ Image not found: {IMAGE_PATH}")
        print("  Please update IMAGE_PATH in this script or provide a test image")
        return
    
    # Run examples
    try:
        example_basic_caption()
        example_vqa()
        example_monitoring()
        example_performance_comparison()
        example_error_handling()
        example_streaming_workflow()
        
        # Optional: batch processing if directory exists
        if Path("test_images").exists():
            example_batch_processing()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    print("=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

