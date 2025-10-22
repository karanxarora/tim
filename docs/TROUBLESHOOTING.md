# EdgeVLM Troubleshooting Guide

This guide covers common issues and their solutions when deploying EdgeVLM on Raspberry Pi and other ARM devices.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Model Loading Problems](#model-loading-problems)
3. [Memory Issues](#memory-issues)
4. [Performance Problems](#performance-problems)
5. [API Errors](#api-errors)
6. [Thermal Issues](#thermal-issues)
7. [Dependency Conflicts](#dependency-conflicts)
8. [Debug Techniques](#debug-techniques)

---

## Installation Issues

### Issue: `llama-cpp-python` Installation Fails

**Symptoms**:
```
error: command 'gcc' failed with exit status 1
Could not build wheels for llama-cpp-python
```

**Solutions**:

1. **Install build dependencies**:
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake python3-dev
sudo apt-get install -y libopenblas-dev liblapack-dev
```

2. **Install with CMAKE args**:
```bash
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" \
    pip install llama-cpp-python --no-cache-dir
```

3. **If still failing, try building from source**:
```bash
git clone --recursive https://github.com/abetlen/llama-cpp-python
cd llama-cpp-python
pip install -e .
```

### Issue: OpenCV Installation Problems

**Symptoms**:
```
ERROR: Could not build wheels for opencv-python
ImportError: libGL.so.1: cannot open shared object file
```

**Solutions**:

1. **Install OpenCV dependencies**:
```bash
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev
```

2. **For headless systems**:
```bash
pip install opencv-python-headless  # Use headless version
```

3. **Build from source (if needed)**:
```bash
sudo apt-get install -y python3-opencv
```

### Issue: PyTorch ARM Installation

**Symptoms**:
```
ERROR: Could not find a version that satisfies the requirement torch
```

**Solutions**:

1. **Use CPU-only PyTorch**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

2. **For specific ARM versions**:
```bash
# For aarch64
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
```

3. **Verify installation**:
```python
import torch
print(torch.__version__)
print(f"ARM NEON available: {torch.backends.cpu.is_available()}")
```

---

## Model Loading Problems

### Issue: Model File Not Found

**Symptoms**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'models/mobilevlm_v2_1.7b_q4.gguf'
```

**Solutions**:

1. **Check model directory**:
```bash
ls -lh models/
```

2. **Download missing models**:
```bash
python download_models.py --list
python download_models.py --model tinyllama
```

3. **Verify config paths match actual files**:
```bash
cat config.yaml | grep path
```

4. **Check file permissions**:
```bash
chmod 644 models/*.gguf
```

### Issue: Model Loading Fails with "Invalid GGUF"

**Symptoms**:
```
RuntimeError: Failed to load model: invalid GGUF file
```

**Solutions**:

1. **Re-download model (may be corrupted)**:
```bash
python download_models.py --model MODEL_NAME --force
```

2. **Verify file integrity**:
```bash
file models/tinyllama_1.1b_q4.gguf
# Should show: GGUF model file
```

3. **Check GGUF format version**:
```bash
# Ensure llama-cpp-python is up-to-date
pip install --upgrade llama-cpp-python
```

4. **Test model loading directly**:
```python
from llama_cpp import Llama

model = Llama(
    model_path="models/tinyllama_1.1b_q4.gguf",
    n_ctx=512,
    n_threads=2,
    verbose=True
)
print("Model loaded successfully!")
```

### Issue: GGUF Version Mismatch

**Symptoms**:
```
ValueError: Unsupported GGUF version
```

**Solutions**:

1. **Update llama-cpp-python**:
```bash
pip install --upgrade llama-cpp-python
```

2. **Convert model to latest GGUF format**:
```bash
# Download llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Convert model
python convert.py path/to/model --outtype q4_k_m --outfile new_model.gguf
```

---

## Memory Issues

### Issue: Out of Memory (OOM) Kills

**Symptoms**:
- Process suddenly terminates
- System becomes unresponsive
- `dmesg` shows: "Out of memory: Killed process"

**Solutions**:

1. **Increase swap space**:
```bash
# Check current swap
free -h

# Increase swap to 4GB
sudo dphys-swapfile swapoff
sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=4096/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Verify
free -h
```

2. **Reduce cache size in config.yaml**:
```yaml
optimizations:
  memory_management:
    max_cache_size_mb: 256  # Reduce from 512
    periodic_cache_clear: true
    clear_interval: 5  # Clear more frequently
```

3. **Enable aggressive cache compression**:
```yaml
optimizations:
  kv_cache_compression:
    gear_enabled: true
    pyramid_enabled: true
    compression_ratio: 0.3  # More aggressive (was 0.5)
```

4. **Disable speculative decoding** (saves ~1.5GB):
```yaml
optimizations:
  speculative_decoding:
    enabled: false
```

5. **Monitor memory usage**:
```python
import requests
metrics = requests.get('http://localhost:8000/metrics').json()
print(f"Memory: {metrics['system_metrics']['avg_memory_mb']:.0f}MB")
```

6. **Add memory limit with systemd**:
```ini
[Service]
MemoryMax=6G
MemoryHigh=5.5G
```

### Issue: Memory Leak

**Symptoms**:
- Memory usage grows over time
- Performance degrades after many requests
- System becomes slow

**Solutions**:

1. **Enable periodic cache clearing**:
```yaml
optimizations:
  memory_management:
    periodic_cache_clear: true
    clear_interval: 10
```

2. **Manual cache clear via API**:
```bash
curl -X POST http://localhost:8000/clear-cache
```

3. **Restart service periodically**:
```bash
# Add to crontab
0 */6 * * * systemctl restart edgevlm
```

4. **Profile memory usage**:
```python
import psutil
import os

process = psutil.Process(os.getpid())
print(f"Memory: {process.memory_info().rss / 1024**2:.0f} MB")
```

---

## Performance Problems

### Issue: Slow Inference (>10s latency)

**Symptoms**:
- Caption generation takes >10 seconds
- Low tokens/second (<5)

**Solutions**:

1. **Enable speculative decoding**:
```yaml
optimizations:
  speculative_decoding:
    enabled: true
    draft_tokens: 4
```

2. **Enable early exit**:
```yaml
optimizations:
  early_exit:
    enabled: true
    confidence_threshold: 0.9
```

3. **Increase thread count**:
```yaml
inference:
  num_threads: 4  # Use all cores
```

4. **Set CPU governor to performance**:
```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

5. **Check for thermal throttling**:
```bash
vcgencmd measure_temp
# Should be < 70°C
```

6. **Reduce batch size** (if memory-bound):
```yaml
inference:
  batch_size: 1  # Already minimal
  n_batch: 256   # Reduce from 512
```

7. **Profile bottlenecks**:
```bash
# Run with DEBUG logging
python main.py --log-level DEBUG

# Check logs for timing
grep "latency" logs/edgevlm.log
```

### Issue: Low GPU Utilization (If Using GPU)

**Note**: EdgeVLM is designed for CPU-only on Raspberry Pi. Skip if CPU-only.

**Solutions**:

1. **Enable GPU layers**:
```python
InferenceConfig(
    n_gpu_layers=20  # Offload layers to GPU
)
```

2. **Check CUDA availability**:
```python
import torch
print(torch.cuda.is_available())
```

### Issue: High Preprocessing Time

**Symptoms**:
- Vision preprocessing takes >200ms
- Overall latency dominated by preprocessing

**Solutions**:

1. **Increase vision threads**:
```yaml
vision:
  num_threads: 4  # More parallelism
```

2. **Use faster resize method**:
```yaml
vision:
  resize_method: "area"  # Faster than bicubic
```

3. **Profile preprocessing**:
```python
result = pipeline.generate_caption(image)
print(f"Preprocessing: {result['preprocessing_time']:.3f}s")
```

---

## API Errors

### Issue: 503 Service Unavailable

**Symptoms**:
```json
{"detail": "Pipeline not initialized"}
```

**Solutions**:

1. **Check API logs**:
```bash
tail -f logs/edgevlm.log
```

2. **Verify models are loaded**:
```bash
ls -lh models/
```

3. **Test pipeline initialization directly**:
```python
from pipeline import EdgeVLMPipeline
pipeline = EdgeVLMPipeline()  # Should not raise errors
```

4. **Check config syntax**:
```bash
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

### Issue: 500 Internal Server Error

**Symptoms**:
- API returns 500 error
- Request completes but fails

**Solutions**:

1. **Check API logs for traceback**:
```bash
tail -50 logs/edgevlm.log | grep ERROR
```

2. **Test endpoint with minimal request**:
```bash
curl -X POST http://localhost:8000/caption \
  -F "image=@test_image.jpg" \
  -F "max_length=10"
```

3. **Verify image format**:
```bash
file test_image.jpg
# Should be JPEG or PNG
```

4. **Check disk space**:
```bash
df -h
# Ensure /tmp has space for uploads
```

### Issue: Timeout Errors

**Symptoms**:
```
httpx.ReadTimeout: Read operation timed out
```

**Solutions**:

1. **Increase API timeout**:
```yaml
api:
  timeout: 60  # Increase from 30
```

2. **Use async processing for long tasks**:
```bash
# Submit request, poll for results
curl -X POST .../caption &  # Background
```

3. **Monitor request queue**:
```bash
curl http://localhost:8000/metrics | jq '.system_metrics'
```

---

## Thermal Issues

### Issue: CPU Overheating (>70°C)

**Symptoms**:
- Thermal throttling detected
- Performance degrades over time
- System becomes unstable

**Solutions**:

1. **Monitor temperature**:
```bash
watch -n 1 vcgencmd measure_temp
```

2. **Add cooling**:
- Install heatsink (passive cooling)
- Add fan (active cooling, recommended)
- Ensure airflow around device

3. **Reduce computational load**:
```yaml
inference:
  num_threads: 2  # Reduce from 4
```

4. **Add delays between requests**:
```python
import time
for image in images:
    result = generate_caption(image)
    time.sleep(2)  # Cool-down period
```

5. **Enable thermal monitoring**:
```python
from core.metrics import SystemMonitor
monitor = SystemMonitor()
stats = monitor.get_current_stats()
print(f"Temp: {stats['temperature_celsius']:.1f}°C")
```

6. **Automatic throttling**:
```python
# In pipeline
if temperature > 70:
    reduce_thread_count()
    wait_for_cooldown()
```

### Issue: Thermal Throttling Detected

**Symptoms**:
```
vcgencmd get_throttled
throttled=0x50000  # Throttling occurred
```

**Solutions**:

1. **Decode throttling status**:
```bash
vcgencmd get_throttled
# 0x0 = all good
# 0x50000 = throttling occurred in past
# 0x50005 = currently throttling
```

2. **Increase cooling solution**

3. **Reduce workload**

4. **Check power supply** (insufficient power can cause throttling):
```bash
vcgencmd get_config int | grep arm_freq
# Should be at full frequency
```

---

## Dependency Conflicts

### Issue: Version Conflicts

**Symptoms**:
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed
```

**Solutions**:

1. **Use fresh virtual environment**:
```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Pin specific versions**:
```bash
# If conflict with package X
pip install 'package==specific.version'
```

3. **Check for incompatibilities**:
```bash
pip check
```

### Issue: Import Errors

**Symptoms**:
```
ImportError: cannot import name 'X' from 'Y'
ModuleNotFoundError: No module named 'X'
```

**Solutions**:

1. **Verify installation**:
```bash
pip list | grep package_name
```

2. **Reinstall package**:
```bash
pip uninstall package_name
pip install package_name
```

3. **Check Python path**:
```python
import sys
print(sys.path)
```

---

## Debug Techniques

### Enable Debug Logging

```bash
# Start with debug logging
python main.py --log-level DEBUG

# Or set in code
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Profile Performance

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
result = pipeline.generate_caption(image)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

### Memory Profiling

```bash
# Install memory_profiler
pip install memory_profiler

# Profile function
python -m memory_profiler your_script.py
```

```python
from memory_profiler import profile

@profile
def generate_caption(image):
    # Function code
    pass
```

### Trace System Calls

```bash
# Linux strace
strace -c python main.py
```

### Monitor in Real-time

```bash
# htop for CPU/Memory
sudo apt-get install htop
htop

# Watch specific metrics
watch -n 1 'curl -s http://localhost:8000/metrics | jq ".system_metrics"'
```

### Test Individual Components

```python
# Test vision processor
from core import ARMOptimizedVisionProcessor
processor = ARMOptimizedVisionProcessor()
tensor = processor.process_image("test.jpg")
print(f"Shape: {tensor.shape}")

# Test inference engine
from core import ARMOptimizedInferenceEngine, InferenceConfig
config = InferenceConfig(model_path="models/tinyllama_1.1b_q4.gguf")
engine = ARMOptimizedInferenceEngine(config)
result = engine.generate("Hello", max_tokens=10)
print(result)
```

---

## Getting Help

If you're still stuck:

1. **Check logs**:
```bash
tail -100 logs/edgevlm.log
```

2. **Create minimal reproduction**:
```python
# Minimal script that reproduces issue
from pipeline import EdgeVLMPipeline
pipeline = EdgeVLMPipeline()
result = pipeline.generate_caption("test.jpg")
```

3. **Gather system info**:
```bash
# Create debug report
cat > debug_report.txt <<EOF
OS: $(uname -a)
Python: $(python --version)
RAM: $(free -h | grep Mem)
Disk: $(df -h | grep /$)
Temperature: $(vcgencmd measure_temp)
Throttling: $(vcgencmd get_throttled)
EOF
```

4. **Open GitHub issue** with:
   - Debug report
   - Minimal reproduction
   - Full error traceback
   - Steps to reproduce

---

## Quick Reference

### Common Commands

```bash
# Check system health
vcgencmd measure_temp
vcgencmd get_throttled
free -h
df -h

# Monitor API
curl http://localhost:8000/health
curl http://localhost:8000/metrics

# Clear cache
curl -X POST http://localhost:8000/clear-cache

# Restart service
sudo systemctl restart edgevlm

# View logs
tail -f logs/edgevlm.log
```

### Emergency Recovery

```bash
# If system becomes unresponsive
sudo systemctl stop edgevlm
sudo sync
sudo reboot

# Clear swap if needed
sudo swapoff -a
sudo swapon -a

# Free up memory
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
```

