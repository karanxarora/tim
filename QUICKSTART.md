# EdgeVLM Quick Start Guide

Get EdgeVLM running on your Raspberry Pi in 15 minutes.

## Prerequisites

- Raspberry Pi 4 or 5 with 8GB RAM
- Ubuntu 22.04 or Raspberry Pi OS 64-bit
- 10GB+ free storage
- Internet connection
- Python 3.8+

## Installation (5 minutes)

### 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/yourusername/edgevlm.git
cd edgevlm

# Run setup script
chmod +x setup.sh
./setup.sh

# This will:
# - Install system dependencies
# - Create Python virtual environment
# - Install all Python packages
# - Configure directories
```

### 2. Download Models (5-10 minutes)

```bash
# Activate environment
source venv/bin/activate

# Download TinyLlama (draft model)
python download_models.py --model tinyllama

# For MobileVLM-V2, follow manual setup (see below)
```

#### Manual MobileVLM-V2 Setup

MobileVLM-V2 requires manual download and conversion:

```bash
# Option 1: Use pre-converted GGUF (if available)
# Check HuggingFace: https://huggingface.co/models?search=mobilevlm

# Option 2: Convert yourself
git clone https://github.com/Meituan-AutoML/MobileVLM.git
cd MobileVLM
# Follow their instructions to download weights
# Then convert to GGUF using llama.cpp
```

**Note**: For testing, you can use TinyLlama alone (disable speculative decoding in config).

## Configuration (2 minutes)

Edit `config.yaml`:

```yaml
# For initial testing, disable speculative decoding
optimizations:
  speculative_decoding:
    enabled: false  # Set to false if only TinyLlama is available
```

## Running (1 minute)

```bash
# Start the API server
python main.py

# Server will start on http://localhost:8000
```

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     EdgeVLM Pipeline initialized successfully
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## Testing (2 minutes)

### Test 1: Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "pipeline_loaded": true,
  ...
}
```

### Test 2: Generate Caption

```bash
# Replace with your image
curl -X POST http://localhost:8000/caption \
  -F "image=@your_image.jpg" \
  -F "max_length=128"
```

Expected response:
```json
{
  "caption": "A description of your image",
  "latency": 3.5,
  ...
}
```

### Test 3: Visual Question Answering

```bash
curl -X POST http://localhost:8000/vqa \
  -F "image=@your_image.jpg" \
  -F "question=What is in this image?" \
  -F "max_length=64"
```

## Performance Optimization

### 1. Enable Performance Mode

```bash
# Set CPU governor to performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Verify
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
```

### 2. Cooling Setup

**Required**: Install active cooling (fan) for sustained performance.

```bash
# Check temperature
vcgencmd measure_temp

# Should be < 60°C under load
```

### 3. Increase Swap (if needed)

```bash
# Increase swap to 4GB
sudo dphys-swapfile swapoff
sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=4096/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

## Troubleshooting

### Issue: Out of Memory

**Solution**:
```yaml
# In config.yaml, reduce cache size
optimizations:
  memory_management:
    max_cache_size_mb: 256
```

### Issue: Slow Inference (>10s)

**Solutions**:
1. Enable early exit (in config.yaml)
2. Check CPU governor (see above)
3. Verify cooling (temperature < 70°C)
4. Increase thread count (in config.yaml)

### Issue: Model Loading Fails

**Solution**:
```bash
# Verify models exist
ls -lh models/

# Re-download if needed
python download_models.py --model tinyllama --force
```

## Next Steps

### 1. Run Benchmarks

```bash
# Run automated tests
python test_api.py your_image.jpg

# Run performance benchmark
curl -X POST http://localhost:8000/benchmark \
  -F "image=@your_image.jpg" \
  -F "num_runs=10"
```

### 2. Monitor Performance

```bash
# View metrics
curl http://localhost:8000/metrics | jq .

# Watch system stats
watch -n 1 'curl -s http://localhost:8000/health | jq .system_info'
```

### 3. Try Example Scripts

```bash
# Run usage examples
python example_usage.py
```

### 4. Deploy as Service

```bash
# Install as systemd service
sudo cp edgevlm.service /etc/systemd/system/
sudo systemctl enable edgevlm
sudo systemctl start edgevlm

# Check status
sudo systemctl status edgevlm
```

## Production Checklist

Before production deployment:

- [ ] Install active cooling (fan required)
- [ ] Set CPU governor to performance
- [ ] Configure adequate swap space
- [ ] Enable systemd service for auto-start
- [ ] Setup monitoring (see docs/API_REFERENCE.md)
- [ ] Test under load (run benchmarks)
- [ ] Configure firewall (if network-accessible)
- [ ] Setup log rotation
- [ ] Add authentication (if needed)
- [ ] Test recovery after reboot

## Performance Targets

You should achieve:

| Metric | Target | Notes |
|--------|--------|-------|
| Caption latency | 2-5s | With optimizations enabled |
| VQA latency | 1.5-4s | Simple questions faster |
| Memory usage | 3-5GB | Peak during inference |
| Temperature | <65°C | With active cooling |
| Early exit rate | 30-40% | For simple queries |

## Getting Help

1. **Check logs**:
   ```bash
   tail -f logs/edgevlm.log
   ```

2. **Read documentation**:
   - README.md - Overview
   - docs/TROUBLESHOOTING.md - Common issues
   - docs/API_REFERENCE.md - API details

3. **Run diagnostics**:
   ```bash
   python test_api.py
   ```

4. **Ask for help**:
   - GitHub Issues
   - GitHub Discussions

## Success!

If you can:
- ✓ Start the server without errors
- ✓ Get healthy response from `/health`
- ✓ Generate a caption in <5 seconds
- ✓ Keep temperature <70°C

**You're ready to use EdgeVLM!** 🎉

For advanced usage, see the full documentation in the `docs/` directory.

---

**Estimated total time**: 15-20 minutes (plus model download time)

