# PyTorch on ARM32 (armv7l) - Complete Guide

## The Problem

**PyTorch does NOT provide official wheels for ARM32 (armv7l) architecture.**

### Available PyTorch Architectures:
- ✅ **x86_64** (Intel/AMD 64-bit)
- ✅ **aarch64** (ARM 64-bit) 
- ❌ **armv7l** (ARM 32-bit) - **NOT AVAILABLE**

## Solutions

### Option 1: Use 64-bit Raspberry Pi OS (RECOMMENDED)

This is the **best solution** for PyTorch compatibility:

1. **Download 64-bit Raspberry Pi OS**:
   - Go to [raspberrypi.org/downloads](https://www.raspberrypi.org/downloads/)
   - Download "Raspberry Pi OS (64-bit)"
   - Use Raspberry Pi Imager to flash to SD card

2. **Install PyTorch on 64-bit OS**:
   ```bash
   pip install torch torchvision torchaudio
   ```

3. **Benefits**:
   - Full PyTorch support
   - Better performance
   - Access to all ML libraries
   - Future-proof

### Option 2: Use llama-cpp-python Only (Current Setup)

Our current setup uses `llama-cpp-python` which **does support ARM32**:

```bash
# This works on ARM32
pip install llama-cpp-python
```

**Pros**:
- Works on ARM32
- Lightweight
- Good for inference

**Cons**:
- Limited to GGUF models
- No training capabilities
- Limited vision model support

### Option 3: Build PyTorch from Source (NOT RECOMMENDED)

This is complex and may not work properly:

```bash
# Install build dependencies
sudo apt install build-essential cmake git

# Clone PyTorch
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch

# Build (this will take hours and may fail)
python setup.py install
```

**Issues**:
- Takes 6+ hours to build
- High memory usage (may fail on 4GB Pi)
- JIT compiler may not work
- Many features disabled

## Current EdgeVLM Setup

Our setup uses **llama-cpp-python** which works perfectly on ARM32:

### What Works:
- ✅ **llama-cpp-python** - Full ARM32 support
- ✅ **OpenCV** - ARM32 wheels available
- ✅ **FastAPI** - Pure Python, works everywhere
- ✅ **GGUF models** - MobileVLM, TinyLlama, etc.

### What Doesn't Work:
- ❌ **PyTorch** - No ARM32 wheels
- ❌ **Transformers** - Depends on PyTorch
- ❌ **Accelerate** - Depends on PyTorch

## Model Compatibility

### Supported Models (GGUF format):
- ✅ MobileVLM V2 1.7B
- ✅ TinyLlama 1.1B
- ✅ LLaVA models
- ✅ Any GGUF quantized model

### Not Supported:
- ❌ Hugging Face Transformers models
- ❌ PyTorch native models
- ❌ Custom trained models (unless converted to GGUF)

## Performance Comparison

### ARM32 (Current Setup):
- **Inference**: Good (llama-cpp-python)
- **Memory**: ~2GB for models
- **Speed**: 3-8 seconds per inference
- **Compatibility**: Limited to GGUF models

### ARM64 (64-bit OS):
- **Inference**: Excellent (PyTorch)
- **Memory**: ~3GB for models
- **Speed**: 1-3 seconds per inference
- **Compatibility**: Full PyTorch ecosystem

## Recommendation

**For best results, use 64-bit Raspberry Pi OS:**

1. **Flash 64-bit OS** to your SD card
2. **Run our setup script** (it will detect ARM64 and install PyTorch)
3. **Get full PyTorch support** with all features

**If you must use 32-bit OS:**
- Our current setup works with llama-cpp-python
- Limited to GGUF models
- Still functional for basic inference

## Quick Fix for Current Setup

If you want to continue with ARM32, our setup script now handles this properly:

```bash
# Run the updated setup script
./setup-arm32.sh

# It will:
# 1. Detect ARM32 architecture
# 2. Skip PyTorch installation
# 3. Install llama-cpp-python
# 4. Set up everything else
```

The system will work, just with limited model support compared to a 64-bit setup.
