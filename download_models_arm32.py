#!/usr/bin/env python3
"""
Model Download Script for ARM32 (Raspberry Pi)
Downloads lightweight models optimized for ARM32 architecture
"""

import os
import sys
import requests
import yaml
from pathlib import Path
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelDownloader")

# ARM32 optimized model configurations
ARM32_MODELS = {
    "mobilevlm_v2_1.7b_q4": {
        "url": "https://huggingface.co/microsoft/MobileVLM-V2-1.7B-gguf/resolve/main/mobilevlm-v2-1.7b-q4_k_m.gguf",
        "filename": "mobilevlm_v2_1.7b_q4.gguf",
        "size_mb": 1200,
        "description": "MobileVLM V2 1.7B Q4 - Optimized for ARM32"
    },
    "tinyllama_1.1b_q4": {
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "filename": "tinyllama_1.1b_q4.gguf", 
        "size_mb": 700,
        "description": "TinyLlama 1.1B Q4 - Lightweight for ARM32"
    },
    "mobilevlm_v2_1.7b_q2": {
        "url": "https://huggingface.co/microsoft/MobileVLM-V2-1.7B-gguf/resolve/main/mobilevlm-v2-1.7b-q2_k.gguf",
        "filename": "mobilevlm_v2_1.7b_q2.gguf",
        "size_mb": 600,
        "description": "MobileVLM V2 1.7B Q2 - Ultra-lightweight for ARM32"
    }
}

def download_file(url: str, filename: str, expected_size_mb: int) -> bool:
    """Download a file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        filepath = Path("models") / filename
        
        with open(filepath, 'wb') as f:
            with tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=filename
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        # Verify file size
        actual_size_mb = filepath.stat().st_size / (1024 * 1024)
        if actual_size_mb < expected_size_mb * 0.9:  # Allow 10% tolerance
            logger.warning(f"File {filename} seems incomplete: {actual_size_mb:.1f}MB < {expected_size_mb}MB")
            return False
        
        logger.info(f"✅ Downloaded {filename} ({actual_size_mb:.1f}MB)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download {filename}: {e}")
        return False

def check_available_space() -> bool:
    """Check if there's enough space for models"""
    try:
        statvfs = os.statvfs(".")
        free_space_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
        
        total_required_gb = sum(model["size_mb"] for model in ARM32_MODELS.values()) / 1024
        
        logger.info(f"Available space: {free_space_gb:.1f}GB")
        logger.info(f"Required space: {total_required_gb:.1f}GB")
        
        if free_space_gb < total_required_gb:
            logger.error(f"❌ Insufficient space! Need {total_required_gb:.1f}GB, have {free_space_gb:.1f}GB")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to check disk space: {e}")
        return False

def update_config_for_arm32():
    """Update config.yaml for ARM32 optimizations"""
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        # Update model paths
        config["models"]["main_vlm"]["path"] = "models/mobilevlm_v2_1.7b_q4.gguf"
        config["models"]["draft_model"]["path"] = "models/tinyllama_1.1b_q4.gguf"
        
        # ARM32 specific optimizations
        config["inference"]["max_tokens"] = 64  # Reduced for ARM32
        config["inference"]["num_threads"] = 4  # Adjust based on Pi cores
        config["vision"]["input_size"] = [224, 224]  # Reduced for ARM32
        
        # Memory optimizations
        config["optimizations"]["memory_management"]["max_cache_size_mb"] = 256
        config["hardware"]["max_ram_mb"] = 3072
        
        with open("config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info("✅ Updated config.yaml for ARM32")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to update config: {e}")
        return False

def main():
    """Main download function"""
    print("=" * 60)
    print("EdgeVLM Model Downloader for ARM32 (Raspberry Pi)")
    print("=" * 60)
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Check available space
    if not check_available_space():
        print("\n❌ Insufficient disk space!")
        print("Please free up space or use a larger SD card.")
        return False
    
    # Show available models
    print("\nAvailable ARM32-optimized models:")
    for model_id, info in ARM32_MODELS.items():
        print(f"  • {info['description']} ({info['size_mb']}MB)")
    
    print("\nSelect models to download:")
    print("1. MobileVLM V2 1.7B Q4 (Recommended)")
    print("2. TinyLlama 1.1B Q4 (Draft model)")
    print("3. MobileVLM V2 1.7B Q2 (Ultra-lightweight)")
    print("4. All models")
    print("5. Skip download")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    models_to_download = []
    
    if choice == "1":
        models_to_download = ["mobilevlm_v2_1.7b_q4"]
    elif choice == "2":
        models_to_download = ["tinyllama_1.1b_q4"]
    elif choice == "3":
        models_to_download = ["mobilevlm_v2_1.7b_q2"]
    elif choice == "4":
        models_to_download = list(ARM32_MODELS.keys())
    elif choice == "5":
        print("Skipping model download")
        return True
    else:
        print("Invalid choice, downloading recommended models")
        models_to_download = ["mobilevlm_v2_1.7b_q4", "tinyllama_1.1b_q4"]
    
    # Download selected models
    print(f"\nDownloading {len(models_to_download)} model(s)...")
    success_count = 0
    
    for model_id in models_to_download:
        if model_id in ARM32_MODELS:
            model_info = ARM32_MODELS[model_id]
            if download_file(model_info["url"], model_info["filename"], model_info["size_mb"]):
                success_count += 1
    
    # Update configuration
    if success_count > 0:
        update_config_for_arm32()
    
    print("\n" + "=" * 60)
    print(f"Download complete: {success_count}/{len(models_to_download)} models downloaded")
    
    if success_count > 0:
        print("✅ Models ready for ARM32 deployment!")
        print("\nNext steps:")
        print("1. Test the system: python main.py --log-level DEBUG")
        print("2. Start the service: sudo systemctl start edgevlm")
    else:
        print("❌ No models downloaded successfully")
    
    print("=" * 60)
    
    return success_count > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
