"""
Model Download Script
Downloads and prepares quantized models for EdgeVLM
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import requests
from tqdm import tqdm
import hashlib


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ModelDownloader")


class ModelDownloader:
    """Downloads and verifies models"""
    
    # Model registry with download links
    # Note: These are placeholder URLs - replace with actual model locations
    MODELS = {
        "tinyllama": {
            "name": "TinyLlama-1.1B-Chat-Q4_K_M",
            "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "filename": "tinyllama_1.1b_q4.gguf",
            "size_mb": 669,
            "sha256": None  # Add checksum if available
        },
        "mobilevlm": {
            "name": "MobileVLM-V2-1.7B-Q4_K_M",
            "url": None,  # Needs to be quantized and hosted
            "filename": "mobilevlm_v2_1.7b_q4.gguf",
            "size_mb": 1000,
            "sha256": None,
            "note": "MobileVLM-V2 needs to be manually quantized and converted to GGUF format"
        },
        "vision_encoder": {
            "name": "MobileVLM Vision Encoder",
            "url": None,
            "filename": "vision_encoder.pth",
            "size_mb": 50,
            "sha256": None,
            "note": "Vision encoder weights from MobileVLM-V2"
        }
    }
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True, parents=True)
    
    def download_file(self, url: str, destination: Path, expected_size_mb: int):
        """Download file with progress bar"""
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            logger.info(f"Downloading to {destination}")
            logger.info(f"Size: {total_size / (1024*1024):.2f} MB")
            
            with open(destination, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            logger.info(f"Download complete: {destination}")
            return True
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            if destination.exists():
                destination.unlink()
            return False
    
    def verify_checksum(self, filepath: Path, expected_sha256: str) -> bool:
        """Verify file checksum"""
        if not expected_sha256:
            logger.warning("No checksum provided, skipping verification")
            return True
        
        logger.info("Verifying checksum...")
        sha256_hash = hashlib.sha256()
        
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        file_hash = sha256_hash.hexdigest()
        
        if file_hash == expected_sha256:
            logger.info("Checksum verified ✓")
            return True
        else:
            logger.error(f"Checksum mismatch!")
            logger.error(f"Expected: {expected_sha256}")
            logger.error(f"Got: {file_hash}")
            return False
    
    def download_model(self, model_key: str, force: bool = False) -> bool:
        """Download a specific model"""
        if model_key not in self.MODELS:
            logger.error(f"Unknown model: {model_key}")
            logger.info(f"Available models: {list(self.MODELS.keys())}")
            return False
        
        model_info = self.MODELS[model_key]
        destination = self.models_dir / model_info['filename']
        
        # Check if already exists
        if destination.exists() and not force:
            logger.info(f"Model already exists: {destination}")
            logger.info("Use --force to re-download")
            return True
        
        # Check if URL is available
        if not model_info.get('url'):
            logger.warning(f"No download URL for {model_key}")
            if 'note' in model_info:
                logger.info(f"Note: {model_info['note']}")
            logger.info("\nManual download instructions:")
            logger.info(f"  1. Download or convert the model")
            logger.info(f"  2. Save as: {destination}")
            return False
        
        # Download
        logger.info(f"Downloading {model_info['name']}...")
        success = self.download_file(
            model_info['url'],
            destination,
            model_info['size_mb']
        )
        
        if not success:
            return False
        
        # Verify checksum if available
        if model_info.get('sha256'):
            if not self.verify_checksum(destination, model_info['sha256']):
                logger.error("Checksum verification failed!")
                destination.unlink()
                return False
        
        logger.info(f"Successfully downloaded: {model_key}")
        return True
    
    def download_all(self, force: bool = False):
        """Download all available models"""
        logger.info("Downloading all models...")
        
        results = {}
        for model_key in self.MODELS:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {model_key}")
            logger.info(f"{'='*60}")
            
            results[model_key] = self.download_model(model_key, force)
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("Download Summary")
        logger.info(f"{'='*60}")
        
        for model_key, success in results.items():
            status = "✓" if success else "✗"
            logger.info(f"{status} {model_key}: {self.MODELS[model_key]['name']}")
        
        successful = sum(1 for s in results.values() if s)
        logger.info(f"\nSuccessfully downloaded: {successful}/{len(results)}")
    
    def list_models(self):
        """List available models"""
        logger.info("Available models:")
        logger.info(f"{'='*60}")
        
        for key, info in self.MODELS.items():
            status = "✓" if (self.models_dir / info['filename']).exists() else "✗"
            logger.info(f"\n{status} {key}:")
            logger.info(f"  Name: {info['name']}")
            logger.info(f"  File: {info['filename']}")
            logger.info(f"  Size: ~{info['size_mb']} MB")
            
            if info.get('url'):
                logger.info(f"  URL: {info['url'][:80]}...")
            else:
                logger.info(f"  URL: Not available (manual download required)")
            
            if info.get('note'):
                logger.info(f"  Note: {info['note']}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare models for EdgeVLM",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory to store models (default: models)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Download specific model (e.g., tinyllama, mobilevlm)"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available models"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file exists"
    )
    
    args = parser.parse_args()
    
    downloader = ModelDownloader(args.models_dir)
    
    if args.list:
        downloader.list_models()
    elif args.all:
        downloader.download_all(args.force)
    elif args.model:
        downloader.download_model(args.model, args.force)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python download_models.py --list")
        print("  python download_models.py --model tinyllama")
        print("  python download_models.py --all")


if __name__ == "__main__":
    main()

