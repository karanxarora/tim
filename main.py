"""
EdgeVLM Main Entry Point
Launch the API server or run standalone inference
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from api import app
import uvicorn


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/edgevlm.log', mode='a')
        ]
    )


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="EdgeVLM - Edge-Optimized Vision-Language Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start API server
  python main.py --host 0.0.0.0 --port 8000
  
  # Start with debug logging
  python main.py --log-level DEBUG
  
  # Enable auto-reload for development
  python main.py --reload
        """
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number (default: 8000)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Configuration file path (default: config.yaml)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("benchmarks", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("cache", exist_ok=True)
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger("EdgeVLM")
    
    logger.info("=" * 60)
    logger.info("EdgeVLM - Edge-Optimized Vision-Language Model")
    logger.info("=" * 60)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Log Level: {args.log_level}")
    logger.info("=" * 60)
    
    # Set config path as environment variable
    os.environ['EDGEVLM_CONFIG'] = args.config
    
    try:
        # Start API server
        uvicorn.run(
            "api:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,
            log_level=args.log_level.lower(),
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("\nShutting down gracefully...")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

