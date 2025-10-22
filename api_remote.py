"""
Remote Camera API for EdgeVLM
Enhanced API endpoints for remote camera integration
"""

import os
import logging
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
import base64
from io import BytesIO

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from pipeline import EdgeVLMPipeline
from core.camera_integration import (
    RemoteImageProcessor,
    CameraManager,
    CameraConfig,
    create_camera_configs
)
from PIL import Image
import numpy as np


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EdgeVLM-Remote-API")

# Initialize FastAPI app
app = FastAPI(
    title="EdgeVLM Remote Camera API",
    description="Real-time Vision-Language API for Remote Camera Integration",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
pipeline: Optional[EdgeVLMPipeline] = None
camera_processor: Optional[RemoteImageProcessor] = None


# Request/Response Models
class CameraConfigRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    source_type: str = Field(..., regex="^(rtsp|http|mjpeg|usb|file)$")
    url: Optional[str] = None
    device_id: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    resolution: List[int] = Field(default=[640, 480], min_items=2, max_items=2)
    fps: int = Field(default=15, ge=1, le=60)
    timeout: int = Field(default=10, ge=1, le=60)


class CameraResponse(BaseModel):
    name: str
    running: bool
    connected: bool
    frame_count: int
    config: Dict[str, Any]
    last_frame_time: Optional[str] = None


class ProcessFrameRequest(BaseModel):
    camera_name: str
    task: str = Field(default="caption", regex="^(caption|vqa)$")
    question: Optional[str] = Field(default=None, max_length=500)
    max_length: int = Field(default=128, ge=1, le=256)


class ProcessFrameResponse(BaseModel):
    camera_name: str
    task: str
    result: Dict[str, Any]
    frame_timestamp: float
    frame_number: int
    processing_time: float
    timestamp: str


class StreamConfig(BaseModel):
    camera_name: str
    task: str = Field(default="caption", regex="^(caption|vqa)$")
    question: Optional[str] = Field(default=None, max_length=500)
    interval: float = Field(default=5.0, ge=1.0, le=60.0)  # seconds between frames


# Startup/Shutdown Events
@app.on_event("startup")
async def startup_event():
    """Initialize pipeline and camera processor on startup"""
    global pipeline, camera_processor
    logger.info("Starting EdgeVLM Remote Camera API...")
    
    try:
        # Initialize main pipeline
        pipeline = EdgeVLMPipeline(
            config_path="config.yaml",
            logger=logger
        )
        
        # Initialize camera processor
        camera_processor = RemoteImageProcessor(
            pipeline=pipeline,
            logger=logger
        )
        
        logger.info("Remote camera API initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize remote camera API: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global camera_processor
    logger.info("Shutting down Remote Camera API...")
    
    if camera_processor:
        camera_processor.camera_manager.stop_all()
    
    if pipeline:
        pipeline.shutdown()
    
    logger.info("Shutdown complete")


# API Endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint with camera-specific info"""
    return {
        "service": "EdgeVLM Remote Camera API",
        "version": "1.0.0",
        "description": "Real-time Vision-Language Model for Remote Camera Integration",
        "endpoints": {
            "cameras": "/cameras - Camera management",
            "process": "/process - Process camera frame",
            "stream": "/stream - Live processing stream",
            "health": "/health - Health check"
        }
    }


@app.get("/health", response_model=dict)
async def health_check():
    """Enhanced health check with camera status"""
    if pipeline is None or camera_processor is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Get system metrics
    system_metrics = pipeline.system_monitor.get_current_stats()
    
    # Get camera status
    camera_status = camera_processor.get_camera_status()
    
    return {
        "status": "healthy",
        "pipeline_loaded": True,
        "cameras_configured": len(camera_status),
        "active_cameras": sum(1 for cam in camera_status.values() if cam['running']),
        "timestamp": datetime.now().isoformat(),
        "system_info": {
            "cpu_percent": system_metrics.get("cpu_percent", 0),
            "memory_percent": system_metrics.get("memory_percent", 0),
            "temperature_celsius": system_metrics.get("temperature_celsius")
        },
        "camera_status": camera_status
    }


# Camera Management Endpoints
@app.post("/cameras", response_model=dict)
async def add_camera(config: CameraConfigRequest):
    """Add a new camera source"""
    if camera_processor is None:
        raise HTTPException(status_code=503, detail="Camera processor not initialized")
    
    try:
        camera_config = CameraConfig(
            source_type=config.source_type,
            url=config.url,
            device_id=config.device_id,
            username=config.username,
            password=config.password,
            resolution=tuple(config.resolution),
            fps=config.fps,
            timeout=config.timeout
        )
        
        success = camera_processor.add_camera(config.name, camera_config)
        
        if success:
            return {
                "status": "success",
                "message": f"Camera '{config.name}' added successfully",
                "camera_name": config.name
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to add camera")
            
    except Exception as e:
        logger.error(f"Failed to add camera: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add camera: {str(e)}")


@app.get("/cameras", response_model=Dict[str, CameraResponse])
async def list_cameras():
    """List all configured cameras"""
    if camera_processor is None:
        raise HTTPException(status_code=503, detail="Camera processor not initialized")
    
    camera_status = camera_processor.get_camera_status()
    
    response = {}
    for name, status in camera_status.items():
        response[name] = CameraResponse(
            name=name,
            running=status['running'],
            connected=status['connected'],
            frame_count=status['frame_count'],
            config=status['config']
        )
    
    return response


@app.post("/cameras/{camera_name}/start", response_model=dict)
async def start_camera(camera_name: str):
    """Start a specific camera"""
    if camera_processor is None:
        raise HTTPException(status_code=503, detail="Camera processor not initialized")
    
    success = camera_processor.start_camera(camera_name)
    
    if success:
        return {
            "status": "success",
            "message": f"Camera '{camera_name}' started successfully"
        }
    else:
        raise HTTPException(status_code=400, detail=f"Failed to start camera '{camera_name}'")


@app.post("/cameras/{camera_name}/stop", response_model=dict)
async def stop_camera(camera_name: str):
    """Stop a specific camera"""
    if camera_processor is None:
        raise HTTPException(status_code=503, detail="Camera processor not initialized")
    
    camera_processor.stop_camera(camera_name)
    
    return {
        "status": "success",
        "message": f"Camera '{camera_name}' stopped successfully"
    }


# Processing Endpoints
@app.post("/process", response_model=ProcessFrameResponse)
async def process_frame(request: ProcessFrameRequest):
    """Process a single frame from a camera"""
    if camera_processor is None:
        raise HTTPException(status_code=503, detail="Camera processor not initialized")
    
    start_time = datetime.now()
    
    try:
        result = camera_processor.process_camera_frame(
            camera_name=request.camera_name,
            task=request.task,
            question=request.question
        )
        
        if result is None:
            raise HTTPException(
                status_code=404, 
                detail=f"No frame available from camera '{request.camera_name}'"
            )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ProcessFrameResponse(
            camera_name=result['camera_name'],
            task=request.task,
            result=result,
            frame_timestamp=result['frame_timestamp'],
            frame_number=result['frame_number'],
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to process frame: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/process/latest", response_model=ProcessFrameResponse)
async def process_latest_frame(
    camera_name: str = Query(..., description="Camera name"),
    task: str = Query("caption", description="Task type: caption or vqa"),
    question: Optional[str] = Query(None, description="Question for VQA"),
    max_length: int = Query(128, description="Maximum response length")
):
    """Process the latest frame from a camera (GET endpoint for easy testing)"""
    request = ProcessFrameRequest(
        camera_name=camera_name,
        task=task,
        question=question,
        max_length=max_length
    )
    
    return await process_frame(request)


# Streaming Endpoints
@app.get("/stream/{camera_name}")
async def stream_processing(
    camera_name: str,
    task: str = Query("caption", description="Task type"),
    question: Optional[str] = Query(None, description="Question for VQA"),
    interval: float = Query(5.0, description="Interval between frames (seconds)")
):
    """Stream live processing results from a camera"""
    if camera_processor is None:
        raise HTTPException(status_code=503, detail="Camera processor not initialized")
    
    async def generate_stream():
        while True:
            try:
                result = camera_processor.process_camera_frame(
                    camera_name=camera_name,
                    task=task,
                    question=question
                )
                
                if result:
                    yield f"data: {result}\n\n"
                else:
                    yield f"data: {{'error': 'No frame available'}}\n\n"
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                yield f"data: {{'error': '{str(e)}'}}\n\n"
                await asyncio.sleep(interval)
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


# Utility Endpoints
@app.get("/cameras/presets", response_model=dict)
async def get_camera_presets():
    """Get common camera configuration presets"""
    presets = create_camera_configs()
    
    # Convert to dict format for JSON response
    response = {}
    for name, config in presets.items():
        response[name] = {
            "source_type": config.source_type,
            "url": config.url,
            "device_id": config.device_id,
            "username": config.username,
            "password": config.password,
            "resolution": config.resolution,
            "fps": config.fps,
            "timeout": config.timeout
        }
    
    return response


@app.post("/cameras/{camera_name}/test", response_model=dict)
async def test_camera(camera_name: str):
    """Test camera connection without processing"""
    if camera_processor is None:
        raise HTTPException(status_code=503, detail="Camera processor not initialized")
    
    try:
        # Try to get a frame
        frame_data = camera_processor.camera_manager.get_frame(camera_name)
        
        if frame_data:
            return {
                "status": "success",
                "message": f"Camera '{camera_name}' is working",
                "frame_info": {
                    "timestamp": frame_data['timestamp'],
                    "frame_number": frame_data['frame_number'],
                    "shape": frame_data['frame'].shape
                }
            }
        else:
            return {
                "status": "error",
                "message": f"No frame available from camera '{camera_name}'"
            }
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"Camera test failed: {str(e)}"
        }


# Main entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EdgeVLM Remote Camera API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8001, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "api_remote:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )
