"""
Camera Registration API for EdgeVLM
Cameras self-register and establish persistent connections
"""

import os
import logging
import asyncio
import time
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from pipeline import EdgeVLMPipeline
from core.camera_integration import (
    RemoteImageProcessor,
    CameraManager,
    CameraConfig,
    RemoteCameraCapture
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EdgeVLM-Camera-Registration")

# Initialize FastAPI app
app = FastAPI(
    title="EdgeVLM Camera Registration API",
    description="Camera self-registration and persistent connection management",
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
registered_cameras: Dict[str, Dict[str, Any]] = {}
camera_sessions: Dict[str, Dict[str, Any]] = {}


@dataclass
class CameraRegistration:
    """Camera registration data"""
    camera_id: str
    name: str
    location: str
    source_type: str
    url: Optional[str] = None
    device_id: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    resolution: tuple = (640, 480)
    fps: int = 15
    capabilities: List[str] = None
    metadata: Dict[str, Any] = None
    registered_at: datetime = None
    last_heartbeat: datetime = None
    status: str = "registered"  # registered, active, inactive, error
    api_key: str = None


# Request/Response Models
class CameraRegistrationRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    location: str = Field(..., min_length=1, max_length=200)
    source_type: str = Field(..., regex="^(rtsp|http|mjpeg|usb|file)$")
    url: Optional[str] = None
    device_id: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    resolution: List[int] = Field(default=[640, 480], min_items=2, max_items=2)
    fps: int = Field(default=15, ge=1, le=60)
    capabilities: List[str] = Field(default=["caption", "vqa"])
    metadata: Dict[str, Any] = Field(default={})


class CameraRegistrationResponse(BaseModel):
    camera_id: str
    api_key: str
    status: str
    message: str
    endpoints: Dict[str, str]
    config: Dict[str, Any]


class HeartbeatRequest(BaseModel):
    camera_id: str
    api_key: str
    status: str = Field(default="active")
    frame_count: int = Field(default=0)
    error_message: Optional[str] = None


class HeartbeatResponse(BaseModel):
    status: str
    message: str
    commands: List[Dict[str, Any]] = Field(default=[])


class FrameUploadRequest(BaseModel):
    camera_id: str
    api_key: str
    frame_data: str  # Base64 encoded image
    timestamp: Optional[float] = None
    frame_number: Optional[int] = None
    task: str = Field(default="caption", regex="^(caption|vqa)$")
    question: Optional[str] = Field(default=None, max_length=500)


class FrameUploadResponse(BaseModel):
    camera_id: str
    task: str
    result: Dict[str, Any]
    processing_time: float
    timestamp: str


class CameraStatusResponse(BaseModel):
    camera_id: str
    name: str
    location: str
    status: str
    registered_at: str
    last_heartbeat: str
    frame_count: int
    uptime_seconds: float


# Startup/Shutdown Events
@app.on_event("startup")
async def startup_event():
    """Initialize pipeline and camera processor on startup"""
    global pipeline, camera_processor
    logger.info("Starting EdgeVLM Camera Registration API...")
    
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
        
        # Start background cleanup task
        asyncio.create_task(cleanup_inactive_cameras())
        
        logger.info("Camera Registration API initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize camera registration API: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global camera_processor
    logger.info("Shutting down Camera Registration API...")
    
    if camera_processor:
        camera_processor.camera_manager.stop_all()
    
    if pipeline:
        pipeline.shutdown()
    
    logger.info("Shutdown complete")


# Background Tasks
async def cleanup_inactive_cameras():
    """Periodically clean up inactive cameras"""
    while True:
        try:
            current_time = datetime.now()
            inactive_threshold = timedelta(minutes=5)  # 5 minutes without heartbeat
            
            inactive_cameras = []
            for camera_id, camera_data in registered_cameras.items():
                if camera_data['last_heartbeat']:
                    time_since_heartbeat = current_time - camera_data['last_heartbeat']
                    if time_since_heartbeat > inactive_threshold:
                        inactive_cameras.append(camera_id)
            
            for camera_id in inactive_cameras:
                logger.info(f"Cleaning up inactive camera: {camera_id}")
                await deactivate_camera(camera_id)
            
            await asyncio.sleep(60)  # Check every minute
            
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
            await asyncio.sleep(60)


async def deactivate_camera(camera_id: str):
    """Deactivate a camera and clean up resources"""
    if camera_id in registered_cameras:
        registered_cameras[camera_id]['status'] = 'inactive'
        
        # Stop camera in processor if it exists
        if camera_processor and camera_id in camera_processor.camera_manager.cameras:
            camera_processor.stop_camera(camera_id)
        
        logger.info(f"Camera {camera_id} deactivated")


# Helper Functions
def generate_api_key() -> str:
    """Generate a unique API key for camera"""
    return str(uuid.uuid4()).replace('-', '')[:32]


def validate_camera_credentials(camera_id: str, api_key: str) -> bool:
    """Validate camera credentials"""
    if camera_id not in registered_cameras:
        return False
    
    camera_data = registered_cameras[camera_id]
    return camera_data.get('api_key') == api_key


def get_camera_endpoints(camera_id: str) -> Dict[str, str]:
    """Get API endpoints for a specific camera"""
    base_url = "http://localhost:8002"  # This API's base URL
    
    return {
        "heartbeat": f"{base_url}/cameras/{camera_id}/heartbeat",
        "upload_frame": f"{base_url}/cameras/{camera_id}/upload",
        "stream": f"{base_url}/cameras/{camera_id}/stream",
        "status": f"{base_url}/cameras/{camera_id}/status",
        "config": f"{base_url}/cameras/{camera_id}/config"
    }


# API Endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint with registration info"""
    return {
        "service": "EdgeVLM Camera Registration API",
        "version": "1.0.0",
        "description": "Camera self-registration and persistent connection management",
        "registered_cameras": len(registered_cameras),
        "active_cameras": sum(1 for cam in registered_cameras.values() if cam['status'] == 'active'),
        "endpoints": {
            "register": "/register - Register new camera",
            "heartbeat": "/cameras/{camera_id}/heartbeat - Send heartbeat",
            "upload": "/cameras/{camera_id}/upload - Upload frame",
            "status": "/cameras/{camera_id}/status - Get camera status"
        }
    }


@app.get("/health", response_model=dict)
async def health_check():
    """Health check with camera statistics"""
    if pipeline is None or camera_processor is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Get system metrics
    system_metrics = pipeline.system_monitor.get_current_stats()
    
    return {
        "status": "healthy",
        "pipeline_loaded": True,
        "registered_cameras": len(registered_cameras),
        "active_cameras": sum(1 for cam in registered_cameras.values() if cam['status'] == 'active'),
        "timestamp": datetime.now().isoformat(),
        "system_info": {
            "cpu_percent": system_metrics.get("cpu_percent", 0),
            "memory_percent": system_metrics.get("memory_percent", 0),
            "temperature_celsius": system_metrics.get("temperature_celsius")
        }
    }


@app.post("/register", response_model=CameraRegistrationResponse)
async def register_camera(request: CameraRegistrationRequest):
    """Register a new camera"""
    if pipeline is None or camera_processor is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Generate unique camera ID and API key
    camera_id = str(uuid.uuid4())
    api_key = generate_api_key()
    
    # Create camera registration
    registration = CameraRegistration(
        camera_id=camera_id,
        name=request.name,
        location=request.location,
        source_type=request.source_type,
        url=request.url,
        device_id=request.device_id,
        username=request.username,
        password=request.password,
        resolution=tuple(request.resolution),
        fps=request.fps,
        capabilities=request.capabilities,
        metadata=request.metadata,
        registered_at=datetime.now(),
        last_heartbeat=datetime.now(),
        status="registered",
        api_key=api_key
    )
    
    # Store registration
    registered_cameras[camera_id] = asdict(registration)
    
    # Create camera config for processor
    camera_config = CameraConfig(
        source_type=request.source_type,
        url=request.url,
        device_id=request.device_id,
        username=request.username,
        password=request.password,
        resolution=tuple(request.resolution),
        fps=request.fps
    )
    
    # Add to processor
    camera_processor.add_camera(camera_id, camera_config)
    
    # Get endpoints
    endpoints = get_camera_endpoints(camera_id)
    
    logger.info(f"Camera registered: {camera_id} ({request.name}) at {request.location}")
    
    return CameraRegistrationResponse(
        camera_id=camera_id,
        api_key=api_key,
        status="registered",
        message="Camera registered successfully",
        endpoints=endpoints,
        config={
            "resolution": request.resolution,
            "fps": request.fps,
            "capabilities": request.capabilities
        }
    )


@app.post("/cameras/{camera_id}/heartbeat", response_model=HeartbeatResponse)
async def camera_heartbeat(camera_id: str, request: HeartbeatRequest):
    """Receive heartbeat from camera"""
    if not validate_camera_credentials(camera_id, request.api_key):
        raise HTTPException(status_code=401, detail="Invalid camera credentials")
    
    if camera_id not in registered_cameras:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    # Update camera status
    registered_cameras[camera_id]['last_heartbeat'] = datetime.now()
    registered_cameras[camera_id]['status'] = request.status
    registered_cameras[camera_id]['frame_count'] = request.frame_count
    
    if request.error_message:
        registered_cameras[camera_id]['last_error'] = request.error_message
        logger.warning(f"Camera {camera_id} error: {request.error_message}")
    
    # Generate commands for camera (if any)
    commands = []
    
    # Example: Adjust frame rate based on system load
    system_metrics = pipeline.system_monitor.get_current_stats()
    if system_metrics.get('cpu_percent', 0) > 80:
        commands.append({
            "type": "adjust_fps",
            "value": 10,
            "reason": "High CPU usage"
        })
    
    # Example: Request specific task
    if camera_id not in camera_sessions:
        camera_sessions[camera_id] = {"last_task": "caption"}
    
    commands.append({
        "type": "set_task",
        "value": camera_sessions[camera_id]["last_task"]
    })
    
    return HeartbeatResponse(
        status="ok",
        message="Heartbeat received",
        commands=commands
    )


@app.post("/cameras/{camera_id}/upload", response_model=FrameUploadResponse)
async def upload_frame(camera_id: str, request: FrameUploadRequest):
    """Upload and process a frame from camera"""
    if not validate_camera_credentials(camera_id, request.api_key):
        raise HTTPException(status_code=401, detail="Invalid camera credentials")
    
    if camera_id not in registered_cameras:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    start_time = time.time()
    
    try:
        # Decode base64 image
        import base64
        from PIL import Image
        import io
        
        image_data = base64.b64decode(request.frame_data)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to numpy array
        import numpy as np
        frame_array = np.array(image)
        
        # Process frame
        if request.task == "caption":
            result = pipeline.generate_caption(
                image_input=frame_array,
                max_length=128
            )
        elif request.task == "vqa":
            question = request.question or "What is in this image?"
            result = pipeline.answer_question(
                image_input=frame_array,
                question=question,
                max_length=64
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid task")
        
        processing_time = time.time() - start_time
        
        # Update camera session
        if camera_id not in camera_sessions:
            camera_sessions[camera_id] = {}
        
        camera_sessions[camera_id].update({
            "last_task": request.task,
            "last_processed": datetime.now(),
            "total_frames": camera_sessions[camera_id].get("total_frames", 0) + 1
        })
        
        # Update camera registration
        registered_cameras[camera_id]['last_heartbeat'] = datetime.now()
        registered_cameras[camera_id]['status'] = 'active'
        
        return FrameUploadResponse(
            camera_id=camera_id,
            task=request.task,
            result=result,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to process frame from camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Frame processing failed: {str(e)}")


@app.get("/cameras/{camera_id}/status", response_model=CameraStatusResponse)
async def get_camera_status(camera_id: str, api_key: str = None):
    """Get camera status"""
    if api_key and not validate_camera_credentials(camera_id, api_key):
        raise HTTPException(status_code=401, detail="Invalid camera credentials")
    
    if camera_id not in registered_cameras:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    camera_data = registered_cameras[camera_id]
    session_data = camera_sessions.get(camera_id, {})
    
    # Calculate uptime
    uptime = 0
    if camera_data['registered_at']:
        uptime = (datetime.now() - camera_data['registered_at']).total_seconds()
    
    return CameraStatusResponse(
        camera_id=camera_id,
        name=camera_data['name'],
        location=camera_data['location'],
        status=camera_data['status'],
        registered_at=camera_data['registered_at'].isoformat(),
        last_heartbeat=camera_data['last_heartbeat'].isoformat() if camera_data['last_heartbeat'] else None,
        frame_count=camera_data.get('frame_count', 0),
        uptime_seconds=uptime
    )


@app.get("/cameras", response_model=Dict[str, CameraStatusResponse])
async def list_cameras():
    """List all registered cameras"""
    cameras = {}
    
    for camera_id, camera_data in registered_cameras.items():
        session_data = camera_sessions.get(camera_id, {})
        
        uptime = 0
        if camera_data['registered_at']:
            uptime = (datetime.now() - camera_data['registered_at']).total_seconds()
        
        cameras[camera_id] = CameraStatusResponse(
            camera_id=camera_id,
            name=camera_data['name'],
            location=camera_data['location'],
            status=camera_data['status'],
            registered_at=camera_data['registered_at'].isoformat(),
            last_heartbeat=camera_data['last_heartbeat'].isoformat() if camera_data['last_heartbeat'] else None,
            frame_count=camera_data.get('frame_count', 0),
            uptime_seconds=uptime
        )
    
    return cameras


@app.post("/cameras/{camera_id}/start")
async def start_camera_processing(camera_id: str, api_key: str = None):
    """Start camera processing (admin only)"""
    if api_key and not validate_camera_credentials(camera_id, api_key):
        raise HTTPException(status_code=401, detail="Invalid camera credentials")
    
    if camera_id not in registered_cameras:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    success = camera_processor.start_camera(camera_id)
    
    if success:
        registered_cameras[camera_id]['status'] = 'active'
        return {"status": "success", "message": f"Camera {camera_id} started"}
    else:
        return {"status": "error", "message": f"Failed to start camera {camera_id}"}


@app.post("/cameras/{camera_id}/stop")
async def stop_camera_processing(camera_id: str, api_key: str = None):
    """Stop camera processing"""
    if api_key and not validate_camera_credentials(camera_id, api_key):
        raise HTTPException(status_code=401, detail="Invalid camera credentials")
    
    if camera_id not in registered_cameras:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    camera_processor.stop_camera(camera_id)
    registered_cameras[camera_id]['status'] = 'inactive'
    
    return {"status": "success", "message": f"Camera {camera_id} stopped"}


# Main entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EdgeVLM Camera Registration API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8002, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "api_camera_registration:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )
