"""
EdgeVLM REST API
FastAPI-based interface for real-time image captioning and VQA
"""

import os
import logging
import asyncio
from typing import Optional
from datetime import datetime
import base64
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from pipeline import EdgeVLMPipeline
from PIL import Image
import numpy as np


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EdgeVLM-API")

# Initialize FastAPI app
app = FastAPI(
    title="EdgeVLM API",
    description="Real-time Vision-Language API for Raspberry Pi",
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

# Global pipeline instance
pipeline: Optional[EdgeVLMPipeline] = None


# Request/Response Models
class CaptionRequest(BaseModel):
    max_length: int = Field(default=128, ge=1, le=256)


class VQARequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    max_length: int = Field(default=64, ge=1, le=128)


class CaptionResponse(BaseModel):
    caption: str
    latency: float
    preprocessing_time: float
    inference_time: float
    tokens_per_second: float
    early_exit: bool
    exit_layer: Optional[int]
    timestamp: str


class VQAResponse(BaseModel):
    question: str
    answer: str
    latency: float
    preprocessing_time: float
    inference_time: float
    tokens_per_second: float
    early_exit: bool
    exit_layer: Optional[int]
    timestamp: str


class MetricsResponse(BaseModel):
    inference_metrics: dict
    vision_metrics: dict
    system_metrics: dict
    cache_metrics: dict


class HealthResponse(BaseModel):
    status: str
    pipeline_loaded: bool
    timestamp: str
    system_info: dict


# Startup/Shutdown Events
@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup"""
    global pipeline
    logger.info("Starting EdgeVLM API...")
    
    try:
        pipeline = EdgeVLMPipeline(
            config_path="config.yaml",
            logger=logger
        )
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global pipeline
    logger.info("Shutting down EdgeVLM API...")
    
    if pipeline:
        pipeline.shutdown()
    
    logger.info("Shutdown complete")


# Helper Functions
async def save_upload_file(upload_file: UploadFile) -> str:
    """Save uploaded file temporarily"""
    import tempfile
    
    # Create temporary file
    suffix = os.path.splitext(upload_file.filename)[1]
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    
    try:
        contents = await upload_file.read()
        temp_file.write(contents)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        logger.error(f"Failed to save upload file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save file")


def cleanup_temp_file(filepath: str):
    """Clean up temporary file"""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file {filepath}: {e}")


# API Endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "service": "EdgeVLM API",
        "version": "1.0.0",
        "description": "Real-time Vision-Language Model for Edge Devices",
        "endpoints": {
            "caption": "/caption - Generate image captions",
            "vqa": "/vqa - Visual Question Answering",
            "metrics": "/metrics - Get performance metrics",
            "health": "/health - Health check"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    # Get system metrics
    system_metrics = pipeline.system_monitor.get_current_stats()
    
    return HealthResponse(
        status="healthy",
        pipeline_loaded=True,
        timestamp=datetime.now().isoformat(),
        system_info={
            "cpu_percent": system_metrics.get("cpu_percent", 0),
            "memory_percent": system_metrics.get("memory_percent", 0),
            "temperature_celsius": system_metrics.get("temperature_celsius")
        }
    )


@app.post("/caption", response_model=CaptionResponse)
async def generate_caption(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    max_length: int = Form(default=128)
):
    """
    Generate caption for uploaded image
    
    Args:
        image: Uploaded image file
        max_length: Maximum caption length in tokens
        
    Returns:
        Caption with performance metrics
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    # Validate image file
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save uploaded file
    temp_path = await save_upload_file(image)
    
    # Schedule cleanup
    background_tasks.add_task(cleanup_temp_file, temp_path)
    
    try:
        # Generate caption
        result = pipeline.generate_caption(
            image_input=temp_path,
            max_length=max_length
        )
        
        return CaptionResponse(
            caption=result['caption'],
            latency=result['latency'],
            preprocessing_time=result['preprocessing_time'],
            inference_time=result['inference_time'],
            tokens_per_second=result['tokens_per_second'],
            early_exit=result['early_exit'],
            exit_layer=result.get('exit_layer'),
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Caption generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Caption generation failed: {str(e)}")


@app.post("/vqa", response_model=VQAResponse)
async def visual_question_answering(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    question: str = Form(...),
    max_length: int = Form(default=64)
):
    """
    Answer question about uploaded image
    
    Args:
        image: Uploaded image file
        question: Question to answer
        max_length: Maximum answer length in tokens
        
    Returns:
        Answer with performance metrics
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    # Validate image
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Validate question
    if not question or len(question.strip()) == 0:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Save uploaded file
    temp_path = await save_upload_file(image)
    
    # Schedule cleanup
    background_tasks.add_task(cleanup_temp_file, temp_path)
    
    try:
        # Answer question
        result = pipeline.answer_question(
            image_input=temp_path,
            question=question,
            max_length=max_length
        )
        
        return VQAResponse(
            question=result['question'],
            answer=result['answer'],
            latency=result['latency'],
            preprocessing_time=result['preprocessing_time'],
            inference_time=result['inference_time'],
            tokens_per_second=result['tokens_per_second'],
            early_exit=result['early_exit'],
            exit_layer=result.get('exit_layer'),
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"VQA failed: {e}")
        raise HTTPException(status_code=500, detail=f"VQA failed: {str(e)}")


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get performance metrics
    
    Returns:
        Comprehensive performance metrics
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        metrics = pipeline.get_metrics()
        
        return MetricsResponse(
            inference_metrics=metrics['inference'],
            vision_metrics=metrics['vision'],
            system_metrics=metrics['system'],
            cache_metrics=metrics['cache']
        )
    
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@app.post("/clear-cache")
async def clear_cache():
    """Clear all caches"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        pipeline.clear_cache()
        return {"status": "success", "message": "Cache cleared"}
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@app.post("/benchmark")
async def run_benchmark(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    task_type: str = Form(default="caption"),
    question: Optional[str] = Form(default=None),
    num_runs: int = Form(default=10)
):
    """
    Run performance benchmark
    
    Args:
        image: Image for benchmarking
        task_type: "caption" or "vqa"
        question: Question for VQA benchmark
        num_runs: Number of iterations
        
    Returns:
        Benchmark results
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    if task_type not in ["caption", "vqa"]:
        raise HTTPException(status_code=400, detail="task_type must be 'caption' or 'vqa'")
    
    # Save uploaded file
    temp_path = await save_upload_file(image)
    
    # Schedule cleanup
    background_tasks.add_task(cleanup_temp_file, temp_path)
    
    try:
        results = pipeline.benchmark_run(
            task_type=task_type,
            image_input=temp_path,
            question=question,
            num_runs=num_runs
        )
        
        return JSONResponse(content=results)
    
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")


# Main entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EdgeVLM API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )

