"""
EdgeVLM Core Components
"""

from core.inference_engine import ARMOptimizedInferenceEngine, InferenceConfig
from core.speculative_decoding import SpeculativeDecoder
from core.vision_processor import ARMOptimizedVisionProcessor, CameraCapture
from core.kv_cache import GEARCache, PyramidCache, CacheManager
from core.metrics import (
    InferenceMetrics,
    SpeculativeMetrics,
    SystemMonitor,
    BenchmarkLogger
)

__all__ = [
    'ARMOptimizedInferenceEngine',
    'InferenceConfig',
    'SpeculativeDecoder',
    'ARMOptimizedVisionProcessor',
    'CameraCapture',
    'GEARCache',
    'PyramidCache',
    'CacheManager',
    'InferenceMetrics',
    'SpeculativeMetrics',
    'SystemMonitor',
    'BenchmarkLogger'
]

