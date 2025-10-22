"""
Vision Processing Pipeline
Multi-threaded OpenCV-based image preprocessing for ARM optimization
"""

import cv2
import numpy as np
import logging
import time
from typing import Tuple, Optional, Dict, Any, Union
from pathlib import Path
from PIL import Image
import concurrent.futures


class ARMOptimizedVisionProcessor:
    """
    ARM-optimized vision preprocessing pipeline
    Uses multi-threaded OpenCV with NEON SIMD optimizations
    """
    
    def __init__(
        self,
        input_size: Tuple[int, int] = (336, 336),
        num_threads: int = 2,
        resize_method: str = "bilinear",
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            input_size: Target image size (H, W)
            num_threads: Number of threads for parallel processing
            resize_method: "bilinear", "bicubic", "nearest"
            normalize: Apply ImageNet normalization
            mean: Normalization mean
            std: Normalization std
        """
        self.input_size = input_size
        self.num_threads = num_threads
        self.resize_method = resize_method
        self.normalize = normalize
        self.mean = np.array(mean).reshape(1, 1, 3)
        self.std = np.array(std).reshape(1, 1, 3)
        self.logger = logger or logging.getLogger(__name__)
        
        # Set OpenCV thread count for ARM optimization
        cv2.setNumThreads(num_threads)
        
        # Map resize method
        self.cv2_interpolation = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
            "lanczos": cv2.INTER_LANCZOS4
        }.get(resize_method, cv2.INTER_LINEAR)
        
        self.logger.info(
            f"Vision processor initialized: size={input_size}, "
            f"threads={num_threads}, method={resize_method}"
        )
        
        self.preprocessing_times = []
    
    def process_image(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        return_tensor: bool = False
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Process image through the vision pipeline
        
        Args:
            image_input: Image file path, numpy array, or PIL Image
            return_tensor: If True, return dict with tensor and metadata
            
        Returns:
            Processed image array or dict with metadata
        """
        start_time = time.time()
        
        # Load image
        image = self._load_image(image_input)
        original_shape = image.shape[:2]
        
        # Resize
        resized = self._resize_image(image)
        
        # Normalize
        if self.normalize:
            normalized = self._normalize_image(resized)
        else:
            normalized = resized.astype(np.float32) / 255.0
        
        # Convert to CHW format (channels first)
        chw_image = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension
        batched = np.expand_dims(chw_image, axis=0)
        
        processing_time = time.time() - start_time
        self.preprocessing_times.append(processing_time)
        
        if return_tensor:
            return {
                'tensor': batched,
                'original_shape': original_shape,
                'processed_shape': self.input_size,
                'preprocessing_time': processing_time
            }
        
        return batched
    
    def _load_image(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image]
    ) -> np.ndarray:
        """Load image from various input types"""
        if isinstance(image_input, (str, Path)):
            # Load from file
            image = cv2.imread(str(image_input))
            if image is None:
                raise ValueError(f"Failed to load image from {image_input}")
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        elif isinstance(image_input, Image.Image):
            # Convert PIL to numpy
            image = np.array(image_input)
            if len(image.shape) == 2:
                # Grayscale to RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        elif isinstance(image_input, np.ndarray):
            image = image_input
            # Ensure RGB
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")
        
        return image
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image using OpenCV with ARM optimization"""
        if image.shape[:2] == self.input_size:
            return image
        
        # Use multi-threaded resize (OpenCV automatically uses NEON on ARM)
        resized = cv2.resize(
            image,
            (self.input_size[1], self.input_size[0]),  # (width, height)
            interpolation=self.cv2_interpolation
        )
        
        return resized
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Apply ImageNet normalization"""
        # Convert to float32 and scale to [0, 1]
        normalized = image.astype(np.float32) / 255.0
        
        # Apply mean and std
        normalized = (normalized - self.mean) / self.std
        
        return normalized
    
    def process_batch(
        self,
        image_inputs: list,
        max_workers: Optional[int] = None
    ) -> np.ndarray:
        """
        Process multiple images in parallel
        
        Args:
            image_inputs: List of image inputs
            max_workers: Number of parallel workers (default: num_threads)
            
        Returns:
            Batched numpy array of processed images
        """
        if max_workers is None:
            max_workers = self.num_threads
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            processed_images = list(executor.map(self.process_image, image_inputs))
        
        # Stack into batch
        batch = np.concatenate(processed_images, axis=0)
        
        batch_time = time.time() - start_time
        self.logger.info(
            f"Processed batch of {len(image_inputs)} images in {batch_time:.3f}s"
        )
        
        return batch
    
    def preprocess_for_caption(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Preprocess image specifically for captioning task
        
        Returns:
            Processed image and metadata dict
        """
        result = self.process_image(image_input, return_tensor=True)
        
        metadata = {
            'task': 'caption',
            'original_shape': result['original_shape'],
            'processed_shape': result['processed_shape'],
            'preprocessing_time': result['preprocessing_time']
        }
        
        return result['tensor'], metadata
    
    def preprocess_for_vqa(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        question: str
    ) -> Tuple[np.ndarray, str, Dict[str, Any]]:
        """
        Preprocess image and question for VQA task
        
        Returns:
            Processed image, formatted question, and metadata
        """
        result = self.process_image(image_input, return_tensor=True)
        
        # Format question for VLM
        formatted_question = f"Question: {question}\nAnswer:"
        
        metadata = {
            'task': 'vqa',
            'question': question,
            'original_shape': result['original_shape'],
            'processed_shape': result['processed_shape'],
            'preprocessing_time': result['preprocessing_time']
        }
        
        return result['tensor'], formatted_question, metadata
    
    def get_stats(self) -> Dict[str, Any]:
        """Get preprocessing statistics"""
        if not self.preprocessing_times:
            return {
                'total_images': 0,
                'avg_time': 0.0,
                'min_time': 0.0,
                'max_time': 0.0
            }
        
        return {
            'total_images': len(self.preprocessing_times),
            'avg_time': np.mean(self.preprocessing_times),
            'min_time': np.min(self.preprocessing_times),
            'max_time': np.max(self.preprocessing_times),
            'std_time': np.std(self.preprocessing_times)
        }
    
    def clear_stats(self):
        """Clear preprocessing statistics"""
        self.preprocessing_times = []


class CameraCapture:
    """
    Real-time camera capture for Raspberry Pi
    Optimized for Raspberry Pi Camera Module or USB cameras
    """
    
    def __init__(
        self,
        camera_id: int = 0,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30,
        logger: Optional[logging.Logger] = None
    ):
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps
        self.logger = logger or logging.getLogger(__name__)
        
        self.capture = None
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Initialize camera capture"""
        self.capture = cv2.VideoCapture(self.camera_id)
        
        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")
        
        # Set camera properties
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)
        
        self.logger.info(
            f"Camera initialized: ID={self.camera_id}, "
            f"resolution={self.resolution}, fps={self.fps}"
        )
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame"""
        if self.capture is None:
            return None
        
        ret, frame = self.capture.read()
        
        if not ret:
            self.logger.warning("Failed to capture frame")
            return None
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return frame_rgb
    
    def release(self):
        """Release camera resources"""
        if self.capture is not None:
            self.capture.release()
            self.logger.info("Camera released")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.release()

