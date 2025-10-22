"""
EdgeVLM Main Pipeline
Integrates all components into a unified vision-language system
"""

import os
import logging
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
import time

from core import (
    ARMOptimizedInferenceEngine,
    InferenceConfig,
    SpeculativeDecoder,
    ARMOptimizedVisionProcessor,
    CameraCapture,
    CacheManager,
    SystemMonitor,
    BenchmarkLogger
)


class EdgeVLMPipeline:
    """
    End-to-end vision-language pipeline for edge devices
    Supports image captioning and visual question answering
    """
    
    def __init__(
        self,
        config_path: str = "config.yaml",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize EdgeVLM pipeline
        
        Args:
            config_path: Path to configuration YAML
            logger: Optional logger instance
        """
        self.logger = logger or self._setup_logger()
        self.logger.info("Initializing EdgeVLM Pipeline...")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize system monitor
        self.system_monitor = SystemMonitor(logger=self.logger)
        
        # Initialize vision processor
        self.vision_processor = self._initialize_vision_processor()
        
        # Initialize inference engine(s)
        self.use_speculative = self.config['optimizations']['speculative_decoding']['enabled']
        
        if self.use_speculative:
            self.logger.info("Initializing with speculative decoding...")
            self.inference_engine = self._initialize_speculative_decoder()
        else:
            self.logger.info("Initializing standard inference engine...")
            self.inference_engine = self._initialize_standard_engine()
        
        # Initialize cache manager
        self.cache_manager = self._initialize_cache_manager()
        
        # Initialize benchmark logger
        benchmark_output = self.config['monitoring']['benchmark_output']
        self.benchmark_logger = BenchmarkLogger(
            output_file=benchmark_output,
            logger=self.logger
        )
        
        self.logger.info("EdgeVLM Pipeline initialized successfully!")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger("EdgeVLM")
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            raise
    
    def _initialize_vision_processor(self) -> ARMOptimizedVisionProcessor:
        """Initialize vision processing pipeline"""
        vision_config = self.config['vision']
        model_config = self.config['models']['vision_encoder']
        
        return ARMOptimizedVisionProcessor(
            input_size=tuple(model_config['input_size']),
            num_threads=vision_config['num_threads'],
            resize_method=vision_config['resize_method'],
            normalize=True,
            mean=tuple(vision_config['normalization']['mean']),
            std=tuple(vision_config['normalization']['std']),
            logger=self.logger
        )
    
    def _initialize_standard_engine(self) -> ARMOptimizedInferenceEngine:
        """Initialize standard inference engine"""
        model_config = self.config['models']['main_vlm']
        inference_config = self.config['inference']
        opt_config = self.config['optimizations']
        
        config = InferenceConfig(
            model_path=model_config['path'],
            n_ctx=2048,
            n_threads=inference_config['num_threads'],
            n_batch=512,
            use_mmap=True,
            f16_kv=inference_config['arm_optimizations']['fp16_inference']
        )
        
        return ARMOptimizedInferenceEngine(
            config=config,
            enable_gear_cache=opt_config['kv_cache_compression']['gear_enabled'],
            enable_pyramid_cache=opt_config['kv_cache_compression']['pyramid_enabled'],
            enable_early_exit=opt_config['early_exit']['enabled'],
            logger=self.logger
        )
    
    def _initialize_speculative_decoder(self) -> SpeculativeDecoder:
        """Initialize speculative decoding engine"""
        main_model = self.config['models']['main_vlm']['path']
        draft_model = self.config['models']['draft_model']['path']
        spec_config = self.config['optimizations']['speculative_decoding']
        inference_config = self.config['inference']
        
        return SpeculativeDecoder(
            verifier_model_path=main_model,
            draft_model_path=draft_model,
            draft_tokens=spec_config['draft_tokens'],
            acceptance_threshold=spec_config['acceptance_threshold'],
            n_threads=inference_config['num_threads'],
            logger=self.logger
        )
    
    def _initialize_cache_manager(self) -> CacheManager:
        """Initialize cache management"""
        cache_config = self.config['optimizations']['kv_cache_compression']
        mem_config = self.config['optimizations']['memory_management']
        
        return CacheManager(
            enable_gear=cache_config['gear_enabled'],
            enable_pyramid=cache_config['pyramid_enabled'],
            max_cache_size_mb=mem_config['max_cache_size_mb'],
            logger=self.logger
        )
    
    def generate_caption(
        self,
        image_input: Union[str, Path],
        max_length: int = 128
    ) -> Dict[str, Any]:
        """
        Generate caption for an image
        
        Args:
            image_input: Path to image file or image array
            max_length: Maximum caption length in tokens
            
        Returns:
            Dictionary with caption and metrics
        """
        start_time = time.time()
        self.logger.info(f"Generating caption for image: {image_input}")
        
        # Sample system state before
        self.system_monitor.sample()
        
        # Preprocess image
        processed_image, vision_metadata = self.vision_processor.preprocess_for_caption(
            image_input
        )
        
        # Create prompt for captioning
        prompt = self._create_caption_prompt()
        
        # Generate caption
        inference_config = self.config['inference']
        result = self.inference_engine.generate(
            prompt=prompt,
            max_tokens=max_length,
            temperature=inference_config['temperature'],
            top_p=inference_config['top_p'],
            repeat_penalty=inference_config['repeat_penalty'],
            stop_sequences=['</s>', '\n\n']
        )
        
        # Sample system state after
        self.system_monitor.sample()
        
        # Extract caption from result
        caption = self._extract_caption(result['text'])
        
        total_time = time.time() - start_time
        
        # Compile response
        response = {
            'caption': caption,
            'latency': total_time,
            'preprocessing_time': vision_metadata['preprocessing_time'],
            'inference_time': result['latency'],
            'tokens_per_second': result['tokens_per_second'],
            'early_exit': result.get('early_exit', False),
            'exit_layer': result.get('exit_layer'),
            'inference_count': result.get('inference_count', 0)
        }
        
        # Add speculative decoding metrics if enabled
        if self.use_speculative:
            response['acceptance_rate'] = result.get('acceptance_rate', 0.0)
            response['speedup_factor'] = result.get('speedup_factor', 1.0)
        
        self.logger.info(f"Caption generated in {total_time:.3f}s: {caption}")
        
        return response
    
    def answer_question(
        self,
        image_input: Union[str, Path],
        question: str,
        max_length: int = 64
    ) -> Dict[str, Any]:
        """
        Answer a question about an image (VQA)
        
        Args:
            image_input: Path to image file
            question: Question to answer
            max_length: Maximum answer length in tokens
            
        Returns:
            Dictionary with answer and metrics
        """
        start_time = time.time()
        self.logger.info(f"Answering question: {question}")
        
        # Sample system state
        self.system_monitor.sample()
        
        # Preprocess image and question
        processed_image, formatted_question, vision_metadata = \
            self.vision_processor.preprocess_for_vqa(image_input, question)
        
        # Create VQA prompt
        prompt = self._create_vqa_prompt(question)
        
        # Generate answer
        inference_config = self.config['inference']
        result = self.inference_engine.generate(
            prompt=prompt,
            max_tokens=max_length,
            temperature=inference_config['temperature'],
            top_p=inference_config['top_p'],
            repeat_penalty=inference_config['repeat_penalty'],
            stop_sequences=['</s>', '\n', '?']
        )
        
        # Sample system state
        self.system_monitor.sample()
        
        # Extract answer
        answer = self._extract_answer(result['text'])
        
        total_time = time.time() - start_time
        
        response = {
            'question': question,
            'answer': answer,
            'latency': total_time,
            'preprocessing_time': vision_metadata['preprocessing_time'],
            'inference_time': result['latency'],
            'tokens_per_second': result['tokens_per_second'],
            'early_exit': result.get('early_exit', False),
            'exit_layer': result.get('exit_layer'),
            'inference_count': result.get('inference_count', 0)
        }
        
        if self.use_speculative:
            response['acceptance_rate'] = result.get('acceptance_rate', 0.0)
            response['speedup_factor'] = result.get('speedup_factor', 1.0)
        
        self.logger.info(f"Answer generated in {total_time:.3f}s: {answer}")
        
        return response
    
    def _create_caption_prompt(self) -> str:
        """Create prompt for image captioning"""
        # This would be model-specific
        # For MobileVLM, use appropriate format
        return "Describe this image in detail:"
    
    def _create_vqa_prompt(self, question: str) -> str:
        """Create prompt for VQA"""
        return f"Question: {question}\nAnswer:"
    
    def _extract_caption(self, raw_text: str) -> str:
        """Extract clean caption from model output"""
        # Clean up the generated text
        caption = raw_text.strip()
        
        # Remove common artifacts
        caption = caption.replace('</s>', '').replace('<s>', '')
        caption = caption.split('\n')[0]  # Take first line
        
        return caption
    
    def _extract_answer(self, raw_text: str) -> str:
        """Extract clean answer from model output"""
        answer = raw_text.strip()
        answer = answer.replace('</s>', '').replace('<s>', '')
        answer = answer.split('\n')[0]
        
        return answer
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        metrics = {
            'inference': self.inference_engine.get_metrics(),
            'vision': self.vision_processor.get_stats(),
            'system': self.system_monitor.get_summary(),
            'cache': self.cache_manager.get_combined_stats()
        }
        
        return metrics
    
    def benchmark_run(
        self,
        task_type: str,
        image_input: Union[str, Path],
        question: Optional[str] = None,
        num_runs: int = 10
    ) -> Dict[str, Any]:
        """
        Run benchmark for performance evaluation
        
        Args:
            task_type: "caption" or "vqa"
            image_input: Image to use for benchmark
            question: Question for VQA task
            num_runs: Number of iterations
            
        Returns:
            Benchmark results
        """
        self.logger.info(f"Running benchmark: {task_type}, {num_runs} iterations")
        
        latencies = []
        
        for i in range(num_runs):
            if task_type == "caption":
                result = self.generate_caption(image_input)
            elif task_type == "vqa":
                if not question:
                    question = "What is in this image?"
                result = self.answer_question(image_input, question)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
            
            latencies.append(result['latency'])
            
            self.logger.info(f"Run {i+1}/{num_runs}: {result['latency']:.3f}s")
        
        # Get comprehensive metrics
        inference_metrics = self.inference_engine.get_metrics()
        system_metrics = self.system_monitor.get_summary()
        
        # Log to benchmark file
        self.benchmark_logger.add_run(
            task_type=task_type,
            inference_metrics=inference_metrics,
            system_metrics=system_metrics,
            config=self.config,
            additional_info={
                'num_runs': num_runs,
                'image': str(image_input),
                'question': question
            }
        )
        
        import numpy as np
        benchmark_results = {
            'task_type': task_type,
            'num_runs': num_runs,
            'avg_latency': np.mean(latencies),
            'min_latency': np.min(latencies),
            'max_latency': np.max(latencies),
            'p50_latency': np.percentile(latencies, 50),
            'p95_latency': np.percentile(latencies, 95),
            'std_latency': np.std(latencies),
            'inference_metrics': inference_metrics,
            'system_metrics': system_metrics
        }
        
        self.logger.info(f"Benchmark complete. Avg latency: {benchmark_results['avg_latency']:.3f}s")
        
        return benchmark_results
    
    def clear_cache(self):
        """Clear all caches"""
        self.inference_engine.reset_cache()
        self.cache_manager.clear_all()
        self.logger.info("All caches cleared")
    
    def shutdown(self):
        """Gracefully shutdown the pipeline"""
        self.logger.info("Shutting down EdgeVLM pipeline...")
        self.clear_cache()
        del self.inference_engine
        self.logger.info("Shutdown complete")

