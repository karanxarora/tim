"""
Metrics and Monitoring System
Tracks latency, memory usage, early exits, cache performance
"""

import time
import psutil
import logging
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np


@dataclass
class InferenceRecord:
    """Single inference record"""
    timestamp: float
    latency: float
    tokens_generated: int
    tokens_per_second: float
    memory_used_mb: float
    early_exit: bool
    exit_layer: Optional[int]
    cache_size_mb: float


class InferenceMetrics:
    """
    Tracks metrics for standard inference
    """
    
    def __init__(self):
        self.records: List[InferenceRecord] = []
        self.current_start_time: Optional[float] = None
        self.process = psutil.Process()
        
    def start_inference(self):
        """Mark start of inference"""
        self.current_start_time = time.time()
    
    def end_inference(
        self,
        latency: float,
        tokens_generated: int,
        early_exit: bool = False,
        exit_layer: Optional[int] = None
    ):
        """Record completed inference"""
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        
        tokens_per_second = tokens_generated / latency if latency > 0 else 0
        
        record = InferenceRecord(
            timestamp=time.time(),
            latency=latency,
            tokens_generated=tokens_generated,
            tokens_per_second=tokens_per_second,
            memory_used_mb=memory_mb,
            early_exit=early_exit,
            exit_layer=exit_layer,
            cache_size_mb=0.0  # Placeholder
        )
        
        self.records.append(record)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        if not self.records:
            return {
                'total_inferences': 0,
                'avg_latency': 0.0,
                'avg_tokens_per_second': 0.0,
                'early_exit_rate': 0.0
            }
        
        latencies = [r.latency for r in self.records]
        tps = [r.tokens_per_second for r in self.records]
        early_exits = sum(1 for r in self.records if r.early_exit)
        
        return {
            'total_inferences': len(self.records),
            'avg_latency': np.mean(latencies),
            'min_latency': np.min(latencies),
            'max_latency': np.max(latencies),
            'p50_latency': np.percentile(latencies, 50),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99),
            'avg_tokens_per_second': np.mean(tps),
            'early_exit_count': early_exits,
            'early_exit_rate': early_exits / len(self.records),
            'avg_memory_mb': np.mean([r.memory_used_mb for r in self.records]),
            'max_memory_mb': np.max([r.memory_used_mb for r in self.records])
        }
    
    def get_recent_records(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get N most recent records"""
        recent = self.records[-n:]
        return [asdict(r) for r in recent]
    
    def clear(self):
        """Clear all records"""
        self.records = []


class SpeculativeMetrics:
    """
    Tracks metrics for speculative decoding
    """
    
    def __init__(self):
        self.total_tokens = 0
        self.accepted_tokens = 0
        self.rejected_tokens = 0
        self.acceptance_rates: List[float] = []
        self.speedups: List[float] = []
        self.inference_count = 0
    
    def record_inference(
        self,
        total_tokens: int,
        accepted_tokens: int,
        acceptance_rate: float,
        speedup: float
    ):
        """Record a speculative decoding inference"""
        self.inference_count += 1
        self.total_tokens += total_tokens
        self.accepted_tokens += accepted_tokens
        self.rejected_tokens += (total_tokens - accepted_tokens)
        self.acceptance_rates.append(acceptance_rate)
        self.speedups.append(speedup)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        if not self.acceptance_rates:
            return {
                'inference_count': 0,
                'avg_acceptance_rate': 0.0,
                'avg_speedup': 0.0
            }
        
        return {
            'inference_count': self.inference_count,
            'total_tokens': self.total_tokens,
            'accepted_tokens': self.accepted_tokens,
            'rejected_tokens': self.rejected_tokens,
            'avg_acceptance_rate': np.mean(self.acceptance_rates),
            'min_acceptance_rate': np.min(self.acceptance_rates),
            'max_acceptance_rate': np.max(self.acceptance_rates),
            'avg_speedup': np.mean(self.speedups),
            'theoretical_speedup': np.mean(self.speedups)
        }
    
    def clear(self):
        """Clear metrics"""
        self.total_tokens = 0
        self.accepted_tokens = 0
        self.rejected_tokens = 0
        self.acceptance_rates = []
        self.speedups = []
        self.inference_count = 0


class SystemMonitor:
    """
    Monitors system resources (CPU, memory, temperature)
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.process = psutil.Process()
        self.warning_temp = 70.0  # Celsius
        self.samples: List[Dict[str, Any]] = []
    
    def sample(self) -> Dict[str, Any]:
        """Take a system resource sample"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        process_memory = self.process.memory_info()
        
        # Temperature (Raspberry Pi specific)
        temperature = self._get_temperature()
        
        sample = {
            'timestamp': time.time(),
            'cpu_percent': cpu_percent,
            'memory_total_mb': memory.total / (1024 * 1024),
            'memory_used_mb': memory.used / (1024 * 1024),
            'memory_available_mb': memory.available / (1024 * 1024),
            'memory_percent': memory.percent,
            'process_memory_mb': process_memory.rss / (1024 * 1024),
            'temperature_celsius': temperature
        }
        
        self.samples.append(sample)
        
        # Check for warnings
        if temperature and temperature > self.warning_temp:
            self.logger.warning(
                f"High temperature detected: {temperature:.1f}°C "
                f"(threshold: {self.warning_temp}°C)"
            )
        
        return sample
    
    def _get_temperature(self) -> Optional[float]:
        """Get CPU temperature (Raspberry Pi)"""
        try:
            # Try Raspberry Pi method
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = float(f.read().strip()) / 1000.0
                return temp
        except:
            # Temperature not available
            return None
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current system statistics"""
        return self.sample()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of collected samples"""
        if not self.samples:
            return {}
        
        cpu_vals = [s['cpu_percent'] for s in self.samples]
        mem_vals = [s['memory_percent'] for s in self.samples]
        temp_vals = [s['temperature_celsius'] for s in self.samples if s['temperature_celsius']]
        
        summary = {
            'sample_count': len(self.samples),
            'avg_cpu_percent': np.mean(cpu_vals),
            'max_cpu_percent': np.max(cpu_vals),
            'avg_memory_percent': np.mean(mem_vals),
            'max_memory_percent': np.max(mem_vals),
            'avg_process_memory_mb': np.mean([s['process_memory_mb'] for s in self.samples]),
            'max_process_memory_mb': np.max([s['process_memory_mb'] for s in self.samples])
        }
        
        if temp_vals:
            summary['avg_temperature_celsius'] = np.mean(temp_vals)
            summary['max_temperature_celsius'] = np.max(temp_vals)
        
        return summary
    
    def clear(self):
        """Clear samples"""
        self.samples = []


class BenchmarkLogger:
    """
    Comprehensive benchmark logging to file
    """
    
    def __init__(
        self,
        output_file: str = "benchmarks/results.json",
        logger: Optional[logging.Logger] = None
    ):
        self.output_file = output_file
        self.logger = logger or logging.getLogger(__name__)
        self.benchmark_data: Dict[str, Any] = {
            'created_at': datetime.now().isoformat(),
            'device': self._get_device_info(),
            'runs': []
        }
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get device information"""
        import platform
        
        return {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'machine': platform.machine(),
            'cpu_count': psutil.cpu_count(),
            'total_memory_mb': psutil.virtual_memory().total / (1024 * 1024)
        }
    
    def add_run(
        self,
        task_type: str,
        inference_metrics: Dict[str, Any],
        system_metrics: Dict[str, Any],
        config: Dict[str, Any],
        additional_info: Optional[Dict[str, Any]] = None
    ):
        """Add a benchmark run"""
        run_data = {
            'timestamp': datetime.now().isoformat(),
            'task_type': task_type,
            'inference_metrics': inference_metrics,
            'system_metrics': system_metrics,
            'config': config
        }
        
        if additional_info:
            run_data['additional_info'] = additional_info
        
        self.benchmark_data['runs'].append(run_data)
        
        # Save to file
        self._save()
    
    def _save(self):
        """Save benchmark data to JSON file"""
        try:
            import os
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            
            with open(self.output_file, 'w') as f:
                json.dump(self.benchmark_data, f, indent=2)
            
            self.logger.info(f"Benchmark data saved to {self.output_file}")
        except Exception as e:
            self.logger.error(f"Failed to save benchmark data: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all runs"""
        if not self.benchmark_data['runs']:
            return {'total_runs': 0}
        
        runs = self.benchmark_data['runs']
        
        latencies = []
        memory_usage = []
        
        for run in runs:
            if 'inference_metrics' in run:
                metrics = run['inference_metrics']
                if 'avg_latency' in metrics:
                    latencies.append(metrics['avg_latency'])
                if 'avg_memory_mb' in metrics:
                    memory_usage.append(metrics['avg_memory_mb'])
        
        summary = {
            'total_runs': len(runs),
            'device': self.benchmark_data['device']
        }
        
        if latencies:
            summary['overall_avg_latency'] = np.mean(latencies)
            summary['overall_min_latency'] = np.min(latencies)
            summary['overall_max_latency'] = np.max(latencies)
        
        if memory_usage:
            summary['overall_avg_memory_mb'] = np.mean(memory_usage)
            summary['overall_max_memory_mb'] = np.max(memory_usage)
        
        return summary

