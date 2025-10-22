"""
KV Cache Compression Implementations
- GEAR: Grouped Early-exit Attention Reduction
- Pyramid: Layered cache size reduction
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class CacheEntry:
    """Represents a KV cache entry"""
    key: np.ndarray
    value: np.ndarray
    layer: int
    timestamp: int
    attention_score: float
    access_count: int = 0


class GEARCache:
    """
    GEAR (Grouped Early-exit Attention Reduction) Cache
    
    Implements attention-based KV cache eviction:
    - Tracks attention scores for each cache entry
    - Evicts low-attention entries to reduce memory
    - Maintains quality while compressing cache
    
    Reference: "GEAR: An Efficient KV Cache Compression Recipe" (2024)
    """
    
    def __init__(
        self,
        compression_ratio: float = 0.5,
        eviction_policy: str = "attention_score",
        min_cache_size: int = 32,
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            compression_ratio: Target compression (0.5 = keep 50% of cache)
            eviction_policy: "attention_score" or "layer_recency"
            min_cache_size: Minimum cache entries to maintain
        """
        self.compression_ratio = compression_ratio
        self.eviction_policy = eviction_policy
        self.min_cache_size = min_cache_size
        self.logger = logger or logging.getLogger(__name__)
        
        self.cache: Dict[int, List[CacheEntry]] = {}  # layer -> entries
        self.global_timestamp = 0
        self.total_evictions = 0
        
        self.logger.info(
            f"GEAR cache initialized: compression={compression_ratio}, "
            f"policy={eviction_policy}"
        )
    
    def add_entry(
        self,
        layer: int,
        key: np.ndarray,
        value: np.ndarray,
        attention_score: float = 1.0
    ):
        """Add a new cache entry"""
        if layer not in self.cache:
            self.cache[layer] = []
        
        entry = CacheEntry(
            key=key,
            value=value,
            layer=layer,
            timestamp=self.global_timestamp,
            attention_score=attention_score
        )
        
        self.cache[layer].append(entry)
        self.global_timestamp += 1
        
        # Apply compression if cache exceeds target size
        target_size = int(len(self.cache[layer]) * self.compression_ratio)
        if len(self.cache[layer]) > max(target_size, self.min_cache_size):
            self._compress_layer(layer)
    
    def _compress_layer(self, layer: int):
        """Compress cache for a specific layer"""
        entries = self.cache[layer]
        target_size = max(
            int(len(entries) * self.compression_ratio),
            self.min_cache_size
        )
        
        if len(entries) <= target_size:
            return
        
        # Sort by eviction policy
        if self.eviction_policy == "attention_score":
            # Keep high-attention entries
            sorted_entries = sorted(
                entries,
                key=lambda e: e.attention_score,
                reverse=True
            )
        elif self.eviction_policy == "layer_recency":
            # Keep recent entries
            sorted_entries = sorted(
                entries,
                key=lambda e: e.timestamp,
                reverse=True
            )
        else:
            sorted_entries = entries
        
        # Keep top entries
        kept_entries = sorted_entries[:target_size]
        evicted_count = len(entries) - len(kept_entries)
        
        self.cache[layer] = kept_entries
        self.total_evictions += evicted_count
        
        self.logger.debug(
            f"Compressed layer {layer}: {len(entries)} -> {len(kept_entries)} "
            f"(evicted {evicted_count})"
        )
    
    def get_layer_cache(self, layer: int) -> List[CacheEntry]:
        """Get cache entries for a layer"""
        return self.cache.get(layer, [])
    
    def clear_layer(self, layer: int):
        """Clear cache for a specific layer"""
        if layer in self.cache:
            self.cache[layer] = []
    
    def clear_all(self):
        """Clear entire cache"""
        self.cache = {}
        self.global_timestamp = 0
        self.logger.info("GEAR cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_entries = sum(len(entries) for entries in self.cache.values())
        
        return {
            'total_entries': total_entries,
            'total_evictions': self.total_evictions,
            'layers_cached': len(self.cache),
            'compression_ratio': self.compression_ratio,
            'eviction_policy': self.eviction_policy
        }


class PyramidCache:
    """
    Pyramid KV Cache Compression
    
    Different compression ratios for different layers:
    - Early layers: High retention (important for understanding)
    - Middle layers: Moderate retention
    - Late layers: Low retention (less critical for output)
    
    Reference: "Pyramid-KV: Dynamic KV Cache Compression" (2024)
    """
    
    def __init__(
        self,
        layer_ratios: Optional[List[float]] = None,
        num_layers: int = 24,
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            layer_ratios: Retention ratios for layer groups (e.g., [1.0, 0.9, 0.7, 0.5])
            num_layers: Total number of transformer layers
        """
        self.num_layers = num_layers
        self.logger = logger or logging.getLogger(__name__)
        
        # Default pyramid: keep more in early layers
        if layer_ratios is None:
            layer_ratios = [1.0, 0.9, 0.7, 0.5, 0.3]
        
        self.layer_ratios = layer_ratios
        self.layer_compression = self._compute_layer_compression(num_layers, layer_ratios)
        
        self.cache: Dict[int, List[CacheEntry]] = {}
        self.global_timestamp = 0
        
        self.logger.info(f"Pyramid cache initialized with {num_layers} layers")
    
    def _compute_layer_compression(
        self,
        num_layers: int,
        ratios: List[float]
    ) -> Dict[int, float]:
        """Compute compression ratio for each layer"""
        compression = {}
        layers_per_group = num_layers / len(ratios)
        
        for i in range(num_layers):
            group_idx = min(int(i / layers_per_group), len(ratios) - 1)
            compression[i] = ratios[group_idx]
        
        return compression
    
    def add_entry(
        self,
        layer: int,
        key: np.ndarray,
        value: np.ndarray,
        attention_score: float = 1.0
    ):
        """Add cache entry with layer-specific compression"""
        if layer not in self.cache:
            self.cache[layer] = []
        
        entry = CacheEntry(
            key=key,
            value=value,
            layer=layer,
            timestamp=self.global_timestamp,
            attention_score=attention_score
        )
        
        self.cache[layer].append(entry)
        self.global_timestamp += 1
        
        # Apply pyramid compression
        compression_ratio = self.layer_compression.get(layer, 0.5)
        target_size = max(int(32 * compression_ratio), 8)  # Min 8 entries
        
        if len(self.cache[layer]) > target_size:
            self._compress_layer(layer, target_size)
    
    def _compress_layer(self, layer: int, target_size: int):
        """Compress layer to target size using attention scores"""
        entries = self.cache[layer]
        
        if len(entries) <= target_size:
            return
        
        # Keep high-attention entries
        sorted_entries = sorted(
            entries,
            key=lambda e: e.attention_score,
            reverse=True
        )
        
        self.cache[layer] = sorted_entries[:target_size]
        
        self.logger.debug(
            f"Pyramid compression layer {layer}: {len(entries)} -> {target_size} "
            f"(ratio: {self.layer_compression[layer]:.2f})"
        )
    
    def get_layer_cache(self, layer: int) -> List[CacheEntry]:
        """Get cache for a layer"""
        return self.cache.get(layer, [])
    
    def clear_all(self):
        """Clear all cache"""
        self.cache = {}
        self.global_timestamp = 0
        self.logger.info("Pyramid cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        layer_sizes = {layer: len(entries) for layer, entries in self.cache.items()}
        total_entries = sum(layer_sizes.values())
        
        return {
            'total_entries': total_entries,
            'layers_cached': len(self.cache),
            'layer_sizes': layer_sizes,
            'compression_ratios': self.layer_compression
        }


class CacheManager:
    """
    Unified cache manager combining GEAR and Pyramid strategies
    """
    
    def __init__(
        self,
        enable_gear: bool = True,
        enable_pyramid: bool = True,
        max_cache_size_mb: int = 512,
        logger: Optional[logging.Logger] = None
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.max_cache_size_mb = max_cache_size_mb
        
        self.gear_cache = GEARCache(logger=logger) if enable_gear else None
        self.pyramid_cache = PyramidCache(logger=logger) if enable_pyramid else None
        
        self.clear_count = 0
        
        self.logger.info(
            f"Cache manager initialized (GEAR: {enable_gear}, Pyramid: {enable_pyramid})"
        )
    
    def add_entry(
        self,
        layer: int,
        key: np.ndarray,
        value: np.ndarray,
        attention_score: float = 1.0
    ):
        """Add entry to active caches"""
        if self.gear_cache:
            self.gear_cache.add_entry(layer, key, value, attention_score)
        
        if self.pyramid_cache:
            self.pyramid_cache.add_entry(layer, key, value, attention_score)
    
    def clear_all(self):
        """Clear all caches"""
        if self.gear_cache:
            self.gear_cache.clear_all()
        if self.pyramid_cache:
            self.pyramid_cache.clear_all()
        
        self.clear_count += 1
        self.logger.info(f"All caches cleared (count: {self.clear_count})")
    
    def get_combined_stats(self) -> Dict[str, Any]:
        """Get combined cache statistics"""
        stats = {
            'clear_count': self.clear_count,
            'max_cache_size_mb': self.max_cache_size_mb
        }
        
        if self.gear_cache:
            stats['gear'] = self.gear_cache.get_stats()
        
        if self.pyramid_cache:
            stats['pyramid'] = self.pyramid_cache.get_stats()
        
        return stats

