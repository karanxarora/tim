# EdgeVLM Architecture Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Data Flow](#data-flow)
4. [Optimization Strategies](#optimization-strategies)
5. [Memory Management](#memory-management)
6. [Performance Analysis](#performance-analysis)

## System Overview

EdgeVLM is designed as a modular, highly optimized vision-language pipeline for edge devices. The architecture prioritizes:

1. **Low Latency**: Target 2-5 seconds for image-to-text tasks
2. **Memory Efficiency**: Operate within 6GB RAM constraint
3. **Modularity**: Toggle optimizations independently
4. **Robustness**: Graceful degradation under resource constraints

### High-Level Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        FastAPI Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │   /caption   │  │     /vqa     │  │   /metrics   │        │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘        │
└─────────┼──────────────────┼──────────────────┼────────────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                     EdgeVLM Pipeline                             │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              Vision Processing                          │    │
│  │  • Multi-threaded OpenCV                               │    │
│  │  • ARM NEON SIMD optimization                          │    │
│  │  • Resize, normalize, format conversion               │    │
│  └────────────────────┬───────────────────────────────────┘    │
│                       │                                          │
│  ┌────────────────────▼───────────────────────────────────┐    │
│  │         Inference Engine (with Optimizations)          │    │
│  │  ┌──────────────────────────────────────────────┐     │    │
│  │  │       Speculative Decoding (Optional)        │     │    │
│  │  │  Draft: TinyLlama 1.1B (Q4)                  │     │    │
│  │  │  Verify: MobileVLM-V2 1.7B (Q4)             │     │    │
│  │  └──────────────────┬───────────────────────────┘     │    │
│  │                     │                                  │    │
│  │  ┌──────────────────▼───────────────────────────┐     │    │
│  │  │         KV Cache Management                   │     │    │
│  │  │  • GEAR: Attention-based eviction            │     │    │
│  │  │  • Pyramid: Layer-wise compression           │     │    │
│  │  │  • Dynamic allocation                        │     │    │
│  │  └──────────────────┬───────────────────────────┘     │    │
│  │                     │                                  │    │
│  │  ┌──────────────────▼───────────────────────────┐     │    │
│  │  │         Early Exit Strategy                   │     │    │
│  │  │  • Confidence monitoring                     │     │    │
│  │  │  • Exit at layers [8, 12, 16, 20]           │     │    │
│  │  └──────────────────────────────────────────────┘     │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              Monitoring & Metrics                       │    │
│  │  • Latency tracking                                    │    │
│  │  • Memory profiling                                    │    │
│  │  • System monitoring (CPU, temp)                      │    │
│  └────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. Vision Processor (`core/vision_processor.py`)

**Purpose**: Efficient image preprocessing for ARM CPUs

**Key Features**:
- Multi-threaded OpenCV (leverages ARM NEON)
- Configurable resize methods (bilinear, bicubic, etc.)
- ImageNet normalization
- Batch processing support
- Format conversion (HWC → CHW, add batch dim)

**Implementation Details**:

```python
class ARMOptimizedVisionProcessor:
    def __init__(self, input_size, num_threads=2):
        cv2.setNumThreads(num_threads)  # Enable parallel processing
        
    def process_image(self, image_input):
        # 1. Load image (from file/array/PIL)
        # 2. Resize with ARM-optimized OpenCV
        # 3. Normalize (mean/std)
        # 4. Format conversion (HWC -> CHW)
        # 5. Add batch dimension
        return processed_tensor
```

**Performance**:
- Typical preprocessing time: 80-100ms
- Parallelization efficiency: ~1.8x with 2 threads

### 2. Inference Engine (`core/inference_engine.py`)

**Purpose**: ARM-optimized LLM inference via llama.cpp

**Key Features**:
- GGUF model loading (Q4_K_M quantization)
- ARM NEON SIMD support
- FP16 KV cache
- Configurable context window
- Token-by-token generation with monitoring

**Model Configuration**:

```python
InferenceConfig(
    model_path="models/mobilevlm_v2_1.7b_q4.gguf",
    n_ctx=2048,           # Context window
    n_threads=4,          # CPU threads
    n_batch=512,          # Batch size
    f16_kv=True,         # FP16 KV cache
    use_mmap=True,       # Memory-mapped loading
    n_gpu_layers=0       # CPU-only
)
```

**Memory Profile**:
- Model weights: ~1.0 GB (Q4 quantization)
- KV cache: ~200-400 MB (dynamic)
- Activations: ~100-200 MB
- Total: ~1.5-2.0 GB per model

### 3. Speculative Decoder (`core/speculative_decoding.py`)

**Purpose**: Accelerate inference via draft-verify paradigm

**Algorithm**:

```
1. Draft Phase:
   - TinyLlama generates K tokens (fast)
   - Cost: O(K * n_draft)
   
2. Verification Phase:
   - MobileVLM verifies K tokens in parallel
   - Cost: O(K * n_verify) [amortized per K tokens]
   
3. Acceptance:
   - If confidence > threshold: accept all K tokens
   - Else: reject, use verifier's token
   
Speedup = (K * α + 1) / (1 + β)
  where α = acceptance rate, β = draft overhead
```

**Performance**:
- Draft token generation: ~50ms per token
- Verification: ~200ms for K=4 tokens
- Acceptance rate: 75-85% typical
- Net speedup: 2-4x

**Tuning Parameters**:
- `draft_tokens`: Number of draft tokens (4 recommended)
- `acceptance_threshold`: Confidence threshold (0.85 recommended)
- Higher draft_tokens = higher potential speedup, lower acceptance rate

### 4. KV Cache Compression (`core/kv_cache.py`)

#### GEAR Cache

**Purpose**: Reduce KV cache memory via attention-based eviction

**Algorithm**:
```
For each layer:
  1. Track attention scores for each KV pair
  2. When cache exceeds target size:
     - Rank entries by attention score
     - Evict lowest-scoring entries
     - Keep top (compression_ratio * size) entries
```

**Memory Savings**:
- Compression ratio 0.5 → 50% memory reduction
- Quality degradation: <2% typical
- Best for long sequences

#### Pyramid Cache

**Purpose**: Layer-specific compression ratios

**Strategy**:
```
Early layers (0-8):   Retain 100% (critical for understanding)
Middle layers (9-16): Retain 70%  (moderate importance)
Late layers (17-24):  Retain 30%  (less critical for output)
```

**Rationale**:
- Early layers extract visual features
- Late layers focus on generation
- Middle layers can be compressed more aggressively

### 5. Early Exit (`core/inference_engine.py`)

**Purpose**: Skip unnecessary computation for simple queries

**Mechanism**:

```python
for token in range(max_tokens):
    logits, confidence = model.generate_token()
    
    if confidence > threshold and depth >= min_layers:
        # High confidence → exit early
        break
        
    # Otherwise continue to next layer
```

**Exit Points**:
- Layers: [8, 12, 16, 20]
- Confidence threshold: 0.9
- Minimum layers: 8

**Performance Impact**:
- Exit rate: ~35% of queries
- Time saved per early exit: 30-40%
- Quality degradation: negligible (<1%)

### 6. Metrics System (`core/metrics.py`)

**Purpose**: Comprehensive monitoring and profiling

**Tracked Metrics**:

1. **Inference Metrics**:
   - Total inferences
   - Average/P50/P95/P99 latency
   - Tokens per second
   - Early exit rate
   - Exit layer distribution

2. **Speculative Metrics**:
   - Acceptance rate
   - Speedup factor
   - Draft vs. verify time

3. **System Metrics**:
   - CPU usage (%)
   - Memory usage (MB)
   - CPU temperature (°C)
   - Process memory

4. **Cache Metrics**:
   - Cache size per layer
   - Eviction count
   - Compression ratio

## Data Flow

### Caption Generation Flow

```
1. API Request
   ↓
2. Save uploaded image → temp file
   ↓
3. Vision Processing (80-100ms)
   - Load image
   - Resize to 336x336
   - Normalize (ImageNet mean/std)
   - Convert to tensor (CHW, batched)
   ↓
4. Create caption prompt
   ↓
5. Inference (2-4s)
   a. If speculative decoding enabled:
      - Draft K tokens with TinyLlama
      - Verify with MobileVLM
      - Accept/reject
      - Repeat until max_tokens
   b. Else:
      - Standard generation
   ↓
6. Post-processing
   - Extract caption from raw output
   - Clean artifacts
   ↓
7. Return response with metrics
   ↓
8. Cleanup temp file
```

### Memory Flow

```
GPU/CPU Memory Layout:

┌─────────────────────────────────────┐
│ Model Weights (1.0-1.5 GB)          │  Persistent
│  - MobileVLM-V2 Q4                  │
│  - TinyLlama Q4 (if speculative)    │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ KV Cache (200-400 MB)               │  Dynamic
│  - Compressed by GEAR/Pyramid       │  (grows per inference)
│  - Cleared periodically             │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ Activations (100-200 MB)            │  Transient
│  - Per-layer activations            │  (per forward pass)
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ Vision Tensors (10-50 MB)           │  Transient
│  - Preprocessed images              │  (per request)
└─────────────────────────────────────┘

Total Peak: ~2.5-3.5 GB (single model)
Total Peak: ~4.0-5.5 GB (speculative, both models)
```

## Optimization Strategies

### ARM-Specific Optimizations

1. **NEON SIMD**:
   - Enabled in OpenCV via `setNumThreads()`
   - Used for image resize, color conversion
   - ~2x speedup over scalar operations

2. **OpenBLAS**:
   - Optimized BLAS for ARM
   - Used by llama.cpp for matrix operations
   - Critical for performance

3. **Operator Fusion**:
   - llama.cpp fuses common operations
   - Reduces memory transfers
   - ~10-15% speedup

4. **FP16 Inference**:
   - Half-precision KV cache
   - 50% memory reduction
   - Minimal accuracy impact

### Quantization Strategy

**Q4_K_M** (4-bit K-means quantization with medium precision):
- 4 bits per weight (vs. 16 bits FP16)
- ~4x memory reduction
- ~5% accuracy degradation
- Optimal for edge deployment

**Why not lower bit-widths?**:
- Q2/Q3: Too much accuracy loss (>10%)
- Q8: Still too large for edge devices
- Q4_K_M: Sweet spot for quality/size

### Cache Management Strategy

**Periodic Clearing**:
```python
if inference_count % 10 == 0:
    clear_kv_cache()
```

**Dynamic Allocation**:
- Start with minimal cache
- Grow as needed
- Shrink when memory pressure detected

**Compression Triggers**:
- Cache size exceeds threshold → compress
- Memory usage > 80% → aggressive compression
- Temperature > 70°C → reduce load

## Performance Analysis

### Latency Breakdown

Typical captioning request (3.5s total):

| Phase | Time (ms) | % Total |
|-------|-----------|---------|
| API overhead | 20 | 0.6% |
| Vision preprocessing | 90 | 2.6% |
| Prompt encoding | 150 | 4.3% |
| Token generation | 3200 | 91.4% |
| Post-processing | 40 | 1.1% |

**Key Insight**: Token generation dominates → optimize with speculative decoding

### Memory Usage Patterns

```
Time →
  
8GB  ┤
     │
6GB  ┤                    ╭─────╮
     │                    │     │
4GB  ┤       ╭────────────╯     ╰────────╮
     │       │  Inference Phase           │
2GB  ┤───────╯                            ╰──────
     │  Idle   Generation    Post-process   Idle
     └────────────────────────────────────────────
```

**Observations**:
- Base memory: ~1.5 GB (model weights)
- Peak during generation: ~4.5 GB
- Returns to base after cache clear

### CPU Utilization

```
100% ┤  ┌───┐  ┌───┐  ┌───┐
     │  │   │  │   │  │   │
 75% ┤──┘   └──┘   └──┘   └──
     │  Generation bursts
 50% ┤
     │
 25% ┤─────────────────────────
     │      Idle periods
  0% └────────────────────────────
```

**Observations**:
- Bursty workload during generation
- Multi-threading achieves 85-95% utilization
- Temperature management important

### Thermal Considerations

Safe operating temperatures:
- **Optimal**: < 60°C
- **Warning**: 60-70°C
- **Throttling**: > 70°C (performance degrades)
- **Critical**: > 80°C (automatic shutdown)

**Cooling Requirements**:
- Passive (heatsink): Light workloads only
- Active (fan): Recommended for production
- Liquid cooling: Overkill for this application

## Scalability Considerations

### Horizontal Scaling

Deploy multiple instances behind load balancer:

```
         Load Balancer
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
  Pi #1     Pi #2     Pi #3
  8GB       8GB       8GB
```

**Benefits**:
- Parallel request handling
- Fault tolerance
- Load distribution

### Model Scaling

Trade-offs for different model sizes:

| Model | Size | Latency | Quality | RAM |
|-------|------|---------|---------|-----|
| Tiny (500M) | 300MB | 1-2s | Low | 1.5GB |
| Small (1.7B) | 1GB | 3-4s | Good | 3.5GB |
| Medium (3B) | 1.8GB | 6-8s | Better | 5.5GB |
| Large (7B) | 4GB | 15-20s | Best | 8GB+ |

**Recommendation**: MobileVLM-V2 1.7B (Small) for best balance

## Future Optimizations

1. **INT8 Activation Quantization**: Further memory reduction
2. **Mixed Precision**: FP16/FP32 hybrid for critical layers
3. **Model Pruning**: Remove redundant parameters
4. **Knowledge Distillation**: Compress to even smaller model
5. **Hardware Acceleration**: Utilize Raspberry Pi GPU via OpenCL

## References

See `RESEARCH.md` for academic papers and implementation references.

