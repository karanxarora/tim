# EdgeVLM API Reference

Complete API documentation for EdgeVLM REST endpoints.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, no authentication is required. For production deployment, consider adding:
- API keys
- OAuth2
- IP whitelisting

---

## Endpoints

### 1. Root Endpoint

**GET /**

Returns service information and available endpoints.

**Response**:
```json
{
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
```

---

### 2. Health Check

**GET /health**

Check API health and system status.

**Response**:
```json
{
  "status": "healthy",
  "pipeline_loaded": true,
  "timestamp": "2025-10-22T10:30:45.123456",
  "system_info": {
    "cpu_percent": 45.2,
    "memory_percent": 62.8,
    "temperature_celsius": 58.5
  }
}
```

**Status Codes**:
- `200`: Healthy
- `503`: Service unavailable (pipeline not initialized)

---

### 3. Image Captioning

**POST /caption**

Generate a descriptive caption for an image.

**Request**:

```bash
curl -X POST "http://localhost:8000/caption" \
  -F "image=@/path/to/image.jpg" \
  -F "max_length=128"
```

**Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image` | File | Yes | - | Image file (JPEG, PNG, etc.) |
| `max_length` | Integer | No | 128 | Maximum caption length in tokens (1-256) |

**Response**:

```json
{
  "caption": "A brown dog sitting on a wooden bench in a park with green trees in the background",
  "latency": 3.245,
  "preprocessing_time": 0.089,
  "inference_time": 3.156,
  "tokens_per_second": 18.5,
  "early_exit": true,
  "exit_layer": 12,
  "timestamp": "2025-10-22T10:30:45.123456"
}
```

**Response Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `caption` | String | Generated caption |
| `latency` | Float | Total request latency (seconds) |
| `preprocessing_time` | Float | Vision preprocessing time (seconds) |
| `inference_time` | Float | Model inference time (seconds) |
| `tokens_per_second` | Float | Generation speed |
| `early_exit` | Boolean | Whether early exit was triggered |
| `exit_layer` | Integer or null | Exit layer if early exit occurred |
| `timestamp` | String | Response timestamp (ISO 8601) |

**Status Codes**:
- `200`: Success
- `400`: Invalid request (bad image format, invalid parameters)
- `500`: Internal server error
- `503`: Service unavailable

**Example (Python)**:

```python
import requests

with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/caption',
        files={'image': f},
        data={'max_length': 128}
    )

result = response.json()
print(f"Caption: {result['caption']}")
print(f"Latency: {result['latency']:.2f}s")
```

---

### 4. Visual Question Answering (VQA)

**POST /vqa**

Answer a question about an image.

**Request**:

```bash
curl -X POST "http://localhost:8000/vqa" \
  -F "image=@/path/to/image.jpg" \
  -F "question=What color is the dog?" \
  -F "max_length=64"
```

**Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image` | File | Yes | - | Image file (JPEG, PNG, etc.) |
| `question` | String | Yes | - | Question to answer (1-500 chars) |
| `max_length` | Integer | No | 64 | Maximum answer length in tokens (1-128) |

**Response**:

```json
{
  "question": "What color is the dog?",
  "answer": "Brown",
  "latency": 2.156,
  "preprocessing_time": 0.087,
  "inference_time": 2.069,
  "tokens_per_second": 22.3,
  "early_exit": false,
  "exit_layer": null,
  "timestamp": "2025-10-22T10:31:12.654321"
}
```

**Response Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `question` | String | Original question |
| `answer` | String | Generated answer |
| `latency` | Float | Total request latency (seconds) |
| `preprocessing_time` | Float | Vision preprocessing time (seconds) |
| `inference_time` | Float | Model inference time (seconds) |
| `tokens_per_second` | Float | Generation speed |
| `early_exit` | Boolean | Whether early exit was triggered |
| `exit_layer` | Integer or null | Exit layer if early exit occurred |
| `timestamp` | String | Response timestamp (ISO 8601) |

**Status Codes**:
- `200`: Success
- `400`: Invalid request
- `500`: Internal server error
- `503`: Service unavailable

**Example (Python)**:

```python
import requests

with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/vqa',
        files={'image': f},
        data={
            'question': 'What is in this image?',
            'max_length': 64
        }
    )

result = response.json()
print(f"Q: {result['question']}")
print(f"A: {result['answer']}")
```

---

### 5. Performance Metrics

**GET /metrics**

Get comprehensive performance metrics.

**Response**:

```json
{
  "inference_metrics": {
    "total_inferences": 150,
    "avg_latency": 3.245,
    "min_latency": 2.156,
    "max_latency": 5.678,
    "p50_latency": 3.189,
    "p95_latency": 4.523,
    "p99_latency": 5.234,
    "avg_tokens_per_second": 18.5,
    "early_exit_count": 52,
    "early_exit_rate": 0.347,
    "avg_memory_mb": 4250.5,
    "max_memory_mb": 5123.8
  },
  "vision_metrics": {
    "total_images": 150,
    "avg_time": 0.089,
    "min_time": 0.065,
    "max_time": 0.123,
    "std_time": 0.015
  },
  "system_metrics": {
    "sample_count": 300,
    "avg_cpu_percent": 87.5,
    "max_cpu_percent": 98.2,
    "avg_memory_percent": 65.3,
    "max_memory_percent": 78.9,
    "avg_process_memory_mb": 4250.5,
    "max_process_memory_mb": 5123.8,
    "avg_temperature_celsius": 62.3,
    "max_temperature_celsius": 68.7
  },
  "cache_metrics": {
    "clear_count": 15,
    "max_cache_size_mb": 512,
    "gear": {
      "total_entries": 1250,
      "total_evictions": 450,
      "layers_cached": 24,
      "compression_ratio": 0.5,
      "eviction_policy": "attention_score"
    },
    "pyramid": {
      "total_entries": 1100,
      "layers_cached": 24,
      "layer_sizes": {"0": 64, "1": 64, "...": "..."},
      "compression_ratios": {"0": 1.0, "8": 0.9, "...": "..."}
    }
  }
}
```

**Status Codes**:
- `200`: Success
- `500`: Failed to retrieve metrics
- `503`: Service unavailable

---

### 6. Clear Cache

**POST /clear-cache**

Manually clear all KV caches to free memory.

**Response**:

```json
{
  "status": "success",
  "message": "Cache cleared"
}
```

**Status Codes**:
- `200`: Success
- `500`: Failed to clear cache
- `503`: Service unavailable

**Example**:

```bash
curl -X POST "http://localhost:8000/clear-cache"
```

---

### 7. Benchmark

**POST /benchmark**

Run performance benchmark.

**Request**:

```bash
curl -X POST "http://localhost:8000/benchmark" \
  -F "image=@/path/to/image.jpg" \
  -F "task_type=caption" \
  -F "num_runs=10"
```

**Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image` | File | Yes | - | Image file for benchmarking |
| `task_type` | String | No | "caption" | Task type: "caption" or "vqa" |
| `question` | String | No | null | Question (required if task_type="vqa") |
| `num_runs` | Integer | No | 10 | Number of benchmark iterations |

**Response**:

```json
{
  "task_type": "caption",
  "num_runs": 10,
  "avg_latency": 3.245,
  "min_latency": 2.987,
  "max_latency": 3.678,
  "p50_latency": 3.189,
  "p95_latency": 3.456,
  "std_latency": 0.234,
  "inference_metrics": {
    "total_inferences": 10,
    "avg_latency": 3.245,
    "...": "..."
  },
  "system_metrics": {
    "avg_cpu_percent": 89.2,
    "avg_memory_mb": 4350.5,
    "...": "..."
  }
}
```

**Status Codes**:
- `200`: Success
- `400`: Invalid parameters
- `500`: Benchmark failed
- `503`: Service unavailable

---

## Error Responses

All endpoints may return error responses in this format:

```json
{
  "detail": "Error description"
}
```

### Common Errors

**400 Bad Request**:
```json
{
  "detail": "File must be an image"
}
```

**500 Internal Server Error**:
```json
{
  "detail": "Caption generation failed: Out of memory"
}
```

**503 Service Unavailable**:
```json
{
  "detail": "Pipeline not initialized"
}
```

---

## Rate Limiting

Currently, no rate limiting is enforced. For production:

- Recommended: 10 requests/minute per client
- Implement with middleware (e.g., slowapi)
- Consider queue-based processing for high load

---

## CORS

CORS is enabled for all origins by default. To restrict:

```python
# In api.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

---

## WebSocket Support (Future)

For real-time streaming captions:

```javascript
// Future implementation
const ws = new WebSocket('ws://localhost:8000/ws/caption');
ws.send(imageBlob);
ws.onmessage = (event) => {
  const token = JSON.parse(event.data);
  updateCaption(token);
};
```

---

## Client Libraries

### Python

```python
class EdgeVLMClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def caption(self, image_path, max_length=128):
        with open(image_path, 'rb') as f:
            response = requests.post(
                f"{self.base_url}/caption",
                files={'image': f},
                data={'max_length': max_length}
            )
        return response.json()
    
    def vqa(self, image_path, question, max_length=64):
        with open(image_path, 'rb') as f:
            response = requests.post(
                f"{self.base_url}/vqa",
                files={'image': f},
                data={'question': question, 'max_length': max_length}
            )
        return response.json()
    
    def metrics(self):
        response = requests.get(f"{self.base_url}/metrics")
        return response.json()

# Usage
client = EdgeVLMClient()
result = client.caption("image.jpg")
print(result['caption'])
```

### JavaScript/Node.js

```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

class EdgeVLMClient {
  constructor(baseURL = 'http://localhost:8000') {
    this.baseURL = baseURL;
  }
  
  async caption(imagePath, maxLength = 128) {
    const form = new FormData();
    form.append('image', fs.createReadStream(imagePath));
    form.append('max_length', maxLength);
    
    const response = await axios.post(
      `${this.baseURL}/caption`,
      form,
      { headers: form.getHeaders() }
    );
    
    return response.data;
  }
  
  async vqa(imagePath, question, maxLength = 64) {
    const form = new FormData();
    form.append('image', fs.createReadStream(imagePath));
    form.append('question', question);
    form.append('max_length', maxLength);
    
    const response = await axios.post(
      `${this.baseURL}/vqa`,
      form,
      { headers: form.getHeaders() }
    );
    
    return response.data;
  }
}

// Usage
const client = new EdgeVLMClient();
const result = await client.caption('image.jpg');
console.log(result.caption);
```

---

## Performance Tips

1. **Batch Processing**: Process multiple images sequentially rather than parallel to avoid OOM
2. **Cache Clearing**: Clear cache periodically if running many requests
3. **Image Size**: Resize large images before uploading to reduce preprocessing time
4. **Connection Pooling**: Reuse HTTP connections for multiple requests

---

## Monitoring

### Prometheus Metrics (Future)

Expose metrics in Prometheus format:

```
# HELP edgevlm_inference_latency_seconds Inference latency
# TYPE edgevlm_inference_latency_seconds histogram
edgevlm_inference_latency_seconds_bucket{le="1.0"} 0
edgevlm_inference_latency_seconds_bucket{le="2.0"} 15
edgevlm_inference_latency_seconds_bucket{le="5.0"} 142
...
```

### Health Checks

For load balancers and orchestration:

```bash
# Kubernetes liveness probe
curl -f http://localhost:8000/health || exit 1

# Docker health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
```

---

## API Versioning (Future)

Future versions will use URL versioning:

```
http://localhost:8000/v1/caption
http://localhost:8000/v2/caption  # Future version
```

---

## Support

For API issues:
- Check logs: `tail -f logs/edgevlm.log`
- Test health: `curl http://localhost:8000/health`
- Review metrics: `curl http://localhost:8000/metrics`

For bugs, open an issue on GitHub with:
- Request/response details
- Error logs
- System information

