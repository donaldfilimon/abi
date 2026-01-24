---
title: "Python Bindings"
tags: [python, bindings, api]
---
# ABI Python Bindings
> **Codebase Status:** Synced with repository as of 2026-01-24.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/pip-abi-blue?style=for-the-badge" alt="pip"/>
  <img src="https://img.shields.io/badge/Status-Ready-success?style=for-the-badge" alt="Ready"/>
</p>

Python bindings for the ABI high-performance AI and vector database framework.

## Features

- **Core Framework** - Initialize, configure, and manage the ABI framework
- **Vector Operations** - SIMD-accelerated vector math (cosine similarity, dot product, etc.)
- **Vector Database** - In-memory and persistent vector storage with HNSW indexing
- **LLM Inference** - Local LLM inference with GGUF model support
- **GPU Acceleration** - Multi-backend GPU support (CUDA, Vulkan, Metal, etc.)
- **Configuration** - Comprehensive configuration system with builder pattern

## Installation

```bash
pip install abi
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import abi

# Initialize the framework
abi.init()

# Check version
print(f"ABI version: {abi.version()}")

# Vector operations
a = [1.0, 2.0, 3.0, 4.0]
b = [4.0, 3.0, 2.0, 1.0]

similarity = abi.cosine_similarity(a, b)
print(f"Cosine similarity: {similarity}")

dot = abi.vector_dot(a, b)
print(f"Dot product: {dot}")

# Vector database
db = abi.VectorDatabase(name="my_db", dimensions=4)
db.add([1.0, 0.0, 0.0, 0.0], metadata={"label": "x-axis"})
db.add([0.0, 1.0, 0.0, 0.0], metadata={"label": "y-axis"})
db.add([0.0, 0.0, 1.0, 0.0], metadata={"label": "z-axis"})

results = db.search([0.9, 0.1, 0.0, 0.0], top_k=2)
for r in results:
    print(f"ID: {r.id}, Score: {r.score:.4f}, Label: {r.metadata['label']}")

# AI Agent
agent = abi.Agent(name="assistant")
response = agent.process("Hello!")
print(response)

# Cleanup
abi.shutdown()
```

## Modules

### Core (`abi`)

```python
import abi

# Initialization
abi.init()                    # Initialize framework
abi.init(config)              # Initialize with configuration
abi.shutdown()                # Shutdown and release resources
abi.version()                 # Get version string
abi.is_initialized()          # Check initialization status

# Vector operations
abi.cosine_similarity(a, b)   # Cosine similarity
abi.vector_dot(a, b)          # Dot product
abi.vector_add(a, b)          # Element-wise addition
abi.l2_norm(vec)              # L2 norm
abi.has_simd()                # Check SIMD availability
```

### Configuration (`abi.config`)

```python
from abi.config import (
    Config, ConfigBuilder,
    GpuConfig, GpuBackend,
    AiConfig, LlmConfig, EmbeddingsConfig,
    DatabaseConfig, IndexType, DistanceMetric,
)

# Create configuration
config = Config(
    gpu=GpuConfig(backend=GpuBackend.CUDA),
    ai=AiConfig(llm=LlmConfig(context_size=4096)),
    database=DatabaseConfig(path="./vectors.db"),
)

# Use builder pattern
config = (ConfigBuilder()
    .with_gpu(GpuConfig.cuda())
    .with_llm(LlmConfig(model_path="./model.gguf"))
    .with_database(DatabaseConfig.in_memory())
    .build())

# Check features
config.is_enabled("gpu")
config.enabled_features()
```

### LLM Inference (`abi.llm`)

```python
from abi.llm import LlmEngine, InferenceConfig

# Create engine
config = InferenceConfig(
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=256,
)
engine = LlmEngine(config)

# Load model
engine.load_model("./models/llama-7b.gguf")
print(f"Model: {engine.model_info.name}")

# Generate text
response = engine.generate("Once upon a time")
print(response)

# Streaming generation
for token in engine.generate_streaming("Hello"):
    print(token, end="", flush=True)

# Chat interface
response = engine.chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
])

# Tokenization
tokens = engine.tokenize("Hello, world!")
text = engine.detokenize(tokens)
count = engine.count_tokens("Some text")

# Statistics
stats = engine.stats
print(f"Decode speed: {stats.decode_tokens_per_second:.1f} tok/s")

# Cleanup
engine.unload_model()
```

### Training (`abi.training`)

```python
from abi.training import (
    Trainer, TrainingConfig, train,
    TrainingMetrics, TrainingReport,
)

# Quick training with convenience function
config = TrainingConfig(epochs=10, batch_size=32, learning_rate=0.001)
report = train(config, verbose=True)
print(f"Final loss: {report.final_loss:.4f}")

# Training with context manager for more control
config = TrainingConfig(
    epochs=5,
    batch_size=16,
    learning_rate=2e-5,
    optimizer="adamw",
    warmup_steps=100,
    checkpoint_interval=500,
)

with Trainer(config) as trainer:
    for metrics in trainer.train():
        if metrics.step % 100 == 0:
            print(f"Step {metrics.step}: loss={metrics.loss:.4f}, acc={metrics.accuracy:.4f}")

    # Get final report
    report = trainer.get_report()
    print(f"Training complete: {report.epochs} epochs, {report.total_time_seconds:.1f}s")

    # Save checkpoint
    trainer.save_checkpoint("./checkpoints/final.ckpt")

# Pre-configured training profiles
finetune_config = TrainingConfig.for_finetuning()
pretrain_config = TrainingConfig.for_pretraining()
```

### Vector Database (`abi.database`)

```python
from abi.database import (
    VectorDatabase, DatabaseConfig,
    DistanceMetric, IndexType,
)

# Create database
config = DatabaseConfig(
    dimensions=384,
    distance_metric=DistanceMetric.COSINE,
    index_type=IndexType.HNSW,
)
db = VectorDatabase(config=config)

# Add vectors
id1 = db.add([0.1, 0.2, ...], metadata={"label": "doc1"})
id2 = db.add([0.3, 0.4, ...], metadata={"label": "doc2"})

# Batch add
result = db.add_batch([
    {"vector": [0.1, ...], "metadata": {"label": "a"}},
    {"vector": [0.2, ...], "metadata": {"label": "b"}},
])
print(f"Added {result.success_count} vectors")

# Search
results = db.search([0.15, 0.25, ...], top_k=10)
for r in results:
    print(f"ID: {r.id}, Score: {r.score:.4f}")

# Filtered search
results = db.search(
    query_vector,
    top_k=10,
    filter={"category": "news", "year": {"$gte": 2023}},
)

# Hybrid search (vector + text)
results = db.hybrid_search(
    query_vector=embedding,
    query_text="machine learning",
    top_k=10,
    alpha=0.7,  # 70% vector, 30% text
)

# CRUD operations
vec = db.get(id1)                          # Get by ID
db.update(id1, metadata={"updated": True}) # Update
db.delete(id1)                             # Delete
db.clear()                                 # Clear all

# Statistics
stats = db.stats()
print(f"Vectors: {stats.vector_count}")

# Persistence
db.save("./vectors.json")
db = VectorDatabase.load("./vectors.json")
```

### GPU Acceleration (`abi.gpu`)

```python
from abi.gpu import (
    GpuContext, GpuConfig, GpuBackend,
    is_gpu_available, get_best_device,
)

# Check availability
if is_gpu_available():
    device = get_best_device()
    print(f"Using: {device.name}")

# List devices
for device in GpuContext.list_devices():
    print(f"{device.id}: {device.name} ({device.backend.name})")

# Create context
ctx = GpuContext(GpuConfig(backend=GpuBackend.CUDA))
print(f"GPU available: {ctx.is_available}")

# Matrix operations
result = ctx.matrix_multiply(a, b)

# Vector operations
result = ctx.vector_add(v1, v2)
dot = ctx.vector_dot(v1, v2)

# Activations
softmax = ctx.softmax(logits)
silu = ctx.silu(x)
normalized = ctx.rms_norm(x, weight)

# Memory info
total, free = ctx.memory_info()
print(f"GPU memory: {free / 1e9:.1f} / {total / 1e9:.1f} GB")

# Statistics
stats = ctx.stats
print(f"GPU utilization: {stats.gpu_utilization:.1%}")
```

### Observability (`abi.observability`)

```python
from abi.observability import (
    MetricsRegistry, Counter, Gauge, Histogram,
    Tracer, Span,
    Profiler,
    HealthChecker, HealthStatus, HealthCheckResult,
)

# Metrics (Prometheus-compatible)
registry = MetricsRegistry(prefix="myapp_")
requests = registry.counter("http_requests_total", "Total HTTP requests")
requests.inc()

latency = registry.histogram("request_duration_seconds", "Request latency")
with latency.time():
    # ... handle request
    pass

active = registry.gauge("active_connections", "Active connections")
with active.track_inprogress():
    # ... connection handling
    pass

# Export metrics
print(registry.to_prometheus())  # Prometheus text format
print(registry.to_json())        # JSON format

# Distributed Tracing
tracer = Tracer("my-service")
with tracer.start_span("process_request") as span:
    span.set_attribute("http.method", "GET")
    span.set_attribute("http.url", "/api/users")

    with tracer.start_span("database_query") as child:
        child.set_attribute("db.statement", "SELECT * FROM users")
        # ... query database

    span.add_event("cache_hit", {"key": "user:123"})

# Export traces
print(tracer.export_json())

# Profiling
profiler = Profiler()
with profiler.measure("expensive_operation"):
    # ... expensive code
    pass

stats = profiler.get_stats("expensive_operation")
print(f"Avg: {stats['avg_ms']:.2f}ms, Count: {stats['count']}")

# Health Checks
checker = HealthChecker()
checker.register("database", lambda: HealthCheckResult(
    name="database",
    status=HealthStatus.HEALTHY,
    message="Connected",
))
checker.register("cache", lambda: HealthCheckResult(
    name="cache",
    status=HealthStatus.DEGRADED,
    message="High latency",
))

print(checker.to_json())  # {"status": "degraded", "checks": [...]}
print(checker.is_healthy())  # False (degraded != healthy)
```

## Configuration Options

### GPU Backends

| Backend | Description | Platform |
|---------|-------------|----------|
| `AUTO` | Auto-detect best available | All |
| `CUDA` | NVIDIA CUDA | Linux, Windows |
| `VULKAN` | Vulkan compute | All |
| `METAL` | Apple Metal | macOS |
| `WEBGPU` | WebGPU | Web, Desktop |
| `OPENGL` | OpenGL compute | All |
| `CPU` | CPU fallback | All |

### Database Index Types

| Type | Description | Use Case |
|------|-------------|----------|
| `HNSW` | Hierarchical Navigable Small World | General purpose, fast search |
| `IVF_PQ` | Inverted File with Product Quantization | Large datasets, memory efficient |
| `FLAT` | Brute force exact search | Small datasets, exact results |

### Distance Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| `COSINE` | Cosine similarity | Text embeddings |
| `EUCLIDEAN` | L2 distance | General purpose |
| `DOT_PRODUCT` | Inner product | Normalized vectors |

## Examples

See the `examples/` directory for complete examples:

- `basic_usage.py` - Core functionality
- `llm_inference.py` - LLM model loading and generation
- `vector_database.py` - Database operations and search
- `gpu_acceleration.py` - GPU device management and operations
- `configuration.py` - Configuration options and patterns

Run an example:

```bash
cd bindings/python
python examples/basic_usage.py
```

## Building from Source

Requires:
- Python 3.8+
- Zig 0.16.x (for native library)

```bash
# Build the native library
cd ../..
zig build

# Install Python bindings
cd bindings/python
pip install -e .
```

## Testing

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=abi --cov-report=html

# Run only unit tests (no native library required)
pytest -m "not native"

# Run specific test files
pytest tests/test_streaming.py -v
pytest tests/test_training.py -v

# Run integration tests (requires native library)
pytest tests/test_integration.py -v
```

### Test Suites

| Suite | File | Coverage |
|-------|------|----------|
| Core | `test_abi.py` | Core, vectors, database, agent |
| Streaming | `test_streaming.py` | StreamingConfig, TokenEvent, generate_streaming |
| Training | `test_training.py` | TrainingConfig, Trainer, TrainingMetrics, TrainingReport |
| Observability | `test_observability.py` | Metrics, Tracing, Profiler, HealthChecker |
| Integration | `test_integration.py` | Native library when available |

## API Reference

### Classes

| Class | Description |
|-------|-------------|
| `VectorDatabase` | Vector storage and similarity search |
| `Agent` | AI agent interface |
| `LlmEngine` | LLM inference engine |
| `GpuContext` | GPU operations context |
| `Config` | Framework configuration |
| `ConfigBuilder` | Fluent configuration builder |

### Core Functions

| Function | Description |
|----------|-------------|
| `init()` | Initialize the framework |
| `shutdown()` | Cleanup resources |
| `version()` | Get version string |
| `is_initialized()` | Check initialization status |
| `cosine_similarity(a, b)` | Compute cosine similarity |
| `vector_dot(a, b)` | Compute dot product |
| `vector_add(a, b)` | Element-wise addition |
| `l2_norm(vec)` | Compute L2 norm |
| `has_simd()` | Check SIMD availability |

### Database Functions

| Function | Description |
|----------|-------------|
| `create_database(name, dimensions)` | Create new database |
| `open_database(path)` | Open existing database |

### GPU Functions

| Function | Description |
|----------|-------------|
| `is_gpu_available()` | Check GPU availability |
| `list_gpu_devices()` | List available devices |
| `list_gpu_backends()` | List available backends |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ABI_AUTO_INIT` | `false` | Auto-initialize on import |
| `ABI_LOG_LEVEL` | `info` | Logging level |
| `CUDA_VISIBLE_DEVICES` | - | CUDA device selection |

## License

MIT License - see LICENSE file for details.

## See Also

- [API Reference](../../API_REFERENCE.md) - Full API documentation
- [CONTRIBUTING.md](../../CONTRIBUTING.md) - Development guidelines
- [docs/gpu.md](../../docs/gpu.md) - GPU backend details
- [docs/database.md](../../docs/database.md) - Database documentation
