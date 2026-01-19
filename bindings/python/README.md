# ABI Python Bindings
> **Codebase Status:** Synced with repository as of 2026-01-18.

Python bindings for the ABI high-performance AI and vector database framework.

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
db = abi.VectorDatabase(name="my_db")
db.add([1.0, 0.0, 0.0], metadata={"label": "x-axis"})
db.add([0.0, 1.0, 0.0], metadata={"label": "y-axis"})
db.add([0.0, 0.0, 1.0], metadata={"label": "z-axis"})

results = db.query([0.9, 0.1, 0.0], top_k=2)
for r in results:
    print(f"ID: {r['id']}, Score: {r['score']:.4f}, Label: {r['metadata']['label']}")

# AI Agent
agent = abi.Agent(name="assistant")
response = agent.process("Hello!")
print(response)

# Cleanup
abi.shutdown()
```

## Features

- **High-performance SIMD operations** - Accelerated vector math when available
- **Vector database** - In-memory vector storage with similarity search
- **AI agents** - Interface for language model interactions
- **Cross-platform** - Works on Linux, macOS, and Windows

## API Reference

### Core Functions

- `abi.init()` - Initialize the framework
- `abi.shutdown()` - Cleanup resources
- `abi.version()` - Get framework version
- `abi.is_initialized()` - Check initialization status

### Vector Operations

- `abi.cosine_similarity(a, b)` - Compute cosine similarity
- `abi.vector_dot(a, b)` - Compute dot product
- `abi.vector_add(a, b)` - Element-wise addition
- `abi.l2_norm(vec)` - Compute L2 norm
- `abi.has_simd()` - Check SIMD availability

### Classes

- `abi.VectorDatabase` - Vector storage and search
- `abi.Agent` - AI agent interface
- `abi.Feature` - Feature flags enumeration

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

## License

MIT License - see LICENSE file for details.

## See Also

- [API Reference](../../API_REFERENCE.md) - Full API documentation
- [CONTRIBUTING.md](../../CONTRIBUTING.md) - Development guidelines
