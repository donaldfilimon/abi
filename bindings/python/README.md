# ABI Framework Python Bindings

Python bindings for the ABI Framework, providing high-performance vector database operations and AI inference capabilities.

## Requirements

- Python 3.8+
- ABI shared library (`libabi.dylib` on macOS, `libabi.so` on Linux, `abi.dll` on Windows)

## Building the Shared Library

From the ABI repository root:

```bash
# Build the shared library
zig build lib

# The library will be in zig-out/lib/
```

## Installation

### From Source

```bash
cd bindings/python
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from abi import ABI

# Initialize framework (automatically finds libabi in zig-out/lib/)
abi = ABI()

# Check version
print(f"ABI Version: {abi.version()}")

# Create a vector database with 128-dimensional vectors
db = abi.create_db(dimension=128)

# Insert vectors
import random
for i in range(100):
    vector = [random.random() for _ in range(128)]
    db.insert(id=i, vector=vector)

# Search for similar vectors
query = [random.random() for _ in range(128)]
results = db.search(query, k=10)

for result in results:
    print(f"ID: {result['id']}, Score: {result['score']}")
```

## API Reference

### ABI Class

```python
class ABI:
    def __init__(self, lib_path=None):
        """Initialize the ABI framework.

        Args:
            lib_path: Optional path to libabi. If None, searches standard locations.
        """

    def version(self) -> str:
        """Get the framework version string."""

    def create_db(self, dimension: int) -> VectorDatabase:
        """Create a new vector database.

        Args:
            dimension: Vector dimension for this database.

        Returns:
            VectorDatabase instance.
        """
```

### VectorDatabase Class

```python
class VectorDatabase:
    def insert(self, id: int, vector: list[float]):
        """Insert a vector into the database.

        Args:
            id: Unique identifier for the vector.
            vector: List of floats (must match database dimension).
        """

    def search(self, vector: list[float], k: int = 10) -> list[dict]:
        """Search for similar vectors.

        Args:
            vector: Query vector (must match database dimension).
            k: Number of results to return.

        Returns:
            List of dicts with 'id' and 'score' keys.
        """
```

## Running Tests

```bash
# From the ABI repository root
zig build lib  # Build shared library first
python -m pytest bindings/python/
```

## NumPy Integration

For NumPy array support, install with the numpy extra:

```bash
pip install -e ".[numpy]"
```

Then you can use NumPy arrays directly:

```python
import numpy as np
from abi import ABI

abi = ABI()
db = abi.create_db(128)

# Insert NumPy array
vector = np.random.rand(128).astype(np.float32)
db.insert(1, vector.tolist())

# Search with NumPy array
query = np.random.rand(128).astype(np.float32)
results = db.search(query.tolist(), k=5)
```

## License

MIT License - see the main ABI repository for details.
