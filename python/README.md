# ABI Framework Python Bindings (Experimental)

Python bindings are provided as an experimental interface for selected ABI
features. The API surface may change.

## Build From Source
```bash
cd python
python setup.py build_ext --inplace
pip install -e .
```

## Usage
```python
import abi

framework = abi.Framework()
print(framework.version())
```

## Status
- Feature coverage is limited and evolving.
- Prefer the Zig API for production workloads.
