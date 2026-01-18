# GPU Module Overview

This directory contains the GPU acceleration layer of the ABI framework.

The files are grouped by responsibility:

- **backends/** – concrete implementations for each GPU backend (Vulkan, CUDA, etc.).
- **dsl/** – domain‑specific language utilities used to generate or manage kernels.
- **tests/** – unit‑tests exercising the GPU API.
- Core files (`backend.zig`, `backend_factory.zig`, `acceleration.zig`, …) provide the public API and orchestration logic.

The module follows the builder pattern described in the project documentation and uses the unified `Gpu` abstraction throughout the codebase.
"""
