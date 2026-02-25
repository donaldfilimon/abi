<<<<<<< Current (Your changes)
=======
---
title: "README"
tags: []
---
# GPU Module Overview
> **Codebase Status:** Synced with repository as of 2026-01-31.

This directory contains the GPU acceleration layer of the ABI framework.

## Structure

The files are grouped by responsibility:

- **backends/** – Concrete implementations for each GPU backend (Vulkan, CUDA, Metal, WebGPU, OpenGL, FPGA, etc.)
- **dsl/** – Domain-specific language utilities for kernel generation
- **tests/** – Unit tests exercising the GPU API

## Core Files

| File | Description |
|------|-------------|
| `mod.zig` | Public API entry point |
| `stub.zig` | Feature-disabled placeholder |
| `unified.zig` | Unified GPU API with multi-backend support |
| `backend.zig` | Backend abstraction layer |
| `backend_factory.zig` | Backend auto-detection and selection |
| `device.zig` | Device enumeration and selection |
| `execution_coordinator.zig` | GPU→SIMD→scalar fallback coordinator |
| `profiling.zig` | GPU profiling and metrics |

## Usage

```zig
const abi = @import("abi");

var gpu = try abi.Gpu.init(allocator, .{
    .enable_profiling = true,
});
defer gpu.deinit();

const a = try gpu.createBufferFromSlice(f32, &input_a, .{});
const b = try gpu.createBufferFromSlice(f32, &input_b, .{});
const result = try gpu.createBuffer(size, .{});

_ = try gpu.vectorAdd(a, b, result);
```

## See Also

- [GPU Documentation](../../docs/content/gpu.html)
- [API Reference](../../API_REFERENCE.md)

>>>>>>> Incoming (Background Agent changes)
