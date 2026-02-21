# GPU Module Overview

This directory contains the GPU acceleration layer of the ABI framework. Neural network workloads can use **GPU** (CUDA, Metal, Vulkan), **WebGPU**, **TPU** (when a runtime is linked), or **multi-threaded CPU** via `abi.runtime.ThreadPool` and `InferenceConfig.num_threads`.

## Structure

| Directory | Description |
|-----------|-------------|
| **backends/** | Concrete implementations: Vulkan, CUDA, Metal, WebGPU, OpenGL, OpenGL ES, WebGL2, FPGA, TPU (stub), simulated |
| **dsl/** | Domain-specific language and codegen for kernel generation |
| **mega/** | Multi-GPU orchestration and hybrid workload routing |
| **tests/** | Unit tests for the GPU API |

## Backends

| Backend | Build flag | Use case |
|---------|------------|----------|
| CUDA | `-Dgpu-backend=cuda` | NVIDIA GPUs, best GPGPU/NN support |
| Vulkan | `-Dgpu-backend=vulkan` | Cross-platform compute |
| Metal | `-Dgpu-backend=metal` | Apple Silicon / macOS |
| WebGPU | `-Dgpu-backend=webgpu` | Browser and native (Dawn/wgpu), NN-first-class |
| TPU | `-Dgpu-backend=tpu` | Tensor Processing Unit (stub until runtime linked) |
| stdgpu / simulated | Always available | CPU fallback for testing |

Auto-selection priority (neural networks): CUDA → TPU → Metal → Vulkan → WebGPU → … → simulated.

## Core Files

| File | Description |
|------|-------------|
| `mod.zig` | Public API entry point |
| `stub.zig` | Feature-disabled placeholder |
| `unified.zig` | Unified GPU API with multi-backend support |
| `backend.zig` | Backend metadata and availability |
| `backend_factory.zig` | Backend instantiation and priority |
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

- [GPU Documentation](../../docs/_docs/gpu.md)
- [API Reference](../../docs/api/)
- [CLAUDE.md](../../../CLAUDE.md) — GPU backends and feature flags


## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for new Zig syntax improvements and validation guidance.
