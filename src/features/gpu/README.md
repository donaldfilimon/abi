# GPU Module Overview

This directory contains the GPU acceleration layer of the ABI framework. Neural network workloads can use **GPU** (CUDA, Metal, Vulkan), **WebGPU**, **TPU** (when a runtime is linked), or **multi-threaded CPU** via `abi.runtime.ThreadPool` and `InferenceConfig.num_threads`.

## Structure

| Directory | Description |
|-----------|-------------|
| **backends/** | Concrete implementations: Vulkan, CUDA, Metal, WebGPU, OpenGL, OpenGL ES, WebGL2, FPGA, TPU (stub), simulated |
| **policy/** | Canonical platform policy for backend ordering and optimization hints |
| **factory/** | Backend creation and selection modules |
| **device/** | Device typing, selection, enumeration, and Android adaptive probing |
| **runtime/** | Runtime-facing context/buffer/ops namespaced exports |
| **multi/** | Multi-device namespace entrypoint |
| **dsl/** | Domain-specific language and codegen for kernel generation |
| **mega/** | Multi-GPU orchestration and hybrid workload routing |
| **tests/** | Unit tests for the GPU API |

## Backends

| Backend | Build flag | Use case |
|---------|------------|----------|
| CUDA | `-Dgpu-backend=cuda` | NVIDIA GPUs, best GPGPU/NN support |
| Vulkan | `-Dgpu-backend=vulkan` | Cross-platform compute (Linux requires Vulkan 1.3+) |
| Metal | `-Dgpu-backend=metal` | Apple Silicon / macOS (requires Metal 4 runtime capability) |
| WebGPU | `-Dgpu-backend=webgpu` | Browser and native (Dawn/wgpu), NN-first-class |
| TPU | `-Dgpu-backend=tpu` | Tensor Processing Unit (stub until runtime linked) |
| stdgpu / simulated | Always available | CPU fallback for testing |

Auto-selection policy is platform-specific and centralized under `policy/`:

- macOS: Metal → Vulkan → OpenGL → stdgpu
- Linux: CUDA → Vulkan → OpenGL → stdgpu
- Windows: CUDA → Vulkan → OpenGL → stdgpu (default). Opt in to `stdgpu`-only with `-Dgpu-auto-windows-safe=true`
- iOS: Metal → OpenGL ES → stdgpu
- Android: adaptive Vulkan/OpenGL ES probe, then stdgpu fallback
- Web: WebGPU → WebGL2 (runtime may fall back to simulated)

OpenGL and OpenGL ES are treated as a unified GL backend family with profile adapters.
Both `-Dgpu-backend=opengl` and `-Dgpu-backend=opengles` remain supported and may be enabled together.

## Core Files

| File | Description |
|------|-------------|
| `mod.zig` | Public API entry point |
| `stub.zig` | Feature-disabled placeholder |
| `unified.zig` | Unified GPU API with multi-backend support |
| `backend.zig` | Backend metadata and availability |
| `backend_factory.zig` | Backend instantiation and priority |
| `device.zig` | Device enumeration and selection |
| `policy/mod.zig` | Canonical platform backend policy + optimization hints |
| `backends/registry.zig` | Backend registry + VTable creation routing |
| `backends/pool.zig` | Multi-backend instance pooling for clusters |
| `execution_coordinator.zig` | GPU→SIMD→scalar fallback coordinator |
| `profiling.zig` | GPU profiling and metrics |

## Namespaced API

The GPU module now exposes explicit namespaces for core surfaces:

- `abi.gpu.backends`
- `abi.gpu.devices`
- `abi.gpu.runtime`
- `abi.gpu.policy`
- `abi.gpu.multi`
- `abi.gpu.factory`

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
