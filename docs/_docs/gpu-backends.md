---
title: GPU Backends
description: Per-backend details, platform requirements, multi-backend configuration, and vtable abstraction
section: GPU
order: 2
---

# GPU Backends

ABI's GPU module supports 11 backends through a unified vtable abstraction layer
defined in `src/features/gpu/interface.zig`. Each backend implements the same
interface, allowing portable code that compiles and runs on any supported
hardware. The `BackendFactory` (`src/features/gpu/backend_factory.zig`) handles
instantiation and auto-detection.

## Backend Selection

Select backends at build time with the `-Dgpu-backend` flag:

```bash
# Single backend
zig build -Dgpu-backend=metal

# Multiple backends (first available wins at runtime)
zig build -Dgpu-backend=cuda,vulkan

# Auto-detect the best available backend
zig build -Dgpu-backend=auto

# Disable GPU entirely
zig build -Dgpu-backend=none
# or
zig build -Denable-gpu=false
```

When multiple backends are specified, the `BackendFactory.createBestBackend()`
method probes them in priority order and returns the first one that initializes
successfully. The `simulated` backend is always available as a last resort.

## Backend Overview

| Backend | Flag Value | Platform | Hardware Required | Key Features |
|---------|------------|----------|-------------------|--------------|
| CUDA | `cuda` | Windows, Linux | NVIDIA GPU | sm_30-sm_90+, Tensor Cores, cuBLAS, NVRTC |
| Vulkan | `vulkan` | Windows, Linux, macOS | Vulkan 1.2+ driver | SPIR-V codegen, compute shaders |
| Metal | `metal` | macOS, iOS | Apple Silicon or AMD | MPS, CoreML, mesh shaders, ray tracing |
| std.gpu | `stdgpu` | All | None (CPU emulation) | Zig 0.16 native GPU address spaces |
| WebGPU | `webgpu` | All (browser/native) | WebGPU-capable browser or Dawn/wgpu | First-class NN support |
| TPU | `tpu` | When runtime linked | Cloud TPU / libtpu | Stub until runtime linked |
| WebGL2 | `webgl2` | Web | WebGL2-capable browser | Legacy web fallback |
| OpenGL | `opengl` | Desktop | OpenGL 4.3+ | Legacy desktop compute |
| OpenGL ES | `opengles` | Mobile, Embedded | OpenGL ES 3.1+ | Mobile GPU compute |
| FPGA | `fpga` | Reconfigurable HW | FPGA board | Hardware synthesis pipelines |
| Simulated | `simulated` | All | None | Always-on software fallback |

## Per-Backend Details

### CUDA

**Source**: `src/features/gpu/backends/cuda/`
**Modules**: `mod.zig`, `loader.zig`, `device_query.zig`, `memory.zig`, `stream.zig`, `vtable.zig`, `cublas.zig`, `nvrtc.zig`, `native.zig`, `llm_kernels.zig`, `quantized_kernels.zig`

The CUDA backend provides full NVIDIA GPU support with architecture-aware
optimization. It dynamically loads the CUDA driver at runtime via `loader.zig`.

- **Architecture detection**: Identifies compute capability from sm_30 through sm_90+
- **Feature support**: Reports per-architecture capabilities (Tensor Cores, RT cores, FP8, FP16, BF16, TF32)
- **cuBLAS integration**: Optimized BLAS operations for linear algebra
- **NVRTC**: Runtime kernel compilation from source
- **Stream management**: Async kernel execution and memory transfers
- **Quantized kernels**: INT8/FP8 inference support for LLM workloads
- **LLM kernels**: Specialized kernels for transformer inference

**Platform**: Windows and Linux with NVIDIA driver installed. Not available on macOS.

### Vulkan

**Source**: `src/features/gpu/backends/vulkan.zig`, `vulkan_types.zig`, `vulkan_test.zig`

Cross-platform compute via Vulkan 1.2+ compute shaders with SPIR-V codegen
from the kernel DSL.

- **SPIR-V codegen**: The kernel DSL compiles directly to SPIR-V
- **Cross-platform**: Works on Windows, Linux, and macOS (via MoltenVK)
- **Compute shaders**: Full compute pipeline support

**Platform**: Any system with a Vulkan 1.2+ driver. On macOS, requires MoltenVK.

### Metal

**Source**: `src/features/gpu/backends/metal.zig`, `metal_types.zig`, `metal_vtable.zig`

The Metal backend is the natural choice on Apple platforms with extensive
Apple-specific enhancements.

- **GPU Family Detection**: Identifies Apple GPU generation and features
- **MPS (Metal Performance Shaders)**: Optimized ML and image processing operations
- **CoreML Bridge**: Execute CoreML models through the Metal pipeline
- **Mesh Shaders**: Apple mesh shader pipeline for geometry processing
- **Ray Tracing**: Hardware-accelerated ray tracing on Apple Silicon
- **Dual target**: Supports both `.macos` and `.ios` targets

**Platform**: macOS (Apple Silicon or AMD GPU) and iOS.

### std.gpu (Zig Native)

**Source**: `src/features/gpu/backends/stdgpu.zig`, `std_gpu_integration.zig`

Leverages Zig 0.16's native GPU address space support. Provides CPU emulation
when no physical GPU is available, making it useful for development and testing.

- **Native address spaces**: `GlobalPtr`, `SharedPtr`, `StoragePtr`, `UniformPtr`, `ConstantPtr`
- **Shader built-ins**: `globalInvocationId`, `workgroupId`, `localInvocationId`
- **Barriers**: `workgroupBarrier()` for intra-workgroup synchronization
- **CPU fallback**: Emulates GPU execution on CPU when hardware is unavailable

**Platform**: All platforms. No hardware requirement.

### WebGPU

**Source**: `src/features/gpu/backends/webgpu.zig`, `webgpu_vtable.zig`

WebGPU support for both browser and native contexts via Dawn or wgpu.
First-class support for neural network inference.

- **Browser + native**: Works in WebGPU-capable browsers and native applications
- **Neural network support**: Optimized for inference workloads
- **Standard compute**: Full compute shader pipeline

**Platform**: WebGPU-capable browser, or native via Dawn/wgpu runtime.

### TPU (Tensor Processing Unit)

The TPU backend provides a slot for Google's Tensor Processing Units. It
operates as a stub until a TPU runtime library (e.g., libtpu or cloud API) is
linked at build time.

- **Runtime-linked**: Enable with `-Dgpu-backend=tpu` and link a TPU runtime
- **Cloud integration**: Designed for Google Cloud TPU pods
- **Stub behavior**: Returns `error.NotAvailable` when runtime is not linked

**Platform**: Available when linked against libtpu or a compatible cloud API.

### WebGL2

**Source**: `src/features/gpu/backends/webgl2.zig`

Legacy web fallback for browsers that do not support WebGPU.

- **Browser compatibility**: Broad support across older browsers
- **Limited compute**: Relies on fragment shader compute emulation

**Platform**: Web browsers with WebGL2 support.

### OpenGL

**Source**: `src/features/gpu/backends/opengl.zig`, `opengl_vtable.zig`

Legacy desktop GPU support via OpenGL 4.3+ compute shaders.

- **Compute shaders**: OpenGL 4.3 compute shader pipeline
- **Wide driver support**: Works with most desktop GPU drivers

**Platform**: Desktop systems with OpenGL 4.3+ drivers.

### OpenGL ES

**Source**: `src/features/gpu/backends/opengles.zig`

Mobile and embedded GPU support via OpenGL ES 3.1+ compute shaders.

- **Mobile-optimized**: Designed for mobile GPU architectures
- **Embedded support**: Works on embedded Linux with GPU support

**Platform**: Mobile devices and embedded systems with OpenGL ES 3.1+.

### FPGA

**Source**: `src/features/gpu/backends/fpga/`
**Modules**: `mod.zig`, `loader.zig`, `memory.zig`, `kernels.zig`, `types.zig`, `vtable.zig`

Reconfigurable hardware support with hardware synthesis pipelines.

- **Kernel synthesis**: Converts compute kernels to hardware description
- **Custom memory**: FPGA-specific memory management
- **Board-specific**: Configuration varies by FPGA board and vendor

**Platform**: Systems with FPGA boards and appropriate toolchains.

### Simulated (Software Fallback)

**Source**: `src/features/gpu/backends/simulated.zig`

CPU-based simulation of GPU kernels. Always enabled regardless of build flags,
providing a guaranteed fallback for testing and development.

- **Always available**: Cannot be disabled; present in every build
- **CPU execution**: Implements common kernels (vector_add, matmul, reduce_sum) in pure CPU code
- **Testing**: Ideal for CI/CD pipelines without GPU hardware
- **Development**: Write and test GPU code without a physical device
- **Compile + launch**: Supports the full `compile()` / `launch()` lifecycle

```zig
// The simulated backend is always available
var instance = try gpu.createBackend(allocator, .simulated);
defer gpu.destroyBackend(&instance);
```

## Vtable Abstraction Layer

All backends implement the vtable interface defined in `interface.zig`. This
provides runtime polymorphism -- code written against the interface works with
any backend without modification.

```zig
// interface.zig defines the Backend struct with VTable
pub const Backend = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    // Uniform operations across all backends
    pub fn allocMemory(...) ...
    pub fn freeMemory(...) ...
    pub fn launchKernel(...) ...
    pub fn synchronize(...) ...
    pub fn getInfo(...) ...
};
```

The `BackendFactory` wraps this into a higher-level API:

```zig
const gpu = abi.gpu;

// Create a specific backend
var instance = try gpu.createBackend(allocator, .cuda);
defer gpu.destroyBackend(&instance);

// Or auto-detect the best
var best = try gpu.createBestBackend(allocator);
defer gpu.destroyBackend(&best);

// Check feature support
if (instance.supportsFeature(.compute)) { ... }
```

### Standard Error Types

The interface defines a unified error taxonomy that all backends map to:

| Error Type | Description |
|------------|-------------|
| `BackendError` | Lifecycle and device operations (init, timeout, driver) |
| `MemoryError` | Allocation and transfer errors |
| `KernelError` | Compilation and execution errors |
| `InterfaceError` | Feature detection and stub errors |
| `GpuError` | Union of all error categories |

Backend-specific errors are mapped via `mapToStandardError()`.

## Multi-Backend Configuration

When multiple backends are enabled, the framework probes them in a
priority order based on expected performance:

1. CUDA (NVIDIA hardware acceleration)
2. Metal (Apple hardware acceleration)
3. Vulkan (cross-platform hardware acceleration)
4. WebGPU (browser/native acceleration)
5. TPU (tensor processing, when linked)
6. std.gpu (Zig native, CPU emulation)
7. OpenGL / OpenGL ES (legacy)
8. WebGL2 (legacy web)
9. FPGA (specialized hardware)
10. Simulated (always-available fallback)

If the preferred backend fails to initialize, the `FailoverManager`
(`src/features/gpu/failover.zig`) automatically tries the next available
backend. The `RecoveryManager` (`src/features/gpu/recovery.zig`) handles
transient device errors during execution.

## Platform Auto-Detection

```zig
const platform = abi.gpu.platform;

const caps = platform.PlatformCapabilities.detect();
if (caps.cuda_supported) { ... }
if (caps.metal_supported) { ... }
if (caps.vulkan_supported) { ... }
```

Convenience functions:

- `isCudaSupported()` -- NVIDIA driver available
- `isMetalSupported()` -- Apple platform with Metal
- `isVulkanSupported()` -- Vulkan 1.2+ driver present
- `isWebGpuSupported()` -- WebGPU runtime available
- `platformDescription()` -- Human-readable platform summary

## Related

- [GPU Overview](gpu.html) -- Main GPU module documentation
- [AI & LLM](ai.html) -- GPU acceleration for inference and training
- [Architecture](architecture.html) -- Comptime feature gating pattern

## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for new Zig syntax improvements and validation guidance.
