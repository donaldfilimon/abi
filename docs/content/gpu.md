---
title: GPU
description: 10 backends with unified vtable abstraction and kernel DSL
section: Modules
order: 7
---

# GPU

The GPU module (`src/features/gpu/mod.zig`) provides a unified interface for
hardware-accelerated compute across 10 backends. A vtable-based abstraction
layer ensures portable code that runs on any supported hardware, with automatic
fallback from GPU to SIMD to scalar execution.

## Features

- **Backend auto-detection**: Selects the best available backend at runtime
- **Unified buffer API**: Cross-platform memory management (`UnifiedBuffer`)
- **Kernel DSL**: Write portable kernels that compile to SPIR-V and other targets
- **Execution coordinator**: Automatic fallback chain (GPU, SIMD, scalar)
- **Multi-device support**: Manage multiple GPUs with peer-to-peer transfers (`mega/`)
- **Profiling**: Built-in timing and occupancy analysis
- **Recovery and failover**: Automatic device recovery and backend failover

## Backends

| Backend | Platform | Flag Value | Notes |
|---------|----------|------------|-------|
| CUDA | NVIDIA GPUs | `cuda` | Architecture detection (sm_XX) |
| Vulkan | Cross-platform | `vulkan` | SPIR-V codegen |
| Metal | Apple (macOS/iOS) | `metal` | MPS, CoreML, mesh shaders, ray tracing |
| std.gpu | Zig native | `stdgpu` | Zig 0.16+ native GPU support |
| WebGPU | Web/Native | `webgpu` | Browser and native via Dawn/wgpu |
| WebGL2 | Web | `webgl2` | Legacy web fallback |
| OpenGL | Desktop | `opengl` | Legacy desktop support |
| OpenGL ES | Mobile/Embedded | `opengles` | Mobile GPU support |
| FPGA | Reconfigurable | `fpga` | Hardware synthesis pipelines |
| Simulated | All platforms | `simulated` | Always-on software fallback |

The **simulated** backend is always enabled regardless of build flags, providing
a software fallback for testing and development without GPU hardware.

## Build Configuration

```bash
# Select a single backend
zig build -Dgpu-backend=vulkan

# Select multiple backends
zig build -Dgpu-backend=cuda,vulkan

# Auto-detect the best backend
zig build -Dgpu-backend=auto

# Disable GPU entirely
zig build -Denable-gpu=false
# or
zig build -Dgpu-backend=none
```

On **macOS**, `metal` is the natural choice. **WASM** targets automatically
disable the GPU module.

## Architecture

```
src/features/gpu/
  mod.zig              Main module (public API)
  stub.zig             Disabled stub (returns error.FeatureDisabled)
  backend.zig          Backend selection logic
  backend_factory.zig  Factory for creating backend instances
  unified.zig          Unified GPU context
  unified_buffer.zig   Cross-backend buffer type
  device.zig           Device discovery and selection
  interface.zig        Backend vtable interface
  platform.zig         Platform capability detection
  profiling.zig        Timing and occupancy analysis
  recovery.zig         Device recovery manager
  failover.zig         Backend failover manager
  backends/            Per-backend implementations
    cuda/              CUDA backend (loader, device_query, kernels)
    vulkan/            Vulkan backend (instance, device, pipeline)
    metal/             Metal backend + Apple enhancements
    fpga/              FPGA synthesis backend
    simulated.zig      Software fallback (always available)
    webgpu/            WebGPU backend
    ...
  dsl/                 Kernel DSL with multi-target codegen
  mega/                Multi-GPU orchestration
  memory/              Memory management (base, pool, lockfree)
  dispatch/            Kernel dispatch (types, coordinator, batch)
```

### Vtable Interface

Each backend implements a common vtable interface defined in `interface.zig`.
The `BackendFactory` creates instances via `createBackend()` or
`createBestBackend()` (auto-detection), and the `KernelDispatcher` routes
kernel launches through the appropriate backend.

```zig
const gpu = abi.gpu;

// Auto-create the best available backend
var instance = try gpu.createBestBackend(allocator);
defer gpu.destroyBackend(&instance);
```

## Quick Start

```zig
const abi = @import("abi");

// Initialize framework with GPU
var fw = try abi.Framework.init(allocator, .{
    .gpu = .{ .backend = .auto },
});
defer fw.deinit();

// Get GPU context
const gpu_ctx = try fw.getGpu();
const gpu = gpu_ctx.getGpu();

// Create buffers
var a = try gpu.createBufferFromSlice(f32, &[_]f32{ 1, 2, 3, 4 }, .{});
var b = try gpu.createBufferFromSlice(f32, &[_]f32{ 5, 6, 7, 8 }, .{});
var result = try gpu.createBuffer(f32, 4, .{});
defer {
    gpu.destroyBuffer(&a);
    gpu.destroyBuffer(&b);
    gpu.destroyBuffer(&result);
}

// Perform vector addition
_ = try gpu.vectorAdd(&a, &b, &result);
```

### Standalone Usage

```zig
const gpu = abi.gpu;

var g = try gpu.Gpu.init(allocator, .{
    .preferred_backend = .vulkan,
    .allow_fallback = true,
});
defer g.deinit();

// Check device capabilities
const health = try g.getHealth();
```

## Kernel DSL

The kernel DSL lets you write portable compute kernels that compile to
SPIR-V, Metal Shading Language, GLSL, and other targets:

```zig
const gpu = abi.gpu;

const kernel = gpu.dsl.KernelBuilder.init()
    .name("my_kernel")
    .addParam(.{ .name = "input", .type = .buffer_f32 })
    .addParam(.{ .name = "output", .type = .buffer_f32 })
    .setBody(
        \\output[gid] = input[gid] * 2.0;
    )
    .build();

// Compile for all backends
const sources = try gpu.dsl.compileAll(kernel);
```

The DSL handles backend-specific differences such as GLSL version
directives, CUDA helper functions (`__fract_helper`, `__sign_helper`),
and atomic operation workarounds.

## Metal Enhancements

The Metal backend includes Apple-specific capabilities:

| Feature | Module | Description |
|---------|--------|-------------|
| GPU Family Detection | `gpu_family.zig` | Identifies Apple GPU generation and capabilities |
| MPS Integration | `mps.zig` | Metal Performance Shaders for optimized ML ops |
| CoreML Bridge | `coreml.zig` | CoreML model execution through Metal |
| Mesh Shaders | `mesh_shaders.zig` | Apple mesh shader pipeline |
| Ray Tracing | `ray_tracing.zig` | Hardware-accelerated ray tracing |

Metal supports both `.macos` and `.ios` targets.

## CUDA Enhancements

The CUDA backend provides architecture-aware optimization:

- **Architecture detection**: `CudaArchitecture` enum identifies GPU compute
  capability (sm_30 through sm_90+)
- **Feature support**: `CudaFeatureSupport` reports per-architecture
  capabilities (Tensor Cores, RT cores, FP8, etc.)
- **Targeted compilation**: Generates `-arch=sm_XY` flags matching the
  detected hardware

## Device Capabilities

The `DeviceCaps` structure reports fine-grained hardware features:

| Capability | Description |
|------------|-------------|
| `bf16` | BFloat16 support |
| `tf32` | TensorFloat-32 support |
| `fp8` | FP8 (E4M3/E5M2) support |
| `mesh` | Mesh shader support |
| `ray_tracing` | Hardware ray tracing |
| `neural_engine` | Apple Neural Engine |
| `mps` | Metal Performance Shaders |

## Execution Coordinator

The execution coordinator provides automatic fallback across execution tiers:

1. **GPU** -- Full hardware acceleration via the selected backend
2. **SIMD** -- Vectorized CPU execution via `abi.simd`
3. **Scalar** -- Plain CPU fallback

The coordinator selects the best available tier at runtime and handles
transparent failover if a tier encounters errors.

## Multi-GPU (Mega)

The `mega/` directory provides multi-GPU orchestration with:

- Device enumeration and selection
- Peer-to-peer memory transfers
- Workload partitioning across devices
- Synchronized kernel launches

## Performance Modules

| Module | Description |
|--------|-------------|
| `profiling` | Kernel timing and GPU utilization metrics |
| `occupancy` | Occupancy calculator for optimal thread/block sizing |
| `fusion` | Kernel fusion for reducing launch overhead |
| `adaptive_tiling` | Dynamic tile size selection based on workload |
| `kernel_cache` | Compiled kernel caching |
| `kernel_ring` | Ring buffer for kernel submission |
| `sync_event` | Cross-stream synchronization events |
| `memory/pool` | Advanced memory pooling with suballocation |
| `memory/lockfree` | Lock-free memory pool for concurrent allocations |

## CLI Commands

```bash
zig build run -- gpu status      # Show GPU status
zig build run -- gpu devices     # List available devices
zig build run -- gpu backends    # List available backends
zig build run -- gpu summary     # Full GPU summary
```

## Platform Detection

```zig
const platform = abi.gpu.platform;

const caps = platform.PlatformCapabilities.detect();
if (caps.cuda_supported) { ... }
if (caps.metal_supported) { ... }
if (caps.vulkan_supported) { ... }
```

Convenience functions: `isCudaSupported()`, `isMetalSupported()`,
`isVulkanSupported()`, `isWebGpuSupported()`, `platformDescription()`.

## Related

- [AI & LLM](ai.md) -- GPU acceleration for inference and training
- [Database](database.md) -- GPU-accelerated distance calculations
