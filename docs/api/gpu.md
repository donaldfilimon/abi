# gpu API Reference

> GPU acceleration framework (Vulkan, CUDA, Metal, WebGPU)

**Source:** [`src/gpu/mod.zig`](../../src/gpu/mod.zig)

---

GPU Module - Hardware Acceleration API

This module provides a unified interface for GPU compute operations across
multiple backends including CUDA, Vulkan, Metal, WebGPU, OpenGL, and std.gpu.

## Overview

The GPU module abstracts away backend differences, allowing you to write
portable GPU code that runs on any supported hardware. Key features include:

- **Backend Auto-detection**: Automatically selects the best available backend
- **Unified Buffer API**: Cross-platform memory management
- **Kernel DSL**: Write portable kernels that compile to any backend
- **Execution Coordinator**: Automatic fallback from GPU to SIMD to scalar
- **Multi-device Support**: Manage multiple GPUs with peer-to-peer transfers
- **Profiling**: Built-in timing and occupancy analysis

## Available Backends

| Backend | Platform | Build Flag |
|---------|----------|------------|
| CUDA | NVIDIA GPUs | `-Dgpu-backend=cuda` |
| Vulkan | Cross-platform | `-Dgpu-backend=vulkan` |
| Metal | Apple devices | `-Dgpu-backend=metal` |
| WebGPU | Web/Native | `-Dgpu-backend=webgpu` |
| OpenGL | Legacy support | `-Dgpu-backend=opengl` |
| std.gpu | Zig native | `-Dgpu-backend=stdgpu` |

## Public API

These exports form the stable interface:
- `Gpu` - Main unified GPU context
- `GpuConfig` - Configuration for GPU initialization
- `UnifiedBuffer` - Cross-backend buffer type
- `Device`, `DeviceType` - Device discovery and selection
- `KernelBuilder`, `KernelIR` - DSL for custom kernels
- `Backend`, `BackendAvailability` - Backend detection

## Quick Start

```zig
const abi = @import("abi");

// Initialize framework with GPU
var fw = try abi.Framework.init(allocator, .{
.gpu = .{ .backend = .auto },  // Auto-detect best backend
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

## Standalone Usage

```zig
const gpu = abi.gpu;

var g = try gpu.Gpu.init(allocator, .{
.preferred_backend = .vulkan,
.allow_fallback = true,
});
defer g.deinit();

// Check device capabilities
const health = try g.getHealth();
std.debug.print("Backend: {t}\n", .{health.backend});
std.debug.print("Memory: {} MB\n", .{health.memory_total / (1024 * 1024)});
```

## Custom Kernels

```zig
const kernel = gpu.KernelBuilder.init()
.name("my_kernel")
.addParam(.{ .name = "input", .type = .buffer_f32 })
.addParam(.{ .name = "output", .type = .buffer_f32 })
.setBody(
\\output[gid] = input[gid] * 2.0;
)
.build();

// Compile for all backends
const sources = try gpu.compileAll(kernel);
```

## Internal (do not depend on)

These may change without notice:
- Direct backend module imports (cuda_loader, vulkan_*, etc.)
- Lifecycle management internals (gpu_lifecycle, cuda_backend_init_lock)
- Backend-specific initialization functions (initCudaComponents, etc.)

---

## API

### `pub const Context`

<sup>**type**</sup>

GPU Context for Framework integration.

The Context struct wraps the `Gpu` struct to provide a consistent interface
with other framework modules. It handles configuration translation and
provides convenient access to GPU operations.

## Thread Safety

The Context itself is not thread-safe. For concurrent GPU operations,
use the underlying Gpu's stream-based operations or external synchronization.

## Example

```zig
var ctx = try Context.init(allocator, .{ .backend = .vulkan });
defer ctx.deinit();

// Get the underlying Gpu instance
const gpu = ctx.getGpu();

// Create and use buffers
var buffer = try ctx.createBuffer(f32, 1024, .{});
defer ctx.destroyBuffer(&buffer);
```

### `pub fn init(allocator: std.mem.Allocator, cfg: config_module.GpuConfig) !*Context`

<sup>**fn**</sup>

Initialize the GPU context with the given configuration.

## Parameters

- `allocator`: Memory allocator for GPU resources
- `cfg`: GPU configuration (backend selection, memory limits, etc.)

## Returns

A pointer to the initialized Context.

## Errors

- `error.GpuDisabled`: GPU feature is disabled at compile time
- `error.NoDeviceAvailable`: No compatible GPU device found
- `error.OutOfMemory`: Memory allocation failed

### `pub fn getGpu(self: *Context) *Gpu`

<sup>**fn**</sup>

Get the underlying Gpu instance.

### `pub fn createBuffer(self: *Context, comptime T: type, count: usize, options: BufferOptions) !UnifiedBuffer`

<sup>**fn**</sup>

Create a buffer.

### `pub fn createBufferFromSlice(self: *Context, comptime T: type, data: []const T, options: BufferOptions) !UnifiedBuffer`

<sup>**fn**</sup>

Create a buffer from a slice.

### `pub fn destroyBuffer(self: *Context, buffer: *UnifiedBuffer) void`

<sup>**fn**</sup>

Destroy a buffer.

### `pub fn vectorAdd(self: *Context, a: *UnifiedBuffer, b: *UnifiedBuffer, result: *UnifiedBuffer) !ExecutionResult`

<sup>**fn**</sup>

Vector addition.

### `pub fn matrixMultiply(self: *Context, a: *UnifiedBuffer, b: *UnifiedBuffer, result: *UnifiedBuffer, dims: MatrixDims) !ExecutionResult`

<sup>**fn**</sup>

Matrix multiplication.

### `pub fn getHealth(self: *Context) !HealthStatus`

<sup>**fn**</sup>

Get GPU health status.

---

*Generated automatically by `zig build gendocs`*
