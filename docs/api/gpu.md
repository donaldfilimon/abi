---
title: gpu API
purpose: Generated API reference for gpu
last_updated: 2026-03-16
target_zig_version: 0.16.0-dev.2962+08416b44f
---

# gpu

> GPU Module - Hardware Acceleration API

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
var fw = try abi.App.init(allocator, .{
.gpu = .{ .backend = .auto },  // Auto-detect best backend
});
defer fw.deinit();

// Get GPU context
const gpu_ctx = try fw.get(.gpu);
const gpu = gpu_ctx.get(.gpu);

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

**Source:** [`src/features/gpu/mod.zig`](../../src/features/gpu/mod.zig)

**Build flag:** `-Dfeat_gpu=true`

---

## API

### <a id="pub-const-context"></a>`pub const Context`

<sup>**const**</sup> | [source](../../src/features/gpu/mod.zig#L405)

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
const gpu = ctx.get(.gpu);

// Create and use buffers
var buffer = try ctx.createBuffer(f32, 1024, .{});
defer ctx.destroyBuffer(&buffer);
```

### <a id="pub-fn-init-allocator-std-mem-allocator-cfg-config-module-gpuconfig-context"></a>`pub fn init(allocator: std.mem.Allocator, cfg: config_module.GpuConfig) !*Context`

<sup>**fn**</sup> | [source](../../src/features/gpu/mod.zig#L427)

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

### <a id="pub-fn-getgpu-self-context-error-gpu"></a>`pub fn getGpu(self: *Context) Error!*Gpu`

<sup>**fn**</sup> | [source](../../src/features/gpu/mod.zig#L469)

Get the underlying Gpu instance.

### <a id="pub-fn-createbuffer-self-context-comptime-t-type-count-usize-options-bufferoptions-unifiedbuffer"></a>`pub fn createBuffer(self: *Context, comptime T: type, count: usize, options: BufferOptions) !UnifiedBuffer`

<sup>**fn**</sup> | [source](../../src/features/gpu/mod.zig#L474)

Create a buffer.

### <a id="pub-fn-createbufferfromslice-self-context-comptime-t-type-data-const-t-options-bufferoptions-unifiedbuffer"></a>`pub fn createBufferFromSlice(self: *Context, comptime T: type, data: []const T, options: BufferOptions) !UnifiedBuffer`

<sup>**fn**</sup> | [source](../../src/features/gpu/mod.zig#L479)

Create a buffer from a slice.

### <a id="pub-fn-destroybuffer-self-context-buffer-unifiedbuffer-void"></a>`pub fn destroyBuffer(self: *Context, buffer: *UnifiedBuffer) void`

<sup>**fn**</sup> | [source](../../src/features/gpu/mod.zig#L484)

Destroy a buffer.

### <a id="pub-fn-vectoradd-self-context-a-unifiedbuffer-b-unifiedbuffer-result-unifiedbuffer-executionresult"></a>`pub fn vectorAdd(self: *Context, a: *UnifiedBuffer, b: *UnifiedBuffer, result: *UnifiedBuffer) !ExecutionResult`

<sup>**fn**</sup> | [source](../../src/features/gpu/mod.zig#L489)

Vector addition.

### <a id="pub-fn-matrixmultiply-self-context-a-unifiedbuffer-b-unifiedbuffer-result-unifiedbuffer-dims-unified-matrixdims-executionresult"></a>`pub fn matrixMultiply(self: *Context, a: *UnifiedBuffer, b: *UnifiedBuffer, result: *UnifiedBuffer, dims: unified.MatrixDims) !ExecutionResult`

<sup>**fn**</sup> | [source](../../src/features/gpu/mod.zig#L494)

Matrix multiplication.

### <a id="pub-fn-gethealth-self-context-healthstatus"></a>`pub fn getHealth(self: *Context) !HealthStatus`

<sup>**fn**</sup> | [source](../../src/features/gpu/mod.zig#L499)

Get GPU health status.



---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use `zig build full-check` / `zig build check-docs` on supported hosts. On Darwin 25+ / macOS 26+, ABI expects a host-built or otherwise known-good Zig matching `.zigversion`. If stock prebuilt Zig is linker-blocked, record `zig fmt --check ...` plus `zig test <file> -fno-emit-bin` as fallback evidence while replacing the toolchain.
