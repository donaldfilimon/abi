# gpu API Reference

> GPU acceleration framework (Vulkan, CUDA, Metal, WebGPU)

**Source:** [`src/gpu/mod.zig`](../../src/gpu/mod.zig)

---

GPU backend detection, kernel management, and memory utilities.

This module provides a unified interface for GPU compute operations across
multiple backends including CUDA, Vulkan, Metal, WebGPU, OpenGL, and std.gpu.

## Public API

These exports form the stable interface:
- `Gpu` - Main unified GPU context
- `GpuConfig` - Configuration for GPU initialization
- `UnifiedBuffer` - Cross-backend buffer type
- `Device`, `DeviceType` - Device discovery and selection
- `KernelBuilder`, `KernelIR` - DSL for custom kernels
- `Backend`, `BackendAvailability` - Backend detection

## Internal (do not depend on)

These may change without notice:
- Direct backend module imports (cuda_loader, vulkan_*, etc.)
- Lifecycle management internals (gpu_lifecycle, cuda_backend_init_lock)
- Backend-specific initialization functions (initCudaComponents, etc.)

## Unified API Example

```zig
const gpu = @import("gpu/mod.zig");

var g = try gpu.Gpu.init(allocator, .{});
defer g.deinit();

var a = try g.createBufferFromSlice(f32, &[_]f32{ 1, 2, 3, 4 }, .{});
var b = try g.createBufferFromSlice(f32, &[_]f32{ 5, 6, 7, 8 }, .{});
var result = try g.createBuffer(4 * @sizeOf(f32), .{});
defer { g.destroyBuffer(&a); g.destroyBuffer(&b); g.destroyBuffer(&result); }

_ = try g.vectorAdd(&a, &b, &result);
```

---

## API

### `pub const Context`

<sup>**type**</sup>

GPU Context for Framework integration.
Wraps the Gpu struct to provide a consistent interface with other modules.

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
