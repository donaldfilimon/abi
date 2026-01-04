# GPU Acceleration

> [!NOTE]
> **Status**: Experimental / Fallback runtime. Backends provide kernel simulation
> and host-backed device memory when native runtimes are unavailable.

The **GPU** module (`abi.gpu`) aims to provide a unified interface for hardware-accelerated compute across different platforms.

## Backends

ABI is designed to support multiple backends:

1.  **CUDA**: NVIDIA GPUs (Linux/Windows).
2.  **Vulkan**: Cross-platform (Linux/Windows/Android).
3.  **Metal**: Apple Silicon (macOS).
4.  **WebGPU**: Browser and native (via Dawn/wgpu).

## Writing Kernels

Kernels are defined in `src/compute/gpu/kernels.zig`. A kernel must provide implementations for supported backends or a CPU fallback.

```zig
pub const VectorAdd = struct {
    pub const cuda_ptx = @embedFile("backends/cuda/kernels/vector_add.ptx");
    pub const metal_lib = @embedFile("backends/metal/kernels/vector_add.metallib");

    pub fn cpu_fallback(a: []f32, b: []f32, out: []f32) void {
        for (a, 0..) |_, i| out[i] = a[i] + b[i];
    }
};
```

## Memory Management

Use `abi.gpu.createPool` to manage device memory efficiently.

```zig
var pool = abi.gpu.createPool(allocator, 1024 * 1024 * 64); // 64MB
defer pool.deinit();

const buffer = try pool.allocate(1024, .{ .device_local = true });
```
