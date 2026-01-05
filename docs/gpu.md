# GPU Acceleration

> [!NOTE]
> **Status**: Production Ready. Backends provide native GPU execution with automatic
> fallback to CPU simulation when native runtimes are unavailable.

The **GPU** module (`abi.gpu`) provides a unified interface for hardware-accelerated compute across different platforms.

## Backends

ABI is designed to support multiple backends:

1.  **CUDA**: NVIDIA GPUs (Linux/Windows) - **Native GPU execution available**
2.  **Vulkan**: Cross-platform (Linux/Windows/Android) - CPU fallback
3.  **Metal**: Apple Silicon (macOS) - CPU fallback
4.  **WebGPU**: Browser and native (via Dawn/wgpu) - CPU fallback

### Native CUDA Implementation

The CUDA backend now supports real GPU execution with automatic fallback:

- **Native GPU mode**: Uses CUDA Driver API for actual GPU execution when CUDA hardware is available
- **Automatic fallback**: Gracefully degrades to CPU simulation if CUDA is unavailable
- **Auto-selection**: System automatically chooses native vs fallback at runtime
- **No manual configuration required**: The system detects and uses available hardware transparently

The native CUDA implementation includes:
- Real kernel compilation via CUDA Driver API
- GPU device memory allocation
- Pinned host memory for faster transfers
- Asynchronous memory copies (H2D, D2H, D2D)
- Stream management with synchronization
- Event-based profiling

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

Use `abi.gpu.GPUMemoryPool` to manage device memory efficiently.

```zig
var pool = abi.gpu.GPUMemoryPool.init(allocator, 1024 * 1024 * 64); // 64MB
defer pool.deinit();

const buffer = try abi.gpu.GPUBuffer.init(
    allocator,
    1024 * 1024, // 1MB
    .{ .device_local = true, .write_only = true }
);
defer buffer.deinit();
```

## CLI Commands

Check GPU status and capabilities:

```bash
# List available backends and their status
abi gpu backends

# Show GPU module summary
abi gpu summary

# List detected GPU devices (shows native vs fallback mode)
abi gpu devices

# Show default GPU device
abi gpu default

# Show detailed CUDA status (native vs fallback)
abi gpu status
```

## Building with GPU Support

Enable GPU backends at build time:

```bash
# Enable all GPU backends (default)
zig build -Denable-gpu=true

# Enable only CUDA
zig build -Denable-gpu=true -Dgpu-cuda=true -Dgpu-vulkan=false -Dgpu-metal=false

# Disable GPU entirely
zig build -Denable-gpu=false
```

