# GPU Acceleration

> [!NOTE]
> > **Status**: Production Ready. Backends provide native GPU execution with automatic
> > fallback to CPU simulation when native runtimes are unavailable.
>
> **Developer Guide**: See [CONTRIBUTING.md](../CONTRIBUTING.md) for GPU coding patterns and [CLAUDE.md](../CLAUDE.md) for backend internals.
> **GPU Backends**: See [GPU Backend Details](gpu-backend-improvements.md) for implementation specifics.

The **GPU** module (`abi.gpu`) provides a unified interface for hardware-accelerated compute across different platforms.

## Unified GPU API (Recommended)

The new unified GPU API provides a single interface for all 8 backends with smart buffer management and optional profiling.

### Quick Start

```zig
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize unified GPU API
    var gpu = try abi.Gpu.init(allocator, .{
        .enable_profiling = true,
        .memory_mode = .automatic,
    });
    defer gpu.deinit();

    // Create buffers with automatic memory management
    const a = try gpu.createBufferFromSlice(f32, &[_]f32{ 1, 2, 3, 4 }, .{});
    defer gpu.destroyBuffer(a);

    const b = try gpu.createBufferFromSlice(f32, &[_]f32{ 5, 6, 7, 8 }, .{});
    defer gpu.destroyBuffer(b);

    const result = try gpu.createBuffer(4 * @sizeOf(f32), .{});
    defer gpu.destroyBuffer(result);

    // Execute vector addition (transfers handled automatically)
    _ = try gpu.vectorAdd(a, b, result);

    // Read results back
    var output: [4]f32 = undefined;
    try result.read(f32, &output);
    // output = { 6, 8, 10, 12 }
}
```

### Configuration Options

```zig
pub const GpuConfig = struct {
    preferred_backend: ?Backend = null,    // null = auto-select best
    allow_fallback: bool = true,
    memory_mode: MemoryMode = .automatic,
    max_memory_bytes: usize = 0,           // 0 = unlimited
    enable_profiling: bool = false,
    multi_gpu: bool = false,
    load_balance_strategy: LoadBalanceStrategy = .memory_aware,
};

pub const MemoryMode = enum {
    automatic,  // API handles all transfers (recommended)
    explicit,   // User controls transfers via toDevice()/toHost()
    unified,    // Use unified memory where available
};
```

### High-Level Operations

The unified API provides these built-in operations:

| Operation | Description |
|-----------|-------------|
| `vectorAdd(a, b, result)` | Element-wise vector addition |
| `matrixMultiply(a, b, result, dims)` | Matrix multiplication |
| `reduceSum(input)` | Sum reduction |
| `dotProduct(a, b)` | Dot product of two vectors |
| `softmax(input, output)` | Softmax activation |

### Multi-GPU Support

```zig
var gpu = try abi.Gpu.init(allocator, .{
    .multi_gpu = true,
    .load_balance_strategy = .memory_aware,
});

// Enable multi-GPU after init
try gpu.enableMultiGpu(.{
    .strategy = .round_robin,
    .enable_peer_access = true,
});

// Get multi-GPU stats
if (gpu.getMultiGpuStats()) |stats| {
    std.debug.print("Active devices: {}\n", .{stats.active_device_count});
}
```

### Profiling and Metrics

```zig
var gpu = try abi.Gpu.init(allocator, .{
    .enable_profiling = true,
});

// ... execute operations ...

// Get metrics summary
if (gpu.getMetricsSummary()) |summary| {
    std.debug.print("Total kernel invocations: {d}\n", .{summary.total_kernel_invocations});
    std.debug.print("Average kernel time: {d:.3}ns\n", .{summary.avg_kernel_time_ns});
}

// Get per-kernel metrics
if (gpu.getKernelMetrics("vectorAdd")) |metrics| {
    std.debug.print("vectorAdd invocations: {d}\n", .{metrics.invocation_count});
}
```

## Portable Kernel DSL

Write kernels once, compile to all backends (CUDA, GLSL, WGSL, MSL).

### Building a Kernel

```zig
const dsl = abi.gpu.dsl;

var builder = dsl.KernelBuilder.init(allocator, "scale_vector");
defer builder.deinit();

// Set workgroup size
_ = builder.setWorkgroupSize(256, 1, 1);

// Define bindings
const input = try builder.addBuffer("input", 0, .{ .scalar = .f32 }, .read_only);
const output = try builder.addBuffer("output", 1, .{ .scalar = .f32 }, .write_only);
const scale = try builder.addUniform("scale", 2, .{ .scalar = .f32 });

// Get global invocation ID
const gid = builder.globalInvocationId();
const idx = try builder.component(try gid.toExpr(), "x");

// output[idx] = input[idx] * scale
const scaled = try builder.mul(
    try builder.index(try input.toExpr(), idx),
    try scale.toExpr()
);
try builder.addStatement(try builder.assignStmt(
    try builder.index(try output.toExpr(), idx),
    scaled
));

// Build and compile
const ir = try builder.build();
var kernel = try gpu.compileKernel(.{ .ir = &ir });
defer kernel.deinit();
```

### Code Generation Targets

| IR Construct | CUDA | GLSL | WGSL | MSL |
|-------------|------|------|------|-----|
| `global_id` | `blockIdx.x * blockDim.x + threadIdx.x` | `gl_GlobalInvocationID.x` | `@builtin(global_invocation_id)` | `thread_position_in_grid` |
| `barrier()` | `__syncthreads()` | `barrier()` | `workgroupBarrier()` | `threadgroup_barrier()` |
| `atomic_add` | `atomicAdd()` | `atomicAdd()` | `atomicAdd()` | `atomic_fetch_add_explicit()` |
| `buffer<f32>` | `float*` | `buffer { float data[]; }` | `var<storage> array<f32>` | `device float*` |

## Backends

ABI supports multiple GPU backends with comprehensive implementations:

| Backend | Status |
|---------|--------|
| CUDA | Complete with tensor core support, async D2D, device queries |
| Vulkan | Complete with SPIR-V generation |
| Metal | Complete with Objective-C runtime bindings |
| WebGPU | Complete with async adapter/device handling |
| OpenGL/ES | Complete with compute shader support |
| std.gpu | Complete with CPU fallback |
| WebGL2 | Correctly returns UnsupportedBackend (no compute support) |

### Backend Details

1.  **CUDA**: NVIDIA GPUs (Linux/Windows) - **Native GPU execution available**
    - Tensor core support for accelerated matrix operations
    - Async device-to-device (D2D) memory transfers
    - Full device capability queries
2.  **Vulkan**: Cross-platform (Linux/Windows/Android)
    - SPIR-V shader generation and compilation
    - Compute shader support
3.  **Metal**: Apple Silicon (macOS)
    - Objective-C runtime bindings for native Metal API access
    - Compute kernel support
4.  **WebGPU**: Browser and native (via Dawn/wgpu)
    - Async adapter and device handling
    - Cross-platform compute shaders
5.  **OpenGL/OpenGL ES**: Legacy and mobile GPU support
    - Compute shader support for OpenGL 4.3+ and ES 3.1+
6.  **std.gpu**: Zig standard library GPU abstraction
    - Complete with automatic CPU fallback
7.  **WebGL2**: Browser-based rendering
    - Correctly returns `UnsupportedBackend` (no compute shader support)

## Memory Management

### Smart Buffers (Unified API)

The unified API handles memory automatically by default:

```zig
// Automatic mode (default) - transfers handled for you
var buf = try gpu.createBufferFromSlice(f32, &data, .{});
_ = try gpu.vectorAdd(buf, other, result);  // Auto upload
try result.read(f32, &output);               // Auto download

// Explicit mode - you control transfers
var buf = try gpu.createBuffer(size, .{ .mode = .explicit });
try buf.write(f32, &data);
try buf.toDevice();   // Explicit upload
// ... operations ...
try buf.toHost();     // Explicit download
```

### Legacy Memory Pool

Use `abi.gpu.GPUMemoryPool` for manual device memory management:

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
zig build run -- gpu backends

# Show GPU module summary
zig build run -- gpu summary

# List detected GPU devices (shows native vs fallback mode)
zig build run -- gpu devices

# Show default GPU device
zig build run -- gpu default
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

## Migration from acceleration.zig

The `acceleration.zig` module is deprecated. Migrate to the unified API:

```zig
// Old API
const accel = try Accelerator.init(allocator, .{});
defer accel.deinit();
_ = try acceleration.vectorAdd(allocator, &a, &b, &result);

// New API
var gpu = try abi.Gpu.init(allocator, .{});
defer gpu.deinit();
const a_buf = try gpu.createBufferFromSlice(f32, &a, .{});
const b_buf = try gpu.createBufferFromSlice(f32, &b, .{});
const result_buf = try gpu.createBuffer(a.len * @sizeOf(f32), .{});
_ = try gpu.vectorAdd(a_buf, b_buf, result_buf);
try result_buf.read(f32, &result);
```

---

## New in 2026.01

### Diagnostics

```zig
const diag = gpu_mod.DiagnosticsInfo.collect(allocator);
if (!diag.isHealthy()) { diag.log(); }
```

Fields: `backend_type`, `device_count`, `memory_stats`, `kernel_cache_stats`, `is_degraded`

### Error Context

```zig
const ctx = error_handling.ErrorContext.init(.backend_error, .cuda, "message");
ctx.log();  // Or ctx.reportErrorFull(allocator)
```

### Graceful Degradation

```zig
var manager = failover.FailoverManager.init(allocator);
manager.setDegradationMode(.automatic);  // .none, .warn_and_continue, .silent
if (manager.isDegraded()) { /* CPU fallback active */ }
```

### SIMD CPU Fallback

When GPU unavailable, `stdgpu` provides AVX/SSE/NEON accelerated operations:
`simdVectorAdd`, `simdDotProduct`, `simdSum`, `simdRelu`, `simdSoftmax`, `simdMatVecMul`

---

## See Also

- [GPU Backend Details](gpu-backend-improvements.md) - Implementation details and improvements
- [Compute Engine](compute.md) - CPU/GPU workload scheduling
- [Monitoring](monitoring.md) - GPU metrics and profiling
- [Troubleshooting](troubleshooting.md) - GPU detection issues
- [API Reference](api_gpu.md) - Complete API documentation
