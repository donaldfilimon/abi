---
title: "gpu"
tags: []
---
# GPU Acceleration
> **Codebase Status:** Synced with repository as of 2026-01-22.

<p align="center">
  <img src="https://img.shields.io/badge/Module-GPU-green?style=for-the-badge&logo=nvidia&logoColor=white" alt="GPU Module"/>
  <img src="https://img.shields.io/badge/Status-Production_Ready-success?style=for-the-badge" alt="Production Ready"/>
  <img src="https://img.shields.io/badge/Backends-8-blue?style=for-the-badge" alt="8 Backends"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/CUDA-Supported-76B900?logo=nvidia&logoColor=white" alt="CUDA"/>
  <img src="https://img.shields.io/badge/Vulkan-Supported-AC162C?logo=vulkan&logoColor=white" alt="Vulkan"/>
  <img src="https://img.shields.io/badge/Metal-Supported-000000?logo=apple&logoColor=white" alt="Metal"/>
  <img src="https://img.shields.io/badge/WebGPU-Supported-005A9C?logo=w3c&logoColor=white" alt="WebGPU"/>
</p>

<p align="center">
  <a href="#unified-gpu-api-recommended">Unified API</a> •
  <a href="#backends">Backends</a> •
  <a href="#portable-kernel-dsl">Kernel DSL</a> •
  <a href="#device-enumeration">Device Enum</a> •
  <a href="#cli-commands">CLI</a>
</p>

---

> **Status**: Production Ready. Backends provide native GPU execution with automatic fallback to CPU simulation when native runtimes are unavailable.

> **Developer Guide**: See [CONTRIBUTING.md](../CONTRIBUTING.md) for GPU coding patterns and [CLAUDE.md](../CLAUDE.md) for backend internals.
> **GPU Backends**: See [GPU Backend Details](gpu-backend-improvements.md) for implementation specifics.

The **GPU** module (`abi.gpu`) provides a unified interface for hardware-accelerated compute across different platforms.

## Feature Overview

| Feature | Description | Status |
|---------|-------------|--------|
| **Unified API** | Single interface for all backends | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **Multi-GPU** | Multi-device load balancing | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **Auto-Detection** | Backend selection with fallback | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **Kernel DSL** | Write once, compile everywhere | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **SIMD Fallback** | AVX/SSE/NEON when no GPU | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **Profiling** | Per-kernel metrics & timing | ![Ready](https://img.shields.io/badge/-Ready-success) |

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

### Device Enumeration

Enumerate all available GPU devices across all backends:

```zig
const device = abi.gpu.device;

// Enumerate all devices
const all_devices = try device.enumerateAllDevices(allocator);
defer allocator.free(all_devices);

for (all_devices) |dev| {
    std.debug.print("Device: {s} ({})\n", .{dev.name, dev.backend});
    std.debug.print("  Type: {}\n", .{dev.device_type});
    if (dev.total_memory) |mem| {
        std.debug.print("  Memory: {} GB\n", .{mem / (1024 * 1024 * 1024)});
    }
}

// Enumerate devices for a specific backend
const cuda_devices = try device.enumerateDevicesForBackend(allocator, .cuda);
defer allocator.free(cuda_devices);

// Select best device with criteria
const criteria = device.DeviceSelectionCriteria{
    .prefer_discrete = true,
    .min_memory_gb = 4,
    .required_features = &.{.fp16},
};
const best = try device.selectBestDevice(allocator, criteria);
```

### Backend Auto-Detection

The backend factory provides enhanced auto-detection with fallback chains:

```zig
const backend_factory = abi.gpu.backend_factory;

// Detect all available backends
const backends = try backend_factory.detectAvailableBackends(allocator);
defer allocator.free(backends);

// Select best backend with fallback chain
const best = try backend_factory.selectBestBackendWithFallback(allocator, .{
    .preferred = .cuda,
    .fallback_chain = &.{ .vulkan, .metal, .stdgpu },
});

// Select backend with specific feature requirements
const fp16_backend = try backend_factory.selectBackendWithFeatures(allocator, .{
    .required_features = &.{.fp16, .atomics},
    .fallback_to_cpu = false,
});
```

### Unified Execution Coordinator

Automatic GPU → SIMD → scalar fallback for optimal performance:

```zig
const exec = abi.gpu.execution_coordinator;

var coordinator = try exec.ExecutionCoordinator.init(allocator, .{
    .prefer_gpu = true,
    .fallback_chain = &.{ .gpu, .simd, .scalar },
    .gpu_threshold_size = 1024,  // Min elements for GPU
    .simd_threshold_size = 4,    // Min elements for SIMD
});
defer coordinator.deinit();

// Automatic method selection
const a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
const b = [_]f32{ 8, 7, 6, 5, 4, 3, 2, 1 };
var result = [_]f32{0} ** 8;

const method = try coordinator.vectorAdd(&a, &b, &result);
// method is .gpu, .simd, or .scalar depending on availability and size

// Explicit method override
const forced_method = try coordinator.vectorAddWithMethod(&a, &b, &result, .simd);
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

ABI supports 8 GPU backends with comprehensive implementations:

| Backend | Platform | Features | Status |
|---------|----------|----------|--------|
| **CUDA** | NVIDIA GPUs | Tensor cores, async D2D, device queries | ![Complete](https://img.shields.io/badge/-Complete-success) |
| **Vulkan** | Cross-platform | SPIR-V generation, compute shaders | ![Complete](https://img.shields.io/badge/-Complete-success) |
| **Metal** | Apple Silicon | Objective-C bindings, compute kernels | ![Complete](https://img.shields.io/badge/-Complete-success) |
| **WebGPU** | Browser/Native | Async handling, cross-platform | ![Complete](https://img.shields.io/badge/-Complete-success) |
| **OpenGL/ES** | Legacy/Mobile | Compute shaders (4.3+/ES 3.1+) | ![Complete](https://img.shields.io/badge/-Complete-success) |
| **std.gpu** | Zig stdlib | CPU fallback, portable | ![Complete](https://img.shields.io/badge/-Complete-success) |
| **OpenCL** | Cross-platform | Legacy compute support | ![Complete](https://img.shields.io/badge/-Complete-success) |
| **WebGL2** | Browser | Rendering only (no compute) | ![Limited](https://img.shields.io/badge/-Limited-yellow) |

### Backend Details

<details>
<summary><strong>CUDA (NVIDIA GPUs)</strong></summary>

- **Platform**: Linux/Windows
- **Features**: Tensor core support, async D2D transfers, full device queries
- **Best for**: High-performance compute, ML training
</details>

<details>
<summary><strong>Vulkan (Cross-platform)</strong></summary>

- **Platform**: Linux/Windows/Android
- **Features**: SPIR-V shader generation, compute shaders
- **Best for**: Cross-platform GPU compute
</details>

<details>
<summary><strong>Metal (Apple Silicon)</strong></summary>

- **Platform**: macOS/iOS
- **Features**: Objective-C runtime bindings, compute kernels
- **Best for**: Apple hardware optimization
</details>

<details>
<summary><strong>WebGPU (Browser/Native)</strong></summary>

- **Platform**: Browser (via Dawn/wgpu)
- **Features**: Async adapter handling, portable compute
- **Best for**: Web-based GPU compute
</details>

<details>
<summary><strong>std.gpu (Zig stdlib)</strong></summary>

- **Platform**: Any
- **Features**: Automatic CPU fallback, SIMD acceleration
- **Best for**: Development, testing, CPU-only environments
</details>

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

## API Reference

**Source:** `src/gpu/unified.zig`

### Types

| Type | Description |
|------|-------------|
| `Gpu` | Main unified GPU API |
| `GpuConfig` | GPU configuration |
| `ExecutionResult` | Execution result with timing and statistics |
| `MatrixDims` | Matrix dimensions for matrix operations |
| `LaunchConfig` | Kernel launch configuration |
| `CompiledKernel` | Compiled kernel handle |
| `MemoryInfo` | GPU memory information |
| `GpuStats` | GPU statistics |
| `HealthStatus` | Health status |
| `MultiGpuConfig` | Multi-GPU configuration |
| `LoadBalanceStrategy` | Load balance strategy for multi-GPU |

### Gpu Methods

| Method | Description |
|--------|-------------|
| `init(allocator, config)` | Initialize the unified GPU API |
| `deinit()` | Deinitialize and cleanup |
| `selectDevice(selector)` | Select a device based on criteria |
| `getActiveDevice()` | Get the currently active device |
| `listDevices()` | List all available devices |
| `enableMultiGpu(config)` | Enable multi-GPU mode |
| `getDeviceGroup()` | Get multi-GPU device group (if enabled) |
| `distributeWork(total_work)` | Distribute work across multiple GPUs |
| `createBuffer(size, options)` | Create a new buffer |
| `createBufferFromSlice(T, slice, options)` | Create a buffer from a typed slice |
| `destroyBuffer(buffer)` | Destroy a buffer |
| `vectorAdd(a, b, result)` | Vector addition: result = a + b |
| `matrixMultiply(a, b, result, dims)` | Matrix multiplication: result = a * b |
| `reduceSum(input)` | Reduce sum: returns sum of all elements |
| `dotProduct(a, b)` | Dot product: returns a . b |
| `softmax(input, output)` | Softmax: output = softmax(input) |
| `compileKernel(source)` | Compile a kernel from portable source |
| `launchKernel(kernel, config, args)` | Launch a compiled kernel |
| `synchronize()` | Synchronize all pending operations |
| `createStream(options)` | Create a new stream |
| `createEvent(options)` | Create a new event |
| `getStats()` | Get GPU statistics |
| `getMemoryInfo()` | Get memory information |
| `checkHealth()` | Check GPU health |
| `isAvailable()` | Check if GPU is available |
| `getBackend()` | Get the active backend |
| `isProfilingEnabled()` | Check if profiling is enabled |
| `enableProfiling()` | Enable profiling |
| `disableProfiling()` | Disable profiling |
| `getMetricsSummary()` | Get metrics summary (if profiling enabled) |
| `getKernelMetrics(name)` | Get kernel-specific metrics |
| `getMetricsCollector()` | Get the metrics collector directly |
| `resetMetrics()` | Reset all profiling metrics |
| `isMultiGpuEnabled()` | Check if multi-GPU is enabled |
| `getMultiGpuStats()` | Get multi-GPU statistics (if enabled) |
| `activeDeviceCount()` | Get the number of active devices |

### ExecutionResult Methods

| Method | Description |
|--------|-------------|
| `throughputGBps()` | Get throughput in GB/s |
| `elementsPerSecond()` | Get elements per second |

---

## See Also

<table>
<tr>
<td>

### Related Guides
- [GPU Backend Details](gpu-backend-improvements.md) — Implementation specifics
- [Compute Engine](compute.md) — CPU/GPU workload scheduling
- [Monitoring](monitoring.md) — GPU metrics and profiling

</td>
<td>

### Resources
- [Troubleshooting](troubleshooting.md) — GPU detection issues
- [API Reference](../API_REFERENCE.md) — GPU API details
- [Examples](../examples/) — GPU code samples

</td>
</tr>
</table>

---

<p align="center">
  <a href="ai.md">← AI Guide</a> •
  <a href="docs-index.md">Documentation Index</a> •
  <a href="database.md">Database Guide →</a>
</p>

