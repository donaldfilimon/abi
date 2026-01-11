//! # GPU Compute Layer
//!
//! GPU-accelerated compute with multiple backends and automatic fallback.
//!
//! ## Features
//!
//! - **Multiple Backends**: CUDA, Vulkan, Metal, WebGPU, OpenGL, OpenGL ES, WebGL2
//! - **Automatic Fallback**: Falls back to simulated/CPU backend when hardware unavailable
//! - **Memory Pooling**: Size-class based allocation with automatic coalescing
//! - **Kernel Caching**: Compiled kernel caching for performance
//! - **Metrics Collection**: Comprehensive performance tracking
//! - **Recovery System**: Automatic device failure recovery
//!
//! ## Backend Status
//!
//! | Backend | Status | Platforms |
//! |---------|--------|-----------|
//! | CUDA | Production | Linux, Windows (NVIDIA) |
//! | Vulkan | Production | Cross-platform |
//! | Metal | Production | macOS, iOS |
//! | WebGPU | Experimental | Web, Native |
//! | OpenGL | Production | Cross-platform |
//! | OpenGL ES | Production | Mobile, Embedded |
//! | WebGL2 | Experimental | Web |
//! | Simulated | Always Available | All platforms |
//!
//! ## Sub-modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | `backends/` | Backend implementations (cuda.zig, vulkan.zig, etc.) |
//! | `tensor/` | GPU tensor operations |
//! | `error_handling.zig` | Structured error tracking |
//! | `kernel_cache.zig` | Compiled kernel caching |
//! | `memory_pool_advanced.zig` | Size-class memory pooling |
//! | `metrics.zig` | Performance metrics collection |
//! | `recovery.zig` | Device failure recovery |
//!
//! ## Usage
//!
//! ### Memory Pool
//!
//! ```zig
//! const abi = @import("abi");
//!
//! var pool = abi.gpu.createPool(allocator, 64 * 1024 * 1024);
//! defer pool.deinit();
//!
//! const buffer = try pool.allocate(4096, .{ .device_local = true });
//! defer _ = pool.free(buffer);
//!
//! try buffer.writeFromHost(&data);
//! try buffer.copyToDevice();
//! ```
//!
//! ### Kernel Execution
//!
//! ```zig
//! const kernel = try abi.gpu.compileKernel(allocator, kernel_source);
//! defer abi.gpu.destroyKernel(allocator, kernel);
//!
//! try abi.gpu.launchKernel(allocator, kernel, .{
//!     .grid = .{ 256, 1, 1 },
//!     .block = .{ 64, 1, 1 },
//! }, &args);
//! ```
//!
//! ### Metrics
//!
//! ```zig
//! var metrics = abi.gpu.MetricsCollector.init(allocator);
//! defer metrics.deinit();
//!
//! metrics.recordKernel("vector_add", duration_ns);
//! metrics.recordTransfer(.host_to_device, bytes, duration_ns);
//!
//! const stats = metrics.getStats();
//! ```
//!
//! ## Device Selection
//!
//! Devices are scored and selected automatically:
//! - Discrete GPU: 1000 points
//! - Integrated GPU: 500 points
//! - Virtual GPU: 100 points
//! - CPU: 50 points
//! - API version bonus applied
//!
//! ## Feature Flags
//!
//! | Flag | Description |
//! |------|-------------|
//! | `-Denable-gpu` | Enable GPU features (default: true) |
//! | `-Dgpu-cuda` | Enable CUDA backend |
//! | `-Dgpu-vulkan` | Enable Vulkan backend |
//! | `-Dgpu-metal` | Enable Metal backend |
//! | `-Dgpu-webgpu` | Enable WebGPU backend |
//! | `-Dgpu-opengl` | Enable OpenGL backend |
//!
//! ## See Also
//!
//! - [GPU Documentation](../../../docs/gpu.md)
//! - [Features GPU](../../features/gpu/README.md)
