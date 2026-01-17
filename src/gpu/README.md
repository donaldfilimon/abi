//! # GPU
//!
//! GPU acceleration framework with multi-backend support and kernel DSL.
//!
//! ## Features
//!
//! - **Multi-Backend**: Vulkan, CUDA, Metal, WebGPU, OpenGL, OpenGL ES, WebGL2
//! - **Kernel DSL**: Portable kernel language compiled to native backends
//! - **Automatic Failover**: Graceful degradation to CPU when GPU unavailable
//! - **Memory Pools**: Efficient GPU memory management
//! - **Profiling**: Built-in performance profiling and metrics
//! - **Diagnostics**: Comprehensive GPU state debugging
//!
//! ## Architecture
//!
//! ```
//! User Code (abi.Gpu.vectorAdd, etc.)
//!        ↓
//! Unified API (unified.zig) - Device/buffer management
//!        ↓
//! KernelDispatcher (dispatcher.zig) - Compilation, caching, CPU fallback
//!        ↓
//! Builtin Kernels (builtin_kernels.zig) - Pre-defined operations via DSL
//!        ↓
//! Kernel DSL (dsl/) - Portable IR with code generators
//!        ↓
//! Backend Factory (backend_factory.zig) - Runtime backend selection
//!        ↓
//! Backend VTables (interface.zig) - Polymorphic dispatch
//!        ↓
//! Native Backends (backends/) - CUDA, Vulkan, Metal, WebGPU, OpenGL, stdgpu
//! ```
//!
//! ## Sub-modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | `mod.zig` | Public API entry point |
//! | `unified.zig` | Gpu struct, GpuConfig, high-level operations |
//! | `dsl/` | Kernel DSL compiler (builder, codegen, optimizer) |
//! | `backends/` | Backend implementations with vtables |
//! | `diagnostics.zig` | GPU state debugging (DiagnosticsInfo) |
//! | `error_handling.zig` | Structured error context (ErrorContext) |
//! | `failover.zig` | Graceful degradation (FailoverManager) |
//! | `memory_pool_advanced.zig` | GPU memory management |
//! | `profiling.zig` | Performance profiling |
//! | `stream.zig` | Async command streams |
//!
//! ## Usage
//!
//! ```zig
//! const abi = @import("abi");
//!
//! // Initialize framework with GPU
//! var fw = try abi.init(allocator, .{
//!     .gpu = .{ .backend = .vulkan },
//! });
//! defer fw.deinit();
//!
//! // Get GPU context
//! const gpu = try fw.getGpu();
//!
//! // Vector operations
//! var result = try gpu.vectorAdd(a_buffer, b_buffer);
//! defer gpu.releaseBuffer(result);
//! ```
//!
//! ## Backends
//!
//! | Backend | Flag | Platform |
//! |---------|------|----------|
//! | Vulkan | `-Dgpu-vulkan` | Cross-platform (default) |
//! | CUDA | `-Dgpu-cuda` | NVIDIA |
//! | Metal | `-Dgpu-metal` | Apple |
//! | WebGPU | `-Dgpu-webgpu` | Web/Native |
//! | OpenGL | `-Dgpu-opengl` | Desktop (legacy) |
//! | OpenGL ES | `-Dgpu-opengles` | Mobile/Embedded |
//! | WebGL2 | `-Dgpu-webgl2` | Web browsers |
//! | stdgpu | `-Dgpu-stdgpu` | CPU fallback |
//!
//! ## Build Options
//!
//! Enable with `-Denable-gpu=true` (default: true).
//!
//! ## See Also
//!
//! - [GPU Documentation](../../docs/gpu.md)
//! - [GPU Backend Improvements](../../docs/gpu-backend-improvements.md)
//! - [API Reference](../../docs/api_gpu.md)

