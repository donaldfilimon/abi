//! GPU Module - Hardware Acceleration API
//!
//! This module provides a unified interface for GPU compute operations across
//! multiple backends including CUDA, Vulkan, Metal, WebGPU, OpenGL, and std.gpu.
//!
//! ## Overview
//!
//! The GPU module abstracts away backend differences, allowing you to write
//! portable GPU code that runs on any supported hardware. Key features include:
//!
//! - **Backend Auto-detection**: Automatically selects the best available backend
//! - **Unified Buffer API**: Cross-platform memory management
//! - **Kernel DSL**: Write portable kernels that compile to any backend
//! - **Execution Coordinator**: Automatic fallback from GPU to SIMD to scalar
//! - **Multi-device Support**: Manage multiple GPUs with peer-to-peer transfers
//! - **Profiling**: Built-in timing and occupancy analysis
//!
//! ## Available Backends
//!
//! | Backend | Platform | Build Flag |
//! |---------|----------|------------|
//! | CUDA | NVIDIA GPUs | `-Dgpu-backend=cuda` |
//! | Vulkan | Cross-platform | `-Dgpu-backend=vulkan` |
//! | Metal | Apple devices | `-Dgpu-backend=metal` |
//! | WebGPU | Web/Native | `-Dgpu-backend=webgpu` |
//! | OpenGL | Legacy support | `-Dgpu-backend=opengl` |
//! | std.gpu | Zig native | `-Dgpu-backend=stdgpu` |
//!
//! ## Public API
//!
//! These exports form the stable interface:
//! - `Gpu` - Main unified GPU context
//! - `GpuConfig` - Configuration for GPU initialization
//! - `UnifiedBuffer` - Cross-backend buffer type
//! - `Device`, `DeviceType` - Device discovery and selection
//! - `KernelBuilder`, `KernelIR` - DSL for custom kernels
//! - `Backend`, `BackendAvailability` - Backend detection
//!
//! ## Quick Start
//!
//! ```zig
//! const abi = @import("abi");
//!
//! // Initialize framework with GPU
//! var fw = try abi.Framework.init(allocator, .{
//!     .gpu = .{ .backend = .auto },  // Auto-detect best backend
//! });
//! defer fw.deinit();
//!
//! // Get GPU context
//! const gpu_ctx = try fw.getGpu();
//! const gpu = gpu_ctx.getGpu();
//!
//! // Create buffers
//! var a = try gpu.createBufferFromSlice(f32, &[_]f32{ 1, 2, 3, 4 }, .{});
//! var b = try gpu.createBufferFromSlice(f32, &[_]f32{ 5, 6, 7, 8 }, .{});
//! var result = try gpu.createBuffer(f32, 4, .{});
//! defer {
//!     gpu.destroyBuffer(&a);
//!     gpu.destroyBuffer(&b);
//!     gpu.destroyBuffer(&result);
//! }
//!
//! // Perform vector addition
//! _ = try gpu.vectorAdd(&a, &b, &result);
//! ```
//!
//! ## Standalone Usage
//!
//! ```zig
//! const gpu = abi.gpu;
//!
//! var g = try gpu.Gpu.init(allocator, .{
//!     .preferred_backend = .vulkan,
//!     .allow_fallback = true,
//! });
//! defer g.deinit();
//!
//! // Check device capabilities
//! const health = try g.getHealth();
//! std.debug.print("Backend: {t}\n", .{health.backend});
//! std.debug.print("Memory: {} MB\n", .{health.memory_total / (1024 * 1024)});
//! ```
//!
//! ## Custom Kernels
//!
//! ```zig
//! const kernel = gpu.KernelBuilder.init()
//!     .name("my_kernel")
//!     .addParam(.{ .name = "input", .type = .buffer_f32 })
//!     .addParam(.{ .name = "output", .type = .buffer_f32 })
//!     .setBody(
//!         \\output[gid] = input[gid] * 2.0;
//!     )
//!     .build();
//!
//! // Compile for all backends
//! const sources = try gpu.compileAll(kernel);
//! ```
//!
//! ## Internal (do not depend on)
//!
//! These may change without notice:
//! - Direct backend module imports (cuda_loader, vulkan_*, etc.)
//! - Lifecycle management internals (gpu_lifecycle, cuda_backend_init_lock)
//! - Backend-specific initialization functions (initCudaComponents, etc.)
const std = @import("std");
const time = @import("../../services/shared/time.zig");
const sync = @import("../../services/shared/sync.zig");
const backend = @import("backend.zig");
const kernels = @import("runtime_kernels.zig");
const memory = @import("memory/base.zig");
const kernel_cache = @import("kernel_cache.zig");
const backend_shared = @import("backends/shared.zig");
pub const profiling = @import("profiling.zig");

// Performance optimization modules
pub const occupancy = @import("occupancy.zig");
pub const fusion = @import("fusion.zig");
pub const execution_coordinator = @import("execution_coordinator.zig");
pub const memory_pool_advanced = @import("memory/pool.zig");
pub const memory_pool_lockfree = @import("memory/lockfree.zig");
pub const sync_event = @import("sync_event.zig");
pub const kernel_ring = @import("kernel_ring.zig");
pub const adaptive_tiling = @import("adaptive_tiling.zig");

// std.gpu integration (Zig 0.16+ native GPU support)
pub const std_gpu = @import("std_gpu.zig");
pub const std_gpu_kernels = @import("std_gpu_kernels.zig");

// Unified API modules
pub const unified = @import("unified.zig");
pub const unified_buffer = @import("unified_buffer.zig");
pub const device = @import("device.zig");
pub const stream = @import("stream.zig");
pub const dsl = @import("dsl/mod.zig");
pub const runtime = @import("runtime/mod.zig");
pub const devices = @import("device/mod.zig");
pub const policy = @import("policy/mod.zig");
pub const multi = @import("multi/mod.zig");
pub const factory = @import("factory/mod.zig");

// Backend interface and loaders
pub const interface = @import("interface.zig");
pub const cuda_loader = if (backend_shared.dynlibSupported)
    @import("backends/cuda/loader.zig")
else
    struct {
        pub const CuResult = enum(i32) { success = 0, _ };
        pub const CoreFunctions = struct {
            cuInit: ?*const fn (u32) callconv(.c) CuResult = null,
            cuDeviceGetCount: ?*const fn (*i32) callconv(.c) CuResult = null,
        };
        pub const CudaFunctions = struct {
            core: CoreFunctions = .{},
        };
        pub fn load(_: std.mem.Allocator) error{PlatformNotSupported}!*const CudaFunctions {
            return error.PlatformNotSupported;
        }
        pub fn unload() void {}
        pub fn getFunctions() ?*const CudaFunctions {
            return null;
        }
        pub fn isAvailableWithAlloc(_: std.mem.Allocator) bool {
            return false;
        }
    };

// Platform detection
pub const platform = @import("platform.zig");

// Modular backend abstraction layer
pub const backends = @import("backends/mod.zig");
pub const dispatch = @import("dispatch/mod.zig");
pub const builtin_kernels = @import("builtin_kernels.zig");

// Recovery and failover
pub const recovery = @import("recovery.zig");
pub const failover = @import("failover.zig");
pub const failover_types = @import("failover_types.zig");

// Diagnostics and error handling
pub const diagnostics = @import("diagnostics.zig");
pub const error_handling = @import("error_handling.zig");

// Multi-device and peer transfer
pub const multi_device = @import("multi_device.zig");
pub const peer_transfer = @import("peer_transfer/mod.zig");

// Mega GPU orchestration (cross-backend coordinator)
pub const mega = @import("mega/mod.zig");

// Include test modules in test builds
comptime {
    if (@import("builtin").is_test) {
        _ = @import("tests/device_enumeration_test.zig");
        _ = @import("tests/backend_detection_test.zig");
        _ = @import("tests/std_gpu_test.zig");
        _ = @import("tests/execution_fallback_test.zig");
        _ = @import("tests/integration_test.zig");
        _ = @import("tests/all_backends_test.zig");
        _ = @import("peer_transfer/tests.zig");
        _ = @import("peer_transfer/network.zig");
        // Performance optimization module tests
        _ = @import("sync_event.zig");
        _ = @import("kernel_ring.zig");
        _ = @import("adaptive_tiling.zig");
        _ = @import("memory/lockfree.zig");
        // std.gpu integration tests
        _ = @import("std_gpu.zig");
        _ = @import("std_gpu_kernels.zig");
        // Shared failover types tests
        _ = @import("failover_types.zig");
        // Mega GPU orchestration tests
        _ = @import("mega/mod.zig");
        // Platform detection tests
        _ = @import("platform.zig");
        // Built-in kernel tests (includes linalg, reduction, etc.)
        _ = @import("builtin_kernels.zig");
        // Extracted submodule tests
        _ = @import("dispatcher_test.zig");
        _ = @import("multi_device_test.zig");
        // Extracted type modules (compile-check)
        _ = @import("dispatch/types.zig");
        _ = @import("dispatch/batch.zig");
        _ = @import("device_group.zig");
        _ = @import("gpu_cluster.zig");
        _ = @import("gradient_sync.zig");
        // Backend extracted tests
        _ = @import("backends/metal_test.zig");
        _ = @import("backends/vulkan_test.zig");
    }
}

const build_options = @import("build_options");

// Import lifecycle management from shared utils
const lifecycle = @import("../../services/shared/utils.zig");
const SimpleModuleLifecycle = lifecycle.SimpleModuleLifecycle;
const LifecycleError = lifecycle.LifecycleError;

var gpu_lifecycle = SimpleModuleLifecycle{};

var cuda_backend_init_lock = sync.Mutex{};
var cuda_backend_initialized = false;
var cached_gpu_allocator: ?std.mem.Allocator = null;

pub const BufferFlags = memory.BufferFlags;
pub const GpuBuffer = memory.GpuBuffer;
pub const Buffer = GpuBuffer; // Alias for convenience
pub const MemoryError = memory.MemoryError;
pub const KernelError = interface.KernelError;
pub const GpuError = memory.MemoryError || error{GpuDisabled};
pub const Error = GpuError;

pub const Stream = kernels.Stream;

pub const Backend = backend.Backend;
pub const isEnabled = backend.isEnabled;

// ============================================================================
// Unified API Exports (essential shared types only)
// ============================================================================

pub const Gpu = unified.Gpu;
pub const GpuConfig = unified.GpuConfig;
pub const GpuDevice = unified.GpuDevice;
pub const ExecutionResult = unified.ExecutionResult;
pub const LaunchConfig = unified.LaunchConfig;
pub const HealthStatus = unified.HealthStatus;

pub const UnifiedBuffer = unified_buffer.Buffer;
pub const BufferOptions = unified_buffer.BufferOptions;

pub const Device = device.Device;
pub const DeviceType = device.DeviceType;

pub const StreamOptions = stream.StreamOptions;
pub const Event = stream.Event;
pub const EventOptions = stream.EventOptions;

pub const KernelBuilder = dsl.KernelBuilder;

pub fn init(allocator: std.mem.Allocator) GpuError!void {
    if (!backend.moduleEnabled()) return error.GpuDisabled;

    cached_gpu_allocator = allocator;
    gpu_lifecycle.init(initCudaComponents) catch {
        return error.GpuDisabled;
    };
}

fn initCudaComponents() !void {
    if (comptime build_options.gpu_cuda and backend_shared.dynlibSupported) {
        cuda_backend_init_lock.lock();
        defer cuda_backend_init_lock.unlock();

        if (!cuda_backend_initialized) {
            const cuda_module = @import("backends/cuda/mod.zig");
            const allocator = cached_gpu_allocator orelse return error.OutOfMemory;

            cuda_module.init(allocator) catch |err| {
                std.log.warn("CUDA backend initialization failed: {t}. Using fallback mode.", .{err});
            };

            if (comptime build_options.enable_gpu) {
                const cuda_stream = @import("backends/cuda/stream.zig");
                cuda_stream.init() catch |err| {
                    std.log.warn("CUDA stream initialization failed: {t}", .{err});
                };

                const cuda_memory = @import("backends/cuda/memory.zig");
                cuda_memory.init(allocator) catch |err| {
                    std.log.warn("CUDA memory initialization failed: {t}", .{err});
                };
            }

            cuda_backend_initialized = true;
        }
    }
}

fn deinitCudaComponents() void {
    if (cuda_backend_initialized) {
        if (comptime build_options.gpu_cuda and backend_shared.dynlibSupported) {
            const cuda_module = @import("backends/cuda/mod.zig");
            cuda_module.deinit();

            if (comptime build_options.enable_gpu) {
                const cuda_stream = @import("backends/cuda/stream.zig");
                cuda_stream.deinit();

                const cuda_memory = @import("backends/cuda/memory.zig");
                cuda_memory.deinit();
            }
        }
        cuda_backend_initialized = false;
    }
}

pub fn ensureInitialized(allocator: std.mem.Allocator) GpuError!void {
    if (!isInitialized()) {
        try init(allocator);
    }
}

pub fn deinit() void {
    deinitCudaComponents();
    gpu_lifecycle.deinit(null);
}

pub fn isInitialized() bool {
    return gpu_lifecycle.isInitialized();
}

// ============================================================================
// Context - Framework Integration
// ============================================================================

const config_module = @import("../../core/config/mod.zig");

/// GPU Context for Framework integration.
///
/// The Context struct wraps the `Gpu` struct to provide a consistent interface
/// with other framework modules. It handles configuration translation and
/// provides convenient access to GPU operations.
///
/// ## Thread Safety
///
/// The Context itself is not thread-safe. For concurrent GPU operations,
/// use the underlying Gpu's stream-based operations or external synchronization.
///
/// ## Example
///
/// ```zig
/// var ctx = try Context.init(allocator, .{ .backend = .vulkan });
/// defer ctx.deinit();
///
/// // Get the underlying Gpu instance
/// const gpu = ctx.getGpu();
///
/// // Create and use buffers
/// var buffer = try ctx.createBuffer(f32, 1024, .{});
/// defer ctx.destroyBuffer(&buffer);
/// ```
pub const Context = struct {
    /// Memory allocator for GPU operations.
    allocator: std.mem.Allocator,
    /// The underlying unified GPU instance.
    gpu: Gpu,

    /// Initialize the GPU context with the given configuration.
    ///
    /// ## Parameters
    ///
    /// - `allocator`: Memory allocator for GPU resources
    /// - `cfg`: GPU configuration (backend selection, memory limits, etc.)
    ///
    /// ## Returns
    ///
    /// A pointer to the initialized Context.
    ///
    /// ## Errors
    ///
    /// - `error.GpuDisabled`: GPU feature is disabled at compile time
    /// - `error.NoDeviceAvailable`: No compatible GPU device found
    /// - `error.OutOfMemory`: Memory allocation failed
    pub fn init(allocator: std.mem.Allocator, cfg: config_module.GpuConfig) !*Context {
        if (!backend.moduleEnabled()) return error.GpuDisabled;

        // Convert config_module.GpuConfig to unified.GpuConfig
        const preferred_backend: ?Backend = switch (cfg.backend) {
            .auto => null,
            .cuda => .cuda,
            .vulkan => .vulkan,
            .stdgpu => .stdgpu,
            .metal => .metal,
            .webgpu => .webgpu,
            .opengl => .opengl,
            .opengles => .opengles,
            .webgl2 => .webgl2,
            .fpga => .fpga,
            .tpu => .tpu,
            .cpu => .stdgpu, // CPU fallback uses stdgpu backend
        };

        const gpu_config = GpuConfig{
            .preferred_backend = preferred_backend,
            .allow_fallback = true,
            .max_memory_bytes = cfg.memory_limit orelse 0,
            .enable_profiling = false,
        };

        const ctx = try allocator.create(Context);
        errdefer allocator.destroy(ctx);

        ctx.* = .{
            .allocator = allocator,
            .gpu = try Gpu.init(allocator, gpu_config),
        };
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        self.gpu.deinit();
        self.allocator.destroy(self);
    }

    /// Get the underlying Gpu instance.
    pub fn getGpu(self: *Context) Error!*Gpu {
        return &self.gpu;
    }

    /// Create a buffer.
    pub fn createBuffer(self: *Context, comptime T: type, count: usize, options: BufferOptions) !UnifiedBuffer {
        return self.gpu.createBuffer(T, count, options);
    }

    /// Create a buffer from a slice.
    pub fn createBufferFromSlice(self: *Context, comptime T: type, data: []const T, options: BufferOptions) !UnifiedBuffer {
        return self.gpu.createBufferFromSlice(T, data, options);
    }

    /// Destroy a buffer.
    pub fn destroyBuffer(self: *Context, buffer: *UnifiedBuffer) void {
        self.gpu.destroyBuffer(buffer);
    }

    /// Vector addition.
    pub fn vectorAdd(self: *Context, a: *UnifiedBuffer, b: *UnifiedBuffer, result: *UnifiedBuffer) !ExecutionResult {
        return self.gpu.vectorAdd(a, b, result);
    }

    /// Matrix multiplication.
    pub fn matrixMultiply(self: *Context, a: *UnifiedBuffer, b: *UnifiedBuffer, result: *UnifiedBuffer, dims: unified.MatrixDims) !ExecutionResult {
        return self.gpu.matrixMultiply(a, b, result, dims);
    }

    /// Get GPU health status.
    pub fn getHealth(self: *Context) !HealthStatus {
        return self.gpu.getHealth();
    }
};

// ============================================================================
// Inline Tests
// ============================================================================

test "gpu module enabled status" {
    try std.testing.expect(backend.moduleEnabled());
    try std.testing.expect(isEnabled(.simulated));
}

test "gpu context init and deinit" {
    const allocator = std.testing.allocator;
    const cfg = config_module.GpuConfig{
        .backend = .auto,
        .memory_limit = null,
    };
    const ctx = Context.init(allocator, cfg) catch return error.SkipZigTest;
    defer ctx.deinit();
    try std.testing.expect(@intFromPtr(ctx) != 0);
}

test "gpu health status with simulated backend" {
    const allocator = std.testing.allocator;
    const gpu_config = GpuConfig{
        .preferred_backend = .simulated,
        .allow_fallback = true,
    };
    var gpu = Gpu.init(allocator, gpu_config) catch return error.SkipZigTest;
    defer gpu.deinit();
    // Verify GPU initialized successfully with simulated backend
    try std.testing.expect(gpu.config.preferred_backend == .simulated);
}

test "gpu backend enum completeness" {
    // Verify that simulated backend is always selectable (compile-time check)
    const simulated: Backend = .simulated;
    try std.testing.expect(simulated == .simulated);
    // Verify all known backends are representable
    const all_backends = [_]Backend{ .cuda, .vulkan, .metal, .webgpu, .opengl, .stdgpu, .simulated };
    try std.testing.expect(all_backends.len >= 7);
}

test "gpu type exports" {
    // Verify key types are accessible (compile-time check)
    _ = GpuConfig{};
    _ = BufferOptions{};
    _ = unified.MatrixDims{ .m = 1, .n = 1, .k = 1 };
    try std.testing.expect(true);
}
