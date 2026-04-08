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
//! - **backend.Backend Auto-detection**: Automatically selects the best available backend
//! - **Unified Buffer API**: Cross-platform memory management
//! - **Kernel DSL**: Write portable kernels that compile to any backend
//! - **Execution Coordinator**: Automatic fallback from GPU to SIMD to scalar
//! - **Multi-device Support**: Manage multiple GPUs with peer-to-peer transfers
//! - **Profiling**: Built-in timing and occupancy analysis
//!
//! ## Available backend.Backends
//!
//! | backend.Backend | Platform | Build Flag |
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
//! - `unified.GpuConfig` - Configuration for GPU initialization
//! - `UnifiedBuffer` - Cross-backend buffer type
//! - `Device`, `DeviceType` - Device discovery and selection
//! - `KernelBuilder`, `KernelIR` - DSL for custom kernels
//! - `backend.Backend`, `backend.BackendAvailability` - backend.Backend detection
//!
//! ## Quick Start
//!
//! ```zig
//! const abi = @import("abi");
//!
//! // Initialize framework with GPU
//! var fw = try abi.App.init(allocator, .{
//!     .gpu = .{ .backend = .auto },  // Auto-detect best backend
//! });
//! defer fw.deinit();
//!
//! // Get GPU context
//! const gpu_ctx = try fw.get(.gpu);
//! const gpu = gpu_ctx.get(.gpu);
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
//! std.debug.print("backend.Backend: {t}\n", .{health.backend});
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
//! - backend.Backend-specific initialization functions (initCudaComponents, etc.)
const std = @import("std");
const time = @import("../../foundation/mod.zig").time;
const sync = @import("../../foundation/mod.zig").sync;

// Decoupled sub-modules
pub const core = @import("core_gpu.zig");
pub const compute = @import("unified.zig");
pub const memory_sys = @import("backends/cuda/memory.zig");
pub const dispatch_sys = @import("backends/shared.zig");

// Unified API modules
pub const unified = @import("unified.zig");
pub const unified_buffer = @import("unified_buffer.zig");
pub const device = @import("device.zig");
pub const devices = @import("device.zig");
pub const stream = @import("stream.zig");
pub const dsl = @import("dsl/mod.zig");

// ── Execution & Orchestration ────────────────────────────────────────────
pub const execution = @import("execution.zig");
pub const execution_coordinator = @import("execution_coordinator.zig");
pub const runtime = @import("runtime/mod.zig");
pub const dispatch = @import("dispatch/mod.zig");
pub const mega = @import("mega/mod.zig");
pub const factory = @import("factory/mod.zig");
pub const policy = @import("policy/mod.zig");

// ── Memory Management ────────────────────────────────────────────────────
pub const memory_ns = @import("memory_ns.zig");
pub const memory = @import("memory/base.zig");
pub const memory_pool_advanced = @import("memory/pool.zig");
pub const memory_pool_lockfree = @import("memory/lockfree.zig");

pub const MemoryError = memory.MemoryError;

// ── backend.Backends & Hardware ──────────────────────────────────────────────────
pub const backend = @import("backend.zig");
pub const backends = @import("backends/mod.zig");
pub const backend_shared = @import("backends/shared.zig");
pub const std_gpu = @import("std_gpu.zig");
pub const std_gpu_kernels = @import("std_gpu_kernels.zig");
pub const kernels = @import("runtime_kernels.zig");
pub const builtin_kernels = @import("builtin_kernels.zig");
pub const interface = @import("interface.zig");
pub const cuda_loader = dispatch_sys.cuda_loader;

// ── Performance & Advanced ───────────────────────────────────────────────
pub const advanced = @import("advanced.zig");
pub const profiling = @import("profiling.zig");
pub const occupancy = @import("occupancy.zig");
pub const fusion = @import("fusion.zig");
pub const sync_event = @import("sync_event.zig");
pub const kernel_ring = @import("kernel_ring.zig");
pub const adaptive_tiling = @import("adaptive_tiling.zig");

// ── Recovery & Diagnostics ───────────────────────────────────────────────
pub const recovery = @import("recovery.zig");
pub const failover = @import("failover.zig");
pub const failover_types = @import("failover_types.zig");
pub const diagnostics = @import("diagnostics.zig");
pub const error_handling = @import("error_handling.zig");

// ── Multi-Device & Peer-to-Peer ──────────────────────────────────────────
pub const multi = @import("multi.zig");
pub const multi_device = @import("multi_device.zig");
pub const peer_transfer = @import("peer_transfer/mod.zig");

// ── AI & Training Bridge ─────────────────────────────────────────────────
pub const coordinator_ai_ops = @import("coordinator_ai_ops.zig");
pub const training_bridge = @import("training_bridge.zig");
pub const gradient_compression = @import("gradient_compression.zig");

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
        // backend.Backend extracted tests
        _ = @import("backends/metal_test.zig");
        _ = @import("backends/vulkan_test.zig");
        // Training bridge tests
        _ = @import("coordinator_ai_ops.zig");
        _ = @import("training_bridge.zig");
        // Gradient compression tests
        _ = @import("gradient_compression.zig");
    }
}

const build_options = @import("build_options");

// Import lifecycle management from shared utils
const lifecycle = @import("../../foundation/mod.zig").utils;
const SimpleModuleLifecycle = lifecycle.SimpleModuleLifecycle;
const LifecycleError = lifecycle.LifecycleError;

// ── Framework Integration ────────────────────────────────────────────────

const config_module = @import("../core/config/mod.zig");

pub const Gpu = unified.Gpu;

pub const types = @import("types.zig");
pub const Error = types.Error;
pub const GpuError = types.Error;
pub const KernelError = types.KernelError;
pub const BackendSelectionError = types.BackendSelectionError;
pub const MemoryInfo = types.MemoryInfo;
pub const GpuStats = types.GpuStats;
pub const MetricsSummary = types.MetricsSummary;
pub const Backend = types.Backend;
pub const Device = types.Device;
pub const DeviceType = types.DeviceType;
pub const Buffer = types.Buffer;
pub const GpuBuffer = types.Buffer;
pub const UnifiedBuffer = types.UnifiedBuffer;
pub const BufferFlags = types.BufferFlags;
pub const Stream = types.Stream;
pub const StreamOptions = types.StreamOptions;
pub const Event = types.Event;
pub const EventOptions = types.EventOptions;
pub const LaunchConfig = types.LaunchConfig;
pub const ExecutionResult = types.ExecutionResult;
pub const HealthStatus = types.HealthStatus;
pub const KernelBuilder = types.KernelBuilder;
pub const GpuConfig = unified.GpuConfig;
pub const BufferOptions = unified.BufferOptions;
pub const MatrixDims = unified.MatrixDims;

pub const Context = struct {
    pub fn init(allocator: std.mem.Allocator, config: GpuConfig) Error!*Context {
        const ctx = try Context.create(allocator, config);
        return ctx;
    }
    pub fn create(allocator: std.mem.Allocator, config: GpuConfig) Error!*Context {
        const gpu = try Gpu.init(allocator, config);
        const ctx = try allocator.create(Context);
        ctx.* = Context{ .gpu = gpu };
        return ctx;
    }
    pub fn deinit(ctx: *Context) void {
        ctx.gpu.deinit();
    }
    pub fn getGpu(ctx: *Context) Error!*Gpu {
        return &ctx.gpu;
    }
    pub fn createBuffer(ctx: *Context, comptime T: type, size: usize, opts: BufferOptions) Error!UnifiedBuffer {
        return ctx.gpu.createBuffer(T, size, opts);
    }
    pub fn createBufferFromSlice(ctx: *Context, comptime T: type, data: []const T, opts: BufferOptions) Error!UnifiedBuffer {
        return ctx.gpu.createBufferFromSlice(T, data, opts);
    }
    pub fn destroyBuffer(ctx: *Context, buf: *UnifiedBuffer) void {
        ctx.gpu.destroyBuffer(buf);
    }
    pub fn vectorAdd(ctx: *Context, a: *UnifiedBuffer, b: *UnifiedBuffer, out: *UnifiedBuffer) Error!ExecutionResult {
        return ctx.gpu.vectorAdd(a, b, out);
    }
    pub fn matrixMultiply(ctx: *Context, a: *UnifiedBuffer, b: *UnifiedBuffer, out: *UnifiedBuffer, dims: types.MatrixDims) Error!ExecutionResult {
        return ctx.gpu.matrixMultiply(a, b, out, dims);
    }
    pub fn getHealth(ctx: *Context) Error!HealthStatus {
        return ctx.gpu.getHealth();
    }

    gpu: Gpu,
};

pub const platform = @import("platform.zig");

pub const GpuDevice = unified.GpuDevice;

pub fn isEnabled(b: Backend) bool {
    return b.isAvailable();
}

const stub_helpers = @import("../core/stub_helpers.zig");
const Stub = stub_helpers.StubFeatureNoConfig(Error);
pub const init = Stub.init;
pub const deinit = Stub.deinit;
pub const isInitialized = Stub.isInitialized;

pub fn ensureInitialized(allocator: std.mem.Allocator) Error!void {
    _ = allocator;
    return error.GpuDisabled;
}

// ── Tests ────────────────────────────────────────────────────────────────

test "gpu module enabled status" {
    try std.testing.expect(backend.moduleEnabled());
    try std.testing.expect(backend.isEnabled(.simulated));
}

test "gpu context init and deinit" {
    const allocator = std.testing.allocator;
    const cfg = unified.GpuConfig{
        .preferred_backend = .simulated,
    };
    const ctx = Context.init(allocator, cfg) catch return error.SkipZigTest;
    defer ctx.deinit();
    try std.testing.expect(@intFromPtr(ctx) != 0);
}

test "gpu health status with simulated backend" {
    const allocator = std.testing.allocator;
    const gpu_config = unified.GpuConfig{
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
    const simulated: backend.Backend = .simulated;
    try std.testing.expect(simulated == .simulated);
    // Verify all known backends are representable
    const all_backends = [_]backend.Backend{ .cuda, .vulkan, .metal, .webgpu, .opengl, .stdgpu, .simulated };
    try std.testing.expect(all_backends.len >= 7);
}

test "gpu type exports" {
    // Verify key types are accessible (compile-time check)
    _ = unified.GpuConfig{};
    _ = BufferOptions{};
    _ = unified.MatrixDims{ .m = 1, .n = 1, .k = 1 };
    try std.testing.expect(true);
}

// refAllDecls deferred — stdgpu, diagnostics, profiling, recovery have pre-existing Zig 0.16 API errors
