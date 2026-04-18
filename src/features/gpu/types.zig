//! Shared types for the gpu feature.
//!
//! Both `mod.zig` (real implementation) and `stub.zig` (disabled no-op)
//! import from here so that type definitions are not duplicated.

const std = @import("std");
<<<<<<< Updated upstream
const backend_mod = @import("internal/backend.zig");
const buffer_mod = @import("internal/unified_buffer.zig");
const device_manager = @import("internal/device_manager.zig");
=======
<<<<<<< HEAD
const backend_mod = @import("backend.zig");
const buffer_mod = @import("unified_buffer.zig");
const device_manager = @import("device_manager.zig");
=======
const backend_mod = @import("backend.zig");
const buffer_mod = @import("unified_buffer.zig");
const device_manager = @import("device_manager.zig");
>>>>>>> 0aad638 (chore: consolidate local main workspace and resolve merge conflicts)
>>>>>>> Stashed changes

// ── Error Sets ─────────────────────────────────────────────────────────────

/// Errors returned by GPU memory operations.
pub const MemoryError = error{
    OutOfMemory,
    InvalidPointer,
    BufferTooSmall,
    HostAccessDisabled,
    DeviceMemoryMissing,
    SizeMismatch,
    InvalidOffset,
    TransferFailed,
};

/// Errors returned by GPU kernel operations.
pub const KernelError = error{
    CompilationFailed,
    InvalidKernel,
    InvalidArgument,
    LaunchFailed,
    BackendUnsupported,
};

/// GPU feature errors visible at the framework level.
pub const FrameworkError = error{
    GpuDisabled,
    NoDeviceAvailable,
    InvalidConfig,
    KernelCompilationFailed,
    KernelExecutionFailed,
} || MemoryError || KernelError;

/// Unified GPU error type combining memory errors and a disabled sentinel.
pub const GpuError = FrameworkError;

/// Convenience alias for the primary GPU error set.
pub const Error = GpuError;

/// Errors returned when selecting a GPU backend.
pub const BackendSelectionError = error{
    RequestedBackendUnavailable,
    NoBackendsAvailable,
    OutOfMemory,
};

// ── Aggregate Stats / Info ─────────────────────────────────────────────────

/// Summary of GPU memory usage.
pub const MemoryInfo = struct {
    total_bytes: u64 = 0,
    used_bytes: u64 = 0,
    free_bytes: u64 = 0,
    peak_used_bytes: u64 = 0,
};

/// Aggregate GPU execution statistics.
pub const GpuStats = struct {
    kernels_launched: u64 = 0,
    buffers_created: u64 = 0,
    bytes_allocated: u64 = 0,
    host_to_device_transfers: u64 = 0,
    device_to_host_transfers: u64 = 0,
    total_execution_time_ns: u64 = 0,
};

/// Aggregate metrics summary for GPU operations.
pub const MetricsSummary = struct {
    total_kernel_invocations: u64 = 0,
    avg_kernel_time_ns: f64 = 0,
    kernels_per_second: f64 = 0,
};

// ── Stub-only types (used by stub.zig when GPU is disabled) ────────────────

pub const Backend = backend_mod.Backend;

pub const Device = struct {
    id: u32 = 0,
    backend: backend_mod.Backend = .stdgpu,
    name: []const u8 = "disabled",
};
pub const DeviceType = enum { cpu, gpu, accelerator };

pub const Buffer = struct {
    pub fn read(_: *Buffer, comptime _: type, _: anytype) Error!void {
        return error.GpuDisabled;
    }
    pub fn readBytes(_: *Buffer, _: []u8) Error!void {
        return error.GpuDisabled;
    }
    pub fn write(_: *Buffer, comptime _: type, _: anytype) Error!void {
        return error.GpuDisabled;
    }
    pub fn writeBytes(_: *Buffer, _: []const u8) Error!void {
        return error.GpuDisabled;
    }
    pub fn size(_: *const Buffer) usize {
        return 0;
    }
    pub fn deinit(_: *Buffer) void {}
};

pub const UnifiedBuffer = struct {
    pub fn read(_: *UnifiedBuffer, comptime _: type, _: anytype) Error!void {
        return error.GpuDisabled;
    }
    pub fn write(_: *UnifiedBuffer, comptime _: type, _: anytype) Error!void {
        return error.GpuDisabled;
    }
    pub fn size(_: *const UnifiedBuffer) usize {
        return 0;
    }
    pub fn deinit(_: *UnifiedBuffer) void {}
};

pub const BufferFlags = packed struct { read: bool = true, write: bool = true };
pub const BufferOptions = struct {};

pub const Stream = struct {};
pub const StreamOptions = struct {};
pub const Event = struct {};
pub const EventOptions = struct {};

pub const LaunchConfig = struct {};
pub const ExecutionResult = struct {
    execution_time_ns: u64 = 0,
    elements_processed: usize = 0,
    bytes_transferred: usize = 0,
    backend: backend_mod.Backend = .stdgpu,
    device_id: u32 = 0,
};
pub const HealthStatus = enum { healthy, degraded, unhealthy, unknown };
pub const MatrixDims = struct { m: usize = 0, n: usize = 0, k: usize = 0 };

pub const KernelBuilder = struct {};

pub const GpuConfig = struct {
    preferred_backend: ?backend_mod.Backend = null,
    allow_fallback: bool = true,
    memory_mode: buffer_mod.MemoryMode = .automatic,
    max_memory_bytes: usize = 0,
    enable_profiling: bool = false,
    multi_gpu: bool = false,
    load_balance_strategy: device_manager.LoadBalanceStrategy = .memory_aware,
};
