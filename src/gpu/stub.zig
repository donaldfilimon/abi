//! GPU Stub Module
//!
//! Provides stub implementations when GPU is disabled at compile time.
//! All operations return error.GpuDisabled.

const std = @import("std");
const config_module = @import("../config/mod.zig");
const stub_common = @import("../shared/stub_common.zig");

// ============================================================================
// Errors
// ============================================================================

pub const Error = error{
    GpuDisabled,
    NoDeviceAvailable,
    InitializationFailed,
    InvalidConfig,
    OutOfMemory,
    KernelCompilationFailed,
    KernelExecutionFailed,
} || stub_common.CommonError;

pub const GpuError = Error;
pub const MemoryError = Error;
pub const KernelError = Error;
pub const AcceleratorError = Error;
pub const CodegenError = Error;
pub const CompileError = Error;

// ============================================================================
// Stub Types
// ============================================================================

pub const Backend = enum {
    auto,
    vulkan,
    cuda,
    metal,
    webgpu,
    opengl,
    cpu,
};

pub const BackendInfo = struct {
    backend: Backend,
    name: []const u8,
    description: []const u8 = "GPU Disabled",
    enabled: bool = false,
    available: bool,
    availability: []const u8 = "disabled",
    device_count: usize = 0,
    build_flag: []const u8 = "",
};

pub const DetectionLevel = enum { none, loader, device_count };

pub const BackendAvailability = struct {
    enabled: bool = false,
    available: bool = false,
    reason: []const u8 = "gpu disabled",
    device_count: usize = 0,
    level: DetectionLevel = .none,
};

pub const Summary = struct {
    module_enabled: bool = false,
    enabled_backend_count: usize = 0,
    available_backend_count: usize = 0,
    device_count: usize = 0,
    emulated_devices: usize = 0,
};

pub const Device = struct {
    id: u32 = 0,
    backend: Backend = .cpu,
    name: []const u8 = "disabled",
};
pub const DeviceType = enum { cpu, gpu, accelerator };
pub const DeviceInfo = struct {
    id: u32 = 0,
    backend: Backend = .cpu,
    name: []const u8 = "disabled",
    total_memory_bytes: ?u64 = null,
    is_emulated: bool = true,
    capability: DeviceCapability = .{},
    device_type: DeviceType = .cpu,
};
pub const DeviceCapability = struct {
    unified_memory: bool = false,
    supports_fp16: bool = false,
    supports_int8: bool = false,
    supports_async_transfers: bool = false,
    max_threads_per_block: ?u32 = null,
    max_shared_memory_bytes: ?u32 = null,
};
pub const DeviceFeature = enum { compute, graphics };
pub const DeviceSelector = struct {};
pub const DeviceManager = struct {};

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
pub const BufferView = struct {};
pub const BufferStats = struct {};
pub const MappedBuffer = struct {};

pub const MemoryPool = struct {};
pub const MemoryStats = struct {};
pub const MemoryInfo = struct {
    used_bytes: usize = 0,
    peak_used_bytes: usize = 0,
    total_bytes: usize = 0,
};
pub const MemoryMode = enum { automatic, explicit, unified };
pub const MemoryLocation = enum { device, host };

pub const Stream = struct {};
pub const StreamOptions = struct {};
pub const StreamPriority = enum { low, normal, high };
pub const StreamFlags = packed struct {};
pub const StreamState = enum { idle, running, @"error" };
pub const StreamManager = struct {};
pub const Event = struct {};
pub const EventOptions = struct {};
pub const EventFlags = packed struct {};
pub const EventState = enum { pending, completed };

pub const KernelBuilder = struct {};
pub const KernelIR = struct {};
pub const KernelSource = struct {};
pub const KernelConfig = struct {};
pub const CompiledKernel = struct {};
pub const KernelCache = struct {};
pub const KernelCacheConfig = struct {};
pub const CacheStats = struct {};
pub const PortableKernelSource = struct {};

pub const dsl = struct {};
pub const ScalarType = enum { f32, f64, i32, i64, u32, u64 };
pub const VectorType = struct {};
pub const MatrixType = struct {};
pub const AddressSpace = enum { global, local, private };
pub const DslType = struct {};
pub const AccessMode = enum { read, write, read_write };
pub const Expr = struct {};
pub const BinaryOp = enum { add, sub, mul, div };
pub const UnaryOp = enum { neg, abs };
pub const BuiltinFn = enum {};
pub const BuiltinVar = enum {};
pub const Stmt = struct {};
pub const GeneratedSource = struct {};
pub const CompileOptions = struct {};

pub const LaunchConfig = struct {};
pub const ExecutionResult = struct {
    execution_time_ns: u64 = 0,
    elements_processed: usize = 0,
    bytes_transferred: usize = 0,
    backend: Backend = .cpu,
    device_id: u32 = 0,
};
pub const ExecutionStats = struct {};
pub const HealthStatus = enum { healthy, degraded, unhealthy };
pub const GpuStats = struct {
    kernels_launched: usize = 0,
    buffers_created: usize = 0,
    bytes_allocated: usize = 0,
    total_execution_time_ns: u64 = 0,
};
pub const MatrixDims = struct { m: usize = 0, n: usize = 0, k: usize = 0 };
pub const MultiGpuConfig = struct {};
pub const LoadBalanceStrategy = enum { round_robin, least_loaded };

pub const Profiler = struct {};
pub const TimingResult = struct {};
pub const OccupancyResult = struct {};
pub const MemoryBandwidth = struct {};

pub const recovery = struct {};
pub const failover = struct {};
pub const RecoveryManager = struct {};
pub const FailoverManager = struct {};

pub const ReduceResult = struct {
    value: f32 = 0.0,
    stats: ExecutionResult = .{},
};

pub const DotProductResult = struct {
    value: f32 = 0.0,
    stats: ExecutionResult = .{},
};

pub const MetricsSummary = struct {
    total_kernel_invocations: usize = 0,
    avg_kernel_time_ns: f64 = 0.0,
    kernels_per_second: f64 = 0.0,
};

pub const DeviceGroup = struct {};
pub const WorkDistribution = struct {};
pub const GroupStats = struct {};
pub const KernelMetrics = struct {};
pub const MetricsCollector = struct {};

pub const Gpu = struct {
    pub fn init(_: std.mem.Allocator, _: GpuConfig) Error!Gpu {
        return stub_common.stubError(error.GpuDisabled);
    }
    pub fn deinit(_: *Gpu) void {}
    pub fn isAvailable(_: *const Gpu) bool {
        return false;
    }
    pub fn getActiveDevice(_: *const Gpu) ?*const Device {
        return null;
    }
    pub fn createBuffer(_: *Gpu, _: usize, _: BufferOptions) Error!*Buffer {
        return error.GpuDisabled;
    }
    pub fn createBufferFromSlice(_: *Gpu, comptime _: type, _: anytype, _: BufferOptions) Error!*Buffer {
        return error.GpuDisabled;
    }
    pub fn destroyBuffer(_: *Gpu, _: *Buffer) void {}
    pub fn vectorAdd(_: *Gpu, _: *Buffer, _: *Buffer, _: *Buffer) Error!ExecutionResult {
        return error.GpuDisabled;
    }
    pub fn reduceSum(_: *Gpu, _: *Buffer) Error!ReduceResult {
        return error.GpuDisabled;
    }
    pub fn dotProduct(_: *Gpu, _: *Buffer, _: *Buffer) Error!DotProductResult {
        return error.GpuDisabled;
    }
    pub fn getStats(_: *const Gpu) GpuStats {
        return .{};
    }
    pub fn getMetricsSummary(_: *const Gpu) ?MetricsSummary {
        return null;
    }
    pub fn getMemoryInfo(_: *const Gpu) MemoryInfo {
        return .{};
    }
    pub fn selectDevice(_: *Gpu, _: DeviceSelector) Error!void {
        return error.GpuDisabled;
    }
    pub fn enableMultiGpu(_: *Gpu, _: MultiGpuConfig) Error!void {
        return error.GpuDisabled;
    }
    pub fn getDeviceGroup(_: *Gpu) ?*DeviceGroup {
        return null;
    }
    pub fn distributeWork(_: *Gpu, _: usize) Error![]WorkDistribution {
        return error.GpuDisabled;
    }
    pub fn compileKernel(_: *Gpu, _: PortableKernelSource) Error!CompiledKernel {
        return error.GpuDisabled;
    }
    pub fn launchKernel(_: *Gpu, _: *const CompiledKernel, _: LaunchConfig, _: anytype) Error!ExecutionResult {
        return error.GpuDisabled;
    }
    pub fn synchronize(_: *Gpu) Error!void {
        return error.GpuDisabled;
    }
    pub fn createStream(_: *Gpu, _: StreamOptions) Error!*Stream {
        return error.GpuDisabled;
    }
    pub fn createEvent(_: *Gpu, _: EventOptions) Error!*Event {
        return error.GpuDisabled;
    }
    pub fn isProfilingEnabled(_: *const Gpu) bool {
        return false;
    }
    pub fn enableProfiling(_: *Gpu) void {}
    pub fn disableProfiling(_: *Gpu) void {}
    pub fn getKernelMetrics(_: *Gpu, _: []const u8) ?KernelMetrics {
        return null;
    }
    pub fn getMetricsCollector(_: *Gpu) ?*MetricsCollector {
        return null;
    }
    pub fn resetMetrics(_: *Gpu) void {}
    pub fn isMultiGpuEnabled(_: *const Gpu) bool {
        return false;
    }
    pub fn getMultiGpuStats(_: *const Gpu) ?GroupStats {
        return null;
    }
    pub fn activeDeviceCount(_: *const Gpu) usize {
        return 0;
    }
    pub fn softmax(_: *Gpu, _: *Buffer, _: *Buffer) Error!ExecutionResult {
        return error.GpuDisabled;
    }
    pub fn checkHealth(_: *const Gpu) HealthStatus {
        return .unhealthy;
    }
};

pub const GpuConfig = struct {
    backend: Backend = .auto,
    enable_profiling: bool = false,
    memory_mode: MemoryMode = .automatic,
};

// ============================================================================
// Context - Stub implementation
// ============================================================================

pub const Context = struct {
    pub fn init(_: std.mem.Allocator, _: config_module.GpuConfig) !*Context {
        return error.GpuDisabled;
    }

    pub fn deinit(_: *Context) void {}

    pub fn getGpu(_: *Context) Error!*Gpu {
        return error.GpuDisabled;
    }

    pub fn createBuffer(_: *Context, comptime _: type, _: usize, _: BufferOptions) Error!UnifiedBuffer {
        return error.GpuDisabled;
    }

    pub fn createBufferFromSlice(_: *Context, comptime T: type, _: []const T, _: BufferOptions) Error!UnifiedBuffer {
        return error.GpuDisabled;
    }

    pub fn destroyBuffer(_: *Context, _: *UnifiedBuffer) void {}

    pub fn vectorAdd(_: *Context, _: *UnifiedBuffer, _: *UnifiedBuffer, _: *UnifiedBuffer) Error!ExecutionResult {
        return error.GpuDisabled;
    }

    pub fn matrixMultiply(_: *Context, _: *UnifiedBuffer, _: *UnifiedBuffer, _: *UnifiedBuffer, _: MatrixDims) Error!ExecutionResult {
        return error.GpuDisabled;
    }

    pub fn getHealth(_: *Context) Error!HealthStatus {
        return error.GpuDisabled;
    }
};

// ============================================================================
// Module-level stub functions
// ============================================================================

pub fn isEnabled(_: Backend) bool {
    return false;
}

pub fn isInitialized() bool {
    return false;
}

pub fn init(_: std.mem.Allocator) Error!void {
    return error.GpuDisabled;
}

pub fn deinit() void {}

pub fn ensureInitialized(_: std.mem.Allocator) Error!void {
    return error.GpuDisabled;
}

pub fn isGpuAvailable() bool {
    return false;
}

pub fn getAvailableBackends() []const Backend {
    return &.{};
}

pub fn availableBackends(allocator: std.mem.Allocator) Error![]Backend {
    _ = allocator;
    return error.GpuDisabled;
}

pub fn getBestBackend() Backend {
    return .cpu;
}

pub fn listBackendInfo(_: std.mem.Allocator) Error![]BackendInfo {
    return error.GpuDisabled;
}

pub fn listDevices(_: std.mem.Allocator) Error![]DeviceInfo {
    return error.GpuDisabled;
}

pub fn defaultDevice(_: std.mem.Allocator) !?DeviceInfo {
    return null;
}

pub fn discoverDevices(_: std.mem.Allocator) Error![]Device {
    return error.GpuDisabled;
}

pub fn backendName(_: Backend) []const u8 {
    return "disabled";
}

pub fn backendDisplayName(_: Backend) []const u8 {
    return "GPU Disabled";
}

pub fn backendDescription(_: Backend) []const u8 {
    return "GPU feature is disabled at compile time";
}

pub fn moduleEnabled() bool {
    return false;
}

pub fn createDefaultKernels(_: std.mem.Allocator) Error!void {
    return error.GpuDisabled;
}

pub fn compileKernel(_: KernelSource, _: KernelConfig) Error!CompiledKernel {
    return error.GpuDisabled;
}

pub fn backendAvailability(_: Backend) BackendAvailability {
    return .{
        .enabled = false,
        .available = false,
        .reason = "gpu module disabled",
        .device_count = 0,
        .level = .none,
    };
}

pub fn summary() Summary {
    return .{
        .module_enabled = false,
        .enabled_backend_count = 0,
        .available_backend_count = 0,
        .device_count = 0,
        .emulated_devices = 0,
    };
}
