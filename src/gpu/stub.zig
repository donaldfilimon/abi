//! GPU Stub Module
//!
//! Provides stub implementations when GPU is disabled at compile time.
//! All operations return error.GpuDisabled.

const std = @import("std");
const config_module = @import("../config.zig");

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
};

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
    available: bool,
};

pub const BackendAvailability = enum { available, unavailable };
pub const DetectionLevel = enum { none, basic, full };

pub const Device = struct {};
pub const DeviceType = enum { cpu, gpu, accelerator };
pub const DeviceInfo = struct {
    name: []const u8 = "disabled",
    device_type: DeviceType = .cpu,
};
pub const DeviceCapability = struct {};
pub const DeviceFeature = enum { compute, graphics };
pub const DeviceSelector = struct {};
pub const DeviceManager = struct {};

pub const Buffer = struct {};
pub const UnifiedBuffer = struct {};
pub const BufferFlags = packed struct { read: bool = true, write: bool = true };
pub const BufferOptions = struct {};
pub const BufferView = struct {};
pub const BufferStats = struct {};
pub const MappedBuffer = struct {};

pub const MemoryPool = struct {};
pub const MemoryStats = struct {};
pub const MemoryInfo = struct {};
pub const MemoryMode = enum { device, host, managed };
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
    success: bool = false,
    execution_time_ns: u64 = 0,
};
pub const ExecutionStats = struct {};
pub const HealthStatus = enum { healthy, degraded, unhealthy };
pub const GpuStats = struct {};
pub const MatrixDims = struct { m: usize = 0, n: usize = 0, k: usize = 0 };
pub const MultiGpuConfig = struct {};
pub const LoadBalanceStrategy = enum { round_robin, least_loaded };

pub const Profiler = struct {};
pub const TimingResult = struct {};
pub const OccupancyResult = struct {};
pub const MemoryBandwidth = struct {};

pub const Accelerator = struct {};
pub const AcceleratorConfig = struct {};
pub const ComputeTask = struct {};

pub const recovery = struct {};
pub const failover = struct {};
pub const RecoveryManager = struct {};
pub const FailoverManager = struct {};

pub const Gpu = struct {
    pub fn init(_: std.mem.Allocator, _: GpuConfig) Error!Gpu {
        return error.GpuDisabled;
    }
    pub fn deinit(_: *Gpu) void {}
};

pub const GpuConfig = struct {
    backend: Backend = .auto,
};

// ============================================================================
// Context - Stub implementation
// ============================================================================

pub const Context = struct {
    pub fn init(_: std.mem.Allocator, _: config_module.GpuConfig) Error!*Context {
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

pub fn isEnabled() bool {
    return false;
}

pub fn isInitialized() bool {
    return false;
}

pub fn init(_: std.mem.Allocator) Error!void {
    return error.GpuDisabled;
}

pub fn deinit() void {}

pub fn isGpuAvailable() bool {
    return false;
}

pub fn getAvailableBackends() []const Backend {
    return &.{};
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

pub fn defaultDevice() ?DeviceInfo {
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
