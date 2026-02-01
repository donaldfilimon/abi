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
// Local Stubs Imports
// ============================================================================

pub const backend = @import("stubs/backend.zig");
pub const device = @import("stubs/device.zig");
pub const memory = @import("stubs/memory.zig");
pub const stream = @import("stubs/stream.zig");
pub const kernel = @import("stubs/kernel.zig");
pub const dsl_mod = @import("stubs/dsl.zig");
pub const execution = @import("stubs/execution.zig");
pub const profiler = @import("stubs/profiler.zig");
pub const recovery_mod = @import("stubs/recovery.zig");
pub const multi_gpu = @import("stubs/multi_gpu.zig");
pub const config = @import("stubs/config.zig");

// ============================================================================
// Re-exports
// ============================================================================

pub const Backend = backend.Backend;
pub const BackendInfo = backend.BackendInfo;
pub const DetectionLevel = backend.DetectionLevel;
pub const BackendAvailability = backend.BackendAvailability;
pub const Summary = backend.Summary;

pub const Device = device.Device;
pub const DeviceType = device.DeviceType;
pub const DeviceInfo = device.DeviceInfo;
pub const DeviceCapability = device.DeviceCapability;
pub const DeviceFeature = device.DeviceFeature;
pub const DeviceSelector = device.DeviceSelector;
pub const DeviceManager = device.DeviceManager;

pub const Buffer = memory.Buffer;
pub const UnifiedBuffer = memory.UnifiedBuffer;
pub const BufferFlags = memory.BufferFlags;
pub const BufferOptions = memory.BufferOptions;
pub const BufferView = memory.BufferView;
pub const BufferStats = memory.BufferStats;
pub const MappedBuffer = memory.MappedBuffer;
pub const MemoryPool = memory.MemoryPool;
pub const MemoryStats = memory.MemoryStats;
pub const MemoryInfo = memory.MemoryInfo;
pub const MemoryMode = memory.MemoryMode;
pub const MemoryLocation = memory.MemoryLocation;

pub const Stream = stream.Stream;
pub const StreamOptions = stream.StreamOptions;
pub const StreamPriority = stream.StreamPriority;
pub const StreamFlags = stream.StreamFlags;
pub const StreamState = stream.StreamState;
pub const StreamManager = stream.StreamManager;
pub const Event = stream.Event;
pub const EventOptions = stream.EventOptions;
pub const EventFlags = stream.EventFlags;
pub const EventState = stream.EventState;

pub const KernelBuilder = kernel.KernelBuilder;
pub const KernelIR = kernel.KernelIR;
pub const KernelSource = kernel.KernelSource;
pub const KernelConfig = kernel.KernelConfig;
pub const CompiledKernel = kernel.CompiledKernel;
pub const KernelCache = kernel.KernelCache;
pub const KernelCacheConfig = kernel.KernelCacheConfig;
pub const CacheStats = kernel.CacheStats;
pub const PortableKernelSource = kernel.PortableKernelSource;

pub const dsl = dsl_mod.dsl;
pub const ScalarType = dsl_mod.ScalarType;
pub const VectorType = dsl_mod.VectorType;
pub const MatrixType = dsl_mod.MatrixType;
pub const AddressSpace = dsl_mod.AddressSpace;
pub const DslType = dsl_mod.DslType;
pub const AccessMode = dsl_mod.AccessMode;
pub const Expr = dsl_mod.Expr;
pub const BinaryOp = dsl_mod.BinaryOp;
pub const UnaryOp = dsl_mod.UnaryOp;
pub const BuiltinFn = dsl_mod.BuiltinFn;
pub const BuiltinVar = dsl_mod.BuiltinVar;
pub const Stmt = dsl_mod.Stmt;
pub const GeneratedSource = dsl_mod.GeneratedSource;
pub const CompileOptions = dsl_mod.CompileOptions;

pub const LaunchConfig = execution.LaunchConfig;
pub const ExecutionResult = execution.ExecutionResult;
pub const ExecutionStats = execution.ExecutionStats;
pub const HealthStatus = execution.HealthStatus;
pub const GpuStats = execution.GpuStats;
pub const MatrixDims = execution.MatrixDims;
pub const MultiGpuConfig = execution.MultiGpuConfig;
pub const LoadBalanceStrategy = execution.LoadBalanceStrategy;
pub const ReduceResult = execution.ReduceResult;
pub const DotProductResult = execution.DotProductResult;

pub const Profiler = profiler.Profiler;
pub const TimingResult = profiler.TimingResult;
pub const OccupancyResult = profiler.OccupancyResult;
pub const MemoryBandwidth = profiler.MemoryBandwidth;
pub const MetricsSummary = profiler.MetricsSummary;
pub const KernelMetrics = profiler.KernelMetrics;
pub const MetricsCollector = profiler.MetricsCollector;

pub const recovery = recovery_mod.recovery;
pub const failover = recovery_mod.failover;
pub const RecoveryManager = recovery_mod.RecoveryManager;
pub const FailoverManager = recovery_mod.FailoverManager;

pub const DeviceGroup = multi_gpu.DeviceGroup;
pub const WorkDistribution = multi_gpu.WorkDistribution;
pub const GroupStats = multi_gpu.GroupStats;

pub const GpuConfig = config.GpuConfig;

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
