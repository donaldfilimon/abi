//! GPU Stub Module — API-compatible no-ops when GPU is disabled at compile time.

const std = @import("std");
const stub_helpers = @import("../core/stub_helpers.zig");
const config_module = @import("../core/config/mod.zig");

// ── Shared types (re-exported) ─────────────────────────────────────────────

pub const types = @import("types.zig");

pub const Error = error{ FeatureDisabled, NoDeviceAvailable, InitializationFailed, InvalidConfig, OutOfMemory, KernelCompilationFailed, KernelExecutionFailed };
pub const GpuError = Error;
pub const MemoryError = types.MemoryError;
pub const KernelError = types.KernelError;
pub const BackendSelectionError = types.BackendSelectionError;
pub const MemoryInfo = types.MemoryInfo;
pub const GpuStats = types.GpuStats;
pub const MetricsSummary = types.MetricsSummary;

pub const Backend = types.Backend;
pub const Device = types.Device;
pub const DeviceType = types.DeviceType;
pub const Buffer = types.Buffer;
pub const GpuBuffer = Buffer;
pub const UnifiedBuffer = types.UnifiedBuffer;
pub const BufferFlags = types.BufferFlags;
pub const BufferOptions = types.BufferOptions;
pub const Stream = types.Stream;
pub const StreamOptions = types.StreamOptions;
pub const Event = types.Event;
pub const EventOptions = types.EventOptions;
pub const LaunchConfig = types.LaunchConfig;
pub const ExecutionResult = types.ExecutionResult;
pub const HealthStatus = types.HealthStatus;
pub const KernelBuilder = types.KernelBuilder;
pub const GpuConfig = types.GpuConfig;

// ── Sub-module namespace stubs ─────────────────────────────────────────────

<<<<<<< Updated upstream
pub const core_gpu = struct {};
pub const execution = struct {};
pub const memory_ns = struct {};
pub const advanced = struct {};

pub const backend = struct {};
pub const kernels = struct {};
pub const memory = struct {};
pub const backend_shared = struct {};
=======
pub const core = struct {};
pub const compute = struct {};
pub const memory_sys = struct {};
pub const dispatch_sys = struct {};
>>>>>>> Stashed changes
pub const profiling = struct {};
pub const occupancy = struct {};
pub const fusion = struct {};
pub const execution_coordinator = struct {};
pub const memory_pool_advanced = struct {};
pub const memory_pool_lockfree = struct {};
pub const sync_event = struct {};
pub const kernel_ring = struct {};
pub const adaptive_tiling = struct {};
pub const std_gpu = struct {};
pub const std_gpu_kernels = struct {};
pub const unified = struct {};
pub const unified_buffer = struct {};
pub const device = struct {};
pub const stream = struct {};
pub const dsl = struct {};
pub const runtime = struct {};
pub const devices = struct {};
pub const policy = struct {};
pub const multi = struct {};
pub const factory = struct {};
pub const interface = struct {};
pub const cuda_loader = struct {};
pub const platform = struct {};
pub const backends = struct {};
pub const dispatch = struct {};
pub const builtin_kernels = struct {};
pub const recovery = struct {};
pub const failover = struct {};
pub const failover_types = struct {};
pub const diagnostics = struct {};
pub const error_handling = struct {};
pub const multi_device = struct {};
pub const peer_transfer = struct {};
pub const mega = struct {};
pub const coordinator_ai_ops = struct {};
pub const training_bridge = struct {};
pub const gradient_compression = struct {};

// ── Gpu struct ─────────────────────────────────────────────────────────────

pub const Gpu = struct {
    config: GpuConfig = .{},

    pub fn init(_: std.mem.Allocator, _: GpuConfig) Error!Gpu {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *Gpu) void {}
    pub fn isAvailable(_: *const Gpu) bool {
        return false;
    }
    pub fn getActiveDevice(_: *const Gpu) ?*const Device {
        return null;
    }
    pub fn createBuffer(_: *Gpu, _: usize, _: BufferOptions) Error!*UnifiedBuffer {
        return error.FeatureDisabled;
    }
    pub fn createBufferFromSlice(_: *Gpu, comptime _: type, _: anytype, _: BufferOptions) Error!*UnifiedBuffer {
        return error.FeatureDisabled;
    }
    pub fn destroyBuffer(_: *Gpu, _: *UnifiedBuffer) void {}
    pub fn vectorAdd(_: *Gpu, _: *UnifiedBuffer, _: *UnifiedBuffer, _: *UnifiedBuffer) Error!ExecutionResult {
        return error.FeatureDisabled;
    }
    pub fn matrixMultiply(_: *Gpu, _: *UnifiedBuffer, _: *UnifiedBuffer, _: *UnifiedBuffer, _: types.MatrixDims) Error!ExecutionResult {
        return error.FeatureDisabled;
    }
    pub fn getHealth(_: *const Gpu) Error!HealthStatus {
        return error.FeatureDisabled;
    }
    pub fn synchronize(_: *Gpu) Error!void {
        return error.FeatureDisabled;
    }
    pub fn createStream(_: *Gpu, _: StreamOptions) Error!*Stream {
        return error.FeatureDisabled;
    }
    pub fn createEvent(_: *Gpu, _: EventOptions) Error!*Event {
        return error.FeatureDisabled;
    }
    pub fn checkHealth(_: *const Gpu) HealthStatus {
        return .unhealthy;
    }
    pub fn reduceSum(_: *Gpu, _: *UnifiedBuffer) Error!struct { value: f32, stats: ExecutionResult } {
        return error.FeatureDisabled;
    }
    pub fn dotProduct(_: *Gpu, _: *UnifiedBuffer, _: *UnifiedBuffer) Error!struct { value: f32, stats: ExecutionResult } {
        return error.FeatureDisabled;
    }
    pub fn getStats(_: *const Gpu) GpuStats {
        return .{};
    }
    pub fn getMemoryInfo(_: *Gpu) MemoryInfo {
        return .{};
    }
    pub fn getMetricsSummary(_: *Gpu) ?MetricsSummary {
        return null;
    }
};

// ── GpuDevice ──────────────────────────────────────────────────────────────

pub const GpuDevice = struct {
    pub const DeviceCaps = struct { name: [256]u8 = undefined, name_len: usize = 0, total_memory: usize = 0 };

    pub fn init(_: std.mem.Allocator, _: GpuConfig) Error!GpuDevice {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *GpuDevice) void {}
    pub fn backendName(_: *const GpuDevice) []const u8 {
        return "disabled";
    }
    pub fn capabilities(_: *const GpuDevice) DeviceCaps {
        return .{};
    }
    pub fn createBuffer(_: *GpuDevice, comptime _: type, _: usize, _: BufferOptions) Error!UnifiedBuffer {
        return error.FeatureDisabled;
    }
    pub fn createBufferFromSlice(_: *GpuDevice, comptime _: type, _: anytype, _: BufferOptions) Error!UnifiedBuffer {
        return error.FeatureDisabled;
    }
    pub fn destroyBuffer(_: *GpuDevice, _: *UnifiedBuffer) void {}
    pub fn vectorAdd(_: *GpuDevice, _: *UnifiedBuffer, _: *UnifiedBuffer, _: *UnifiedBuffer) Error!ExecutionResult {
        return error.FeatureDisabled;
    }
    pub fn matrixMultiply(_: *GpuDevice, _: *UnifiedBuffer, _: *UnifiedBuffer, _: *UnifiedBuffer, _: types.MatrixDims) Error!ExecutionResult {
        return error.FeatureDisabled;
    }
    pub fn compileAndRun(_: *GpuDevice, _: *const anyopaque, _: LaunchConfig, _: anytype) Error!ExecutionResult {
        return error.FeatureDisabled;
    }
    pub fn memoryInfo(_: *GpuDevice) MemoryInfo {
        return .{};
    }
    pub fn stats(_: *const GpuDevice) GpuStats {
        return .{};
    }
    pub fn sync(_: *GpuDevice) Error!void {
        return error.FeatureDisabled;
    }
    pub fn isAvailable(_: *const GpuDevice) bool {
        return false;
    }
    pub fn checkHealth(_: *const GpuDevice) HealthStatus {
        return .unhealthy;
    }
};

// ── Context ────────────────────────────────────────────────────────────────

pub const Context = struct {
    pub fn init(_: std.mem.Allocator, _: config_module.GpuConfig) !*Context {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *Context) void {}
    pub fn getGpu(_: *Context) Error!*Gpu {
        return error.FeatureDisabled;
    }
    pub fn createBuffer(_: *Context, comptime _: type, _: usize, _: BufferOptions) Error!UnifiedBuffer {
        return error.FeatureDisabled;
    }
    pub fn createBufferFromSlice(_: *Context, comptime T: type, _: []const T, _: BufferOptions) Error!UnifiedBuffer {
        return error.FeatureDisabled;
    }
    pub fn destroyBuffer(_: *Context, _: *UnifiedBuffer) void {}
    pub fn vectorAdd(_: *Context, _: *UnifiedBuffer, _: *UnifiedBuffer, _: *UnifiedBuffer) Error!ExecutionResult {
        return error.FeatureDisabled;
    }
    pub fn matrixMultiply(_: *Context, _: *UnifiedBuffer, _: *UnifiedBuffer, _: *UnifiedBuffer, _: types.MatrixDims) Error!ExecutionResult {
        return error.FeatureDisabled;
    }
    pub fn getHealth(_: *Context) Error!HealthStatus {
        return error.FeatureDisabled;
    }
};

// ── Module-level functions ─────────────────────────────────────────────────
// init, deinit, isInitialized use canonical StubFeatureNoConfig helper.
// isEnabled takes a Backend param (not no-arg) so remains custom.

const Stub = stub_helpers.StubFeatureNoConfig(Error);
pub const init = Stub.init;
pub const deinit = Stub.deinit;
pub const isInitialized = Stub.isInitialized;

pub fn isEnabled(_: Backend) bool {
    return false;
}
pub fn ensureInitialized(_: std.mem.Allocator) Error!void {
    return error.FeatureDisabled;
}

test {
    std.testing.refAllDecls(@This());
}
