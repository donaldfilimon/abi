//! GPU Stub Module — API-compatible no-ops when GPU is disabled at compile time.

const std = @import("std");
const config_module = @import("../../core/config/mod.zig");
const stub_common = @import("../../services/shared/mod.zig").stub_common;

// ── Errors ─────────────────────────────────────────────────────────────────

pub const Error = error{ FeatureDisabled, NoDeviceAvailable, InitializationFailed, InvalidConfig, OutOfMemory, KernelCompilationFailed, KernelExecutionFailed } || stub_common.CommonError;
pub const GpuError = Error;
pub const MemoryError = error{ OutOfMemory, InvalidPointer, BufferTooSmall, HostAccessDisabled, DeviceMemoryMissing, SizeMismatch, InvalidOffset, TransferFailed };
pub const KernelError = error{ CompilationFailed, InvalidKernel, InvalidArgument, LaunchFailed, BackendUnsupported };

// ── Local Stubs Imports ────────────────────────────────────────────────────

const backend = @import("backend.zig");
const memory = @import("stubs/memory.zig");
const kernel = @import("stubs/kernel.zig");
const dsl_mod = @import("stubs/dsl.zig");
const execution = @import("stubs/execution.zig");
const recovery_mod = @import("stubs/recovery.zig");
const multi_gpu = @import("stubs/multi_gpu.zig");
const config = @import("stubs/config.zig");
const platform_mod = @import("stubs/platform.zig");
const backend_factory_mod = @import("stubs/backend_factory.zig");
const dispatcher_mod = @import("stubs/dispatcher.zig");
const diagnostics_mod = @import("stubs/diagnostics.zig");
const execution_coordinator_mod = @import("stubs/execution_coordinator.zig");
const std_gpu_mod = @import("stubs/std_gpu.zig");
const misc = @import("stubs/misc.zig");
const profiler = @import("stubs/profiler.zig");

// ── Essential Shared Types ─────────────────────────────────────────────────

pub const Backend = @import("backend.zig").Backend;

pub const Device = @import("stubs/device.zig").Device;
pub const DeviceType = @import("stubs/device.zig").DeviceType;

pub const Buffer = memory.Buffer;
pub const GpuBuffer = Buffer;
pub const UnifiedBuffer = memory.UnifiedBuffer;
pub const BufferFlags = memory.BufferFlags;
pub const BufferOptions = memory.BufferOptions;

pub const Stream = @import("stubs/stream.zig").Stream;
pub const StreamOptions = @import("stubs/stream.zig").StreamOptions;
pub const Event = @import("stubs/stream.zig").Event;
pub const EventOptions = @import("stubs/stream.zig").EventOptions;

pub const LaunchConfig = execution.LaunchConfig;
pub const ExecutionResult = execution.ExecutionResult;
pub const HealthStatus = execution.HealthStatus;

pub const KernelBuilder = kernel.KernelBuilder;

pub const GpuConfig = config.GpuConfig;

// ── Sub-module Namespace Stubs ─────────────────────────────────────────────

pub const profiling = misc.profiling;
pub const occupancy = misc.occupancy;
pub const fusion = misc.fusion;
pub const execution_coordinator = misc.execution_coordinator;
pub const memory_pool_advanced = misc.memory_pool_advanced;
pub const memory_pool_lockfree = misc.memory_pool_lockfree;
pub const sync_event = misc.sync_event;
pub const kernel_ring = misc.kernel_ring;
pub const adaptive_tiling = misc.adaptive_tiling;
pub const std_gpu = misc.std_gpu;
pub const std_gpu_kernels = misc.std_gpu_kernels;
pub const unified = misc.unified;
pub const unified_buffer = misc.unified_buffer;
pub const device = @import("stubs/device.zig");
pub const stream = @import("stubs/stream.zig");
pub const dsl = dsl_mod.dsl;
pub const interface = misc.interface;
pub const cuda_loader = misc.cuda_loader;
pub const builtin_kernels = misc.builtin_kernels;
pub const diagnostics = diagnostics_mod;
pub const error_handling = misc.error_handling;
pub const multi_device = misc.multi_device;
pub const peer_transfer = misc.peer_transfer;
pub const mega = misc.mega;
pub const platform = platform_mod;
pub const dispatch = dispatcher_mod;
pub const recovery = recovery_mod.recovery;
pub const failover = recovery_mod.failover;
pub const failover_types = misc.failover_types;

// AI training bridge stubs (parity with mod.zig)
pub const coordinator_ai_ops = struct {
    pub const CoordinatorAiOps = struct {
        allocator: std.mem.Allocator,
        coordinator: ?void = null,
        initialized: bool = false,

        pub fn init(allocator: std.mem.Allocator) CoordinatorAiOps {
            return .{ .allocator = allocator };
        }
        pub fn deinit(_: *CoordinatorAiOps) void {}
        pub fn isAvailable(_: *const CoordinatorAiOps) bool {
            return false;
        }
    };
};

pub const training_bridge = struct {
    pub const GpuTrainingStats = struct {
        total_gpu_ops: u64 = 0,
        gpu_time_ns: u64 = 0,
        cpu_fallback_ops: u64 = 0,
        utilization: f32 = 0,
        backend_name: []const u8 = "none",
        gpu_available: bool = false,

        pub fn avgKernelTimeMs(self: GpuTrainingStats) f32 {
            if (self.total_gpu_ops == 0) return 0;
            return @as(f32, @floatFromInt(self.gpu_time_ns)) / @as(f32, @floatFromInt(self.total_gpu_ops)) / 1e6;
        }
        pub fn gpuRatio(self: GpuTrainingStats) f32 {
            const total = self.total_gpu_ops + self.cpu_fallback_ops;
            if (total == 0) return 0;
            return @as(f32, @floatFromInt(self.total_gpu_ops)) / @as(f32, @floatFromInt(total));
        }
    };

    pub const GpuTrainingBridge = struct {
        gpu_ops: ?void = null,
        gpu_available: bool = false,
        stats: GpuTrainingStats = .{},
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) GpuTrainingBridge {
            return .{ .allocator = allocator };
        }
        pub fn deinit(_: *GpuTrainingBridge) void {}
        pub fn getStats(self: *const GpuTrainingBridge) GpuTrainingStats {
            return self.stats;
        }
        pub fn matmul(_: *GpuTrainingBridge, _: []const f32, _: []const f32, _: []f32, _: u32, _: u32, _: u32) void {}
        pub fn rmsNorm(_: *GpuTrainingBridge, _: []f32, _: []const f32, _: f32) void {}
        pub fn softmax(_: *GpuTrainingBridge, _: []f32) void {}
        pub fn silu(_: *GpuTrainingBridge, _: []f32) void {}
        pub fn elementwiseMul(_: *GpuTrainingBridge, _: []f32, _: []const f32) void {}
        pub fn elementwiseAdd(_: *GpuTrainingBridge, _: []f32, _: []const f32) void {}
        pub fn updateUtilization(_: *GpuTrainingBridge) void {}
    };
};

pub const gradient_compression = struct {
    pub const CompressedGradient = struct {
        indices: []u32 = &.{},
        values: []f32 = &.{},
        original_size: usize = 0,
        compression_ratio: f32 = 0,
        allocator: std.mem.Allocator,

        pub fn deinit(_: *CompressedGradient) void {}
        pub fn compressedBytes(_: *const CompressedGradient) usize {
            return 0;
        }
        pub fn originalBytes(_: *const CompressedGradient) usize {
            return 0;
        }
    };

    pub const CompressionStats = struct {
        compressed_bytes: usize = 0,
        original_bytes: usize = 0,
        ratio: f32 = 0,
        residual_norm: f32 = 0,
        compressions_count: u64 = 0,
    };

    pub const GradientCompressor = struct {
        residual: []f32 = &.{},
        ratio: f32 = 0,
        allocator: std.mem.Allocator,
        gradient_size: usize = 0,
        compressions_count: u64 = 0,

        pub fn init(allocator: std.mem.Allocator, _: usize, _: f32) !GradientCompressor {
            return .{ .allocator = allocator };
        }
        pub fn deinit(_: *GradientCompressor) void {}
        pub fn compress(_: *GradientCompressor, _: []const f32) !CompressedGradient {
            return error.FeatureDisabled;
        }
        pub fn decompress(_: *const CompressedGradient, output: []f32) void {
            @memset(output, 0);
        }
        pub fn getStats(_: *const GradientCompressor) CompressionStats {
            return .{};
        }
    };

    pub const GradientBucketManager = struct {
        bucket_size: usize = 0,
        allocator: std.mem.Allocator,

        pub const Bucket = struct {
            data: []f32 = &.{},
            used: usize = 0,
            capacity: usize = 0,

            pub fn deinit(_: *Bucket, _: std.mem.Allocator) void {}
        };

        pub fn init(allocator: std.mem.Allocator, _: usize) GradientBucketManager {
            return .{ .allocator = allocator };
        }
        pub fn deinit(_: *GradientBucketManager) void {}
        pub fn addGradient(_: *GradientBucketManager, _: u32, _: []const f32) !void {
            return error.FeatureDisabled;
        }
        pub fn hasReadyBucket(_: *const GradientBucketManager) bool {
            return false;
        }
        pub fn bucketCount(_: *const GradientBucketManager) usize {
            return 0;
        }
    };
};

// Namespaced GPU API surface (hard API cutover)
pub const backends = struct {
    pub const types = struct {
        pub const Backend = @import("backend.zig").Backend;
        pub const DetectionLevel = @import("backend.zig").DetectionLevel;
        pub const BackendAvailability = @import("backend.zig").BackendAvailability;
        pub const BackendInfo = @import("backend.zig").BackendInfo;
        pub const DeviceCapability = @import("backend.zig").DeviceCapability;
        pub const DeviceInfo = @import("backend.zig").DeviceInfo;
        pub const Summary = @import("backend.zig").Summary;
    };

    pub const detect = struct {
        pub fn moduleEnabled() bool {
            return false;
        }

        pub fn isEnabled(_: types.Backend) bool {
            return false;
        }

        pub fn backendAvailability(_: types.Backend) types.BackendAvailability {
            return .{ .enabled = false, .available = false, .reason = "gpu module disabled", .device_count = 0, .level = .none };
        }

        pub fn availableBackends(_: std.mem.Allocator) Error![]types.Backend {
            return error.FeatureDisabled;
        }
    };

    pub const meta = struct {
        pub fn backendName(_: types.Backend) []const u8 {
            return "disabled";
        }

        pub fn backendDisplayName(_: types.Backend) []const u8 {
            return "GPU Disabled";
        }

        pub fn backendDescription(_: types.Backend) []const u8 {
            return "GPU feature is disabled at compile time";
        }

        pub fn backendFromString(_: []const u8) ?types.Backend {
            return null;
        }

        pub fn backendSupportsKernels(_: types.Backend) bool {
            return false;
        }

        pub fn backendFlag(_: types.Backend) []const u8 {
            return "disabled";
        }
    };

    pub const listing = struct {
        pub fn listBackendInfo(_: std.mem.Allocator) Error![]types.BackendInfo {
            return error.FeatureDisabled;
        }

        pub fn listDevices(_: std.mem.Allocator) Error![]types.DeviceInfo {
            return error.FeatureDisabled;
        }

        pub fn defaultDevice(_: std.mem.Allocator) !?types.DeviceInfo {
            return null;
        }

        pub fn defaultDeviceLabel() []const u8 {
            return "disabled";
        }

        pub fn summary() types.Summary {
            return .{
                .module_enabled = false,
                .enabled_backend_count = 0,
                .available_backend_count = 0,
                .device_count = 0,
                .emulated_devices = 0,
            };
        }
    };

    pub const libs = struct {};
    pub const registry = struct {};
    pub const pool = struct {};
};
pub const devices = device;
pub const runtime = struct {};
pub const policy = struct {};
pub const multi = multi_gpu;
pub const factory = backend_factory_mod;

// ── Additional Types ──────────────────────────────────────────────────────

pub const MemoryInfo = struct {
    total_bytes: u64 = 0,
    used_bytes: u64 = 0,
    free_bytes: u64 = 0,
    peak_used_bytes: u64 = 0,
};

pub const GpuStats = struct {
    kernels_launched: u64 = 0,
    buffers_created: u64 = 0,
    bytes_allocated: u64 = 0,
    host_to_device_transfers: u64 = 0,
    device_to_host_transfers: u64 = 0,
    total_execution_time_ns: u64 = 0,
};

pub const MetricsSummary = struct {
    total_kernel_invocations: u64 = 0,
    avg_kernel_time_ns: f64 = 0,
    kernels_per_second: f64 = 0,
};

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
    pub fn matrixMultiply(_: *Gpu, _: *UnifiedBuffer, _: *UnifiedBuffer, _: *UnifiedBuffer, _: execution.MatrixDims) Error!ExecutionResult {
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

// ── GpuDevice (ergonomic wrapper stub) ─────────────────────────────────────

pub const GpuDevice = struct {
    pub const DeviceCaps = struct {
        name: [256]u8 = undefined,
        name_len: usize = 0,
        total_memory: usize = 0,
    };

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
    pub fn matrixMultiply(_: *GpuDevice, _: *UnifiedBuffer, _: *UnifiedBuffer, _: *UnifiedBuffer, _: execution.MatrixDims) Error!ExecutionResult {
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
    pub fn matrixMultiply(_: *Context, _: *UnifiedBuffer, _: *UnifiedBuffer, _: *UnifiedBuffer, _: execution.MatrixDims) Error!ExecutionResult {
        return error.FeatureDisabled;
    }
    pub fn getHealth(_: *Context) Error!HealthStatus {
        return error.FeatureDisabled;
    }
};

// ── Module-level functions ─────────────────────────────────────────────────

pub fn isEnabled(_: Backend) bool {
    return false;
}
pub fn isInitialized() bool {
    return false;
}
pub fn init(_: std.mem.Allocator) Error!void {
    return error.FeatureDisabled;
}
pub fn deinit() void {}
pub fn ensureInitialized(_: std.mem.Allocator) Error!void {
    return error.FeatureDisabled;
}

test {
    std.testing.refAllDecls(@This());
}
