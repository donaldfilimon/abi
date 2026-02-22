//! GPU Stub Module — API-compatible no-ops when GPU is disabled at compile time.

const std = @import("std");
const config_module = @import("../../core/config/mod.zig");
const stub_common = @import("../../services/shared/stub_common.zig");

// ── Errors ─────────────────────────────────────────────────────────────────

pub const Error = error{ GpuDisabled, NoDeviceAvailable, InitializationFailed, InvalidConfig, OutOfMemory, KernelCompilationFailed, KernelExecutionFailed } || stub_common.CommonError;
pub const GpuError = Error;
pub const MemoryError = error{ OutOfMemory, InvalidPointer, BufferTooSmall, HostAccessDisabled, DeviceMemoryMissing, SizeMismatch, InvalidOffset, TransferFailed };
pub const KernelError = error{ CompilationFailed, InvalidKernel, InvalidArgument, LaunchFailed, BackendUnsupported };

// ── Local Stubs Imports ────────────────────────────────────────────────────

const backend = @import("stubs/backend.zig");
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

pub const Backend = backend.Backend;

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
            return error.GpuDisabled;
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
            return error.GpuDisabled;
        }

        pub fn listDevices(_: std.mem.Allocator) Error![]types.DeviceInfo {
            return error.GpuDisabled;
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

// ── Gpu struct ─────────────────────────────────────────────────────────────

pub const Gpu = struct {
    config: GpuConfig = .{},

    pub fn init(_: std.mem.Allocator, _: GpuConfig) Error!Gpu {
        return error.GpuDisabled;
    }
    pub fn deinit(_: *Gpu) void {}
    pub fn isAvailable(_: *const Gpu) bool {
        return false;
    }
    pub fn getActiveDevice(_: *const Gpu) ?*const Device {
        return null;
    }
    pub fn createBuffer(_: *Gpu, comptime _: type, _: usize, _: BufferOptions) Error!UnifiedBuffer {
        return error.GpuDisabled;
    }
    pub fn createBufferFromSlice(_: *Gpu, comptime _: type, _: anytype, _: BufferOptions) Error!UnifiedBuffer {
        return error.GpuDisabled;
    }
    pub fn destroyBuffer(_: *Gpu, _: *UnifiedBuffer) void {}
    pub fn vectorAdd(_: *Gpu, _: *UnifiedBuffer, _: *UnifiedBuffer, _: *UnifiedBuffer) Error!ExecutionResult {
        return error.GpuDisabled;
    }
    pub fn matrixMultiply(_: *Gpu, _: *UnifiedBuffer, _: *UnifiedBuffer, _: *UnifiedBuffer, _: execution.MatrixDims) Error!ExecutionResult {
        return error.GpuDisabled;
    }
    pub fn getHealth(_: *const Gpu) Error!HealthStatus {
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
    pub fn checkHealth(_: *const Gpu) HealthStatus {
        return .unhealthy;
    }
};

// ── GpuDevice (ergonomic wrapper stub) ─────────────────────────────────────

pub const GpuDevice = struct {
    pub const DeviceCaps = struct {
        name: [256]u8 = undefined,
        name_len: usize = 0,
        total_memory: usize = 0,
    };
    pub const MemoryInfo = struct {
        total_bytes: u64 = 0,
        used_bytes: u64 = 0,
        free_bytes: u64 = 0,
        peak_used_bytes: u64 = 0,
    };

    pub fn init(_: std.mem.Allocator, _: GpuConfig) Error!GpuDevice {
        return error.GpuDisabled;
    }
    pub fn deinit(_: *GpuDevice) void {}
    pub fn backendName(_: *const GpuDevice) []const u8 {
        return "disabled";
    }
    pub fn capabilities(_: *const GpuDevice) DeviceCaps {
        return .{};
    }
    pub fn createBuffer(_: *GpuDevice, comptime _: type, _: usize, _: BufferOptions) Error!UnifiedBuffer {
        return error.GpuDisabled;
    }
    pub fn createBufferFromSlice(_: *GpuDevice, comptime _: type, _: anytype, _: BufferOptions) Error!UnifiedBuffer {
        return error.GpuDisabled;
    }
    pub fn destroyBuffer(_: *GpuDevice, _: *UnifiedBuffer) void {}
    pub fn vectorAdd(_: *GpuDevice, _: *UnifiedBuffer, _: *UnifiedBuffer, _: *UnifiedBuffer) Error!ExecutionResult {
        return error.GpuDisabled;
    }
    pub fn matrixMultiply(_: *GpuDevice, _: *UnifiedBuffer, _: *UnifiedBuffer, _: *UnifiedBuffer, _: execution.MatrixDims) Error!ExecutionResult {
        return error.GpuDisabled;
    }
    pub fn compileAndRun(_: *GpuDevice, _: *const anyopaque, _: LaunchConfig, _: anytype) Error!ExecutionResult {
        return error.GpuDisabled;
    }
    pub fn memoryInfo(_: *GpuDevice) MemoryInfo {
        return .{};
    }
    pub fn stats(_: *const GpuDevice) execution.GpuStats {
        return .{};
    }
    pub fn sync(_: *GpuDevice) Error!void {
        return error.GpuDisabled;
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
    pub fn matrixMultiply(_: *Context, _: *UnifiedBuffer, _: *UnifiedBuffer, _: *UnifiedBuffer, _: execution.MatrixDims) Error!ExecutionResult {
        return error.GpuDisabled;
    }
    pub fn getHealth(_: *Context) Error!HealthStatus {
        return error.GpuDisabled;
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
    return error.GpuDisabled;
}
pub fn deinit() void {}
pub fn ensureInitialized(_: std.mem.Allocator) Error!void {
    return error.GpuDisabled;
}
