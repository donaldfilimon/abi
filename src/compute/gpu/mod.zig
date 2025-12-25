//! GPU support module
//!
//! Provides GPU workload execution and backend management.
//! Feature-gated: only compiled when enable_gpu is true.

const std = @import("std");

const build_options = @import("build_options");
const workload = @import("../runtime/workload.zig");

pub const GPUBackend = enum {
    none,
    cuda,
    vulkan,
    metal,
    webgpu,
};

pub const GPUWorkloadVTable = struct {
    base: workload.WorkloadVTable,
    gpu_exec: *const fn (user: *anyopaque, ctx: *workload.ExecutionContext, a: std.mem.Allocator, gpu_ctx: GPUExecutionContext) anyerror!*anyopaque,
    get_memory_requirements: *const fn (user: *anyopaque) GPUMemoryRequirements,
    can_execute_on_gpu: *const fn (user: *anyopaque) bool,
};

pub const GPUMemoryRequirements = struct {
    device_memory_bytes: u64,
    host_memory_bytes: u64,
    requires_shared_memory: bool,
};

pub const GPUExecutionContext = struct {
    backend: GPUBackend,
    device_id: u32,
    stream_id: u32,
    dedicated_memory: ?*anyopaque = null,
};

pub const GPUWorkloadHints = struct {
    prefers_gpu: bool = false,
    requires_double_precision: bool = false,
    min_compute_capability: ?u32 = null,
    estimated_memory_bytes: ?u64 = null,
};

pub const GPUManager = struct {
    backend: GPUBackend,
    allocator: std.mem.Allocator,
    available_devices: []GPUDeviceInfo,

    pub fn init(allocator: std.mem.Allocator, backend: GPUBackend) !GPUManager {
        return GPUManager{
            .backend = backend,
            .allocator = allocator,
            .available_devices = &.{},
        };
    }

    pub fn deinit(self: *GPUManager) void {
        _ = self.available_devices;
    }

    pub fn getBestDevice(self: *GPUManager, requirements: GPUMemoryRequirements) ?u32 {
        _ = self;
        _ = requirements;
        return null;
    }
};

pub const GPUDeviceInfo = struct {
    device_id: u32,
    name: []const u8,
    total_memory_bytes: u64,
    compute_capability: ?u32 = null,
};

pub const DEFAULT_GPU_HINTS = GPUWorkloadHints{};

pub const memory = @import("memory.zig");

pub const GPUBuffer = memory.GPUBuffer;
pub const BufferFlags = memory.BufferFlags;
pub const GPUMemoryPool = memory.GPUMemoryPool;
pub const AsyncTransfer = memory.AsyncTransfer;
