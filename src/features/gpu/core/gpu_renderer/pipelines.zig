const std = @import("std");
const gpu = std.gpu;

const config = @import("config.zig");

const GPUHandle = config.GPUHandle;
const ShaderStage = config.ShaderStage;

pub const Shader = struct {
    handle: GPUHandle,
    stage: ShaderStage,
    source: []const u8,

    pub fn compile(allocator: std.mem.Allocator, stage: ShaderStage, source: []const u8) !Shader {
        _ = allocator;

        // Generate a unique handle based on stage and source hash
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(std.mem.asBytes(&stage));
        hasher.update(source);
        const hash = hasher.final();

        return Shader{
            .handle = GPUHandle{ .id = @truncate(hash), .generation = 1 },
            .stage = stage,
            .source = source,
        };
    }

    pub fn deinit(self: *Shader) void {
        _ = self;
        // No cleanup needed for CPU fallback
    }
};

/// Bind group for resource binding
pub const BindGroup = struct {
    handle: GPUHandle,
    buffer_handles: std.ArrayList(u32),
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, id: u64) Self {
        return .{
            .handle = GPUHandle{ .id = id, .generation = 1 },
            .buffer_handles = std.ArrayList(u32){},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.buffer_handles.deinit(self.allocator);
    }

    pub fn addBufferHandle(self: *Self, handle: u32) !void {
        try self.buffer_handles.append(self.allocator, handle);
    }
};

pub const BindGroupDesc = struct {
    buffers: []const u32 = &[_]u32{},
};

/// Lightweight renderer statistics
pub const RendererStats = struct {
    buffers_created: u64 = 0,
    buffers_destroyed: u64 = 0,
    bytes_current: u64 = 0,
    bytes_peak: u64 = 0,
    bytes_written: u64 = 0,
    bytes_read: u64 = 0,
    bytes_copied: u64 = 0,
    compute_operations: u64 = 0,
    last_operation_time_ns: u64 = 0,
    shaders_compiled: u64 = 0,
    compilation_time_ns: u64 = 0,
};

/// Compute dispatch metadata shared with CPU fallbacks
pub const ComputeDispatchInfo = struct {
    workgroups: [3]u32,
    push_constants: []const u8,
};

pub const CpuFallbackFn = *const fn (
    *anyopaque,
    []const u32,
    ComputeDispatchInfo,
    ?*anyopaque,
) anyerror!void;

/// Compute pipeline description used during creation
pub const ComputePipelineDesc = struct {
    label: []const u8 = "",
    shader_source: []const u8,
    entry_point: []const u8 = "main",
    workgroup_size: [3]u32 = .{ 1, 1, 1 },
    cpu_fallback: ?CpuFallbackFn = null,
    cpu_fallback_ctx: ?*anyopaque = null,
};

/// Runtime compute pipeline representation
pub const ComputePipeline = struct {
    handle: GPUHandle,
    label: []const u8,
    shader_source: []const u8,
    entry_point: []const u8,
    workgroup_size: [3]u32,
    cpu_fallback: ?CpuFallbackFn,
    cpu_fallback_ctx: ?*anyopaque,
    hardware: ?HardwareState = null,

    const HardwareState = struct {
        pipeline: *gpu.ComputePipeline,
        layout: *gpu.BindGroupLayout,

        fn deinit(self: *HardwareState) void {
            self.layout.deinit();
            self.pipeline.deinit();
        }
    };

    fn init(allocator: std.mem.Allocator, desc: ComputePipelineDesc, id: u64) !ComputePipeline {
        const label = try allocator.dupe(u8, desc.label);
        errdefer allocator.free(label);
        const shader_source = try allocator.dupe(u8, desc.shader_source);
        errdefer allocator.free(shader_source);
        const entry_point = try allocator.dupe(u8, desc.entry_point);
        errdefer allocator.free(entry_point);

        return .{
            .handle = .{ .id = id, .generation = 1 },
            .label = label,
            .shader_source = shader_source,
            .entry_point = entry_point,
            .workgroup_size = desc.workgroup_size,
            .cpu_fallback = desc.cpu_fallback,
            .cpu_fallback_ctx = desc.cpu_fallback_ctx,
        };
    }

    fn deinit(self: *ComputePipeline, allocator: std.mem.Allocator) void {
        allocator.free(self.label);
        allocator.free(self.shader_source);
        allocator.free(self.entry_point);
        if (self.hardware) |*hw| hw.deinit();
    }
};

pub const ComputeDispatch = struct {
    pipeline: u32,
    bind_group: u32,
    workgroups_x: u32,
    workgroups_y: u32,
    workgroups_z: u32,
    push_constants: []const u8 = &[_]u8{},
};

/// Main GPU renderer with cross-platform support and CPU fallbacks
