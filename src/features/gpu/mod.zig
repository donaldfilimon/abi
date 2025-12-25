const std = @import("std");
const build_options = @import("build_options");

pub const Backend = enum {
    cuda,
    vulkan,
    metal,
    webgpu,
};

pub const DeviceInfo = struct {
    name: []const u8,
    backend: Backend,
    total_memory_bytes: u64,
};

pub const Buffer = struct {
    allocator: std.mem.Allocator,
    bytes: []u8,

    pub fn init(allocator: std.mem.Allocator, size: usize) !Buffer {
        return .{
            .allocator = allocator,
            .bytes = try allocator.alloc(u8, size),
        };
    }

    pub fn deinit(self: *Buffer) void {
        self.allocator.free(self.bytes);
        self.* = undefined;
    }
};

pub fn init(_: std.mem.Allocator) !void {}

pub fn deinit() void {}

pub fn availableBackends(allocator: std.mem.Allocator) ![]Backend {
    var list = std.ArrayList(Backend).empty;
    errdefer list.deinit(allocator);
    if (build_options.gpu_cuda) try list.append(allocator, .cuda);
    if (build_options.gpu_vulkan) try list.append(allocator, .vulkan);
    if (build_options.gpu_metal) try list.append(allocator, .metal);
    if (build_options.gpu_webgpu) try list.append(allocator, .webgpu);
    return list.toOwnedSlice(allocator);
}

pub fn listDevices(allocator: std.mem.Allocator) ![]DeviceInfo {
    const backends = try availableBackends(allocator);
    defer allocator.free(backends);

    var devices = std.ArrayList(DeviceInfo).empty;
    errdefer devices.deinit(allocator);
    for (backends) |backend| {
        const name = switch (backend) {
            .cuda => "CUDA Device",
            .vulkan => "Vulkan Device",
            .metal => "Metal Device",
            .webgpu => "WebGPU Device",
        };
        try devices.append(allocator, .{
            .name = name,
            .backend = backend,
            .total_memory_bytes = 0,
        });
    }
    return devices.toOwnedSlice(allocator);
}
