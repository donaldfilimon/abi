const std = @import("std");
const build_options = @import("build_options");

pub const GpuError = error{
    GpuDisabled,
    SizeMismatch,
};

pub const Backend = enum {
    cuda,
    vulkan,
    metal,
    webgpu,
};

pub const BackendInfo = struct {
    backend: Backend,
    name: []const u8,
    description: []const u8,
    enabled: bool,
};

pub const DeviceInfo = struct {
    id: u32,
    backend: Backend,
    name: []const u8,
    total_memory_bytes: ?u64 = null,
    is_emulated: bool = true,
};

pub const Summary = struct {
    module_enabled: bool,
    enabled_backend_count: usize,
    device_count: usize,
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

    pub fn fromBytes(allocator: std.mem.Allocator, data: []const u8) !Buffer {
        var buffer = try Buffer.init(allocator, data.len);
        std.mem.copyForwards(u8, buffer.bytes, data);
        return buffer;
    }

    pub fn deinit(self: *Buffer) void {
        self.allocator.free(self.bytes);
        self.* = undefined;
    }

    pub fn len(self: *const Buffer) usize {
        return self.bytes.len;
    }

    pub fn fill(self: *Buffer, value: u8) void {
        @memset(self.bytes, value);
    }

    pub fn copyFrom(self: *Buffer, data: []const u8) GpuError!void {
        if (data.len != self.bytes.len) return GpuError.SizeMismatch;
        std.mem.copyForwards(u8, self.bytes, data);
    }

    pub fn asSlice(self: *Buffer) []u8 {
        return self.bytes;
    }

    pub fn asConstSlice(self: *const Buffer) []const u8 {
        return self.bytes;
    }
};

var initialized: bool = false;

pub fn init(_: std.mem.Allocator) !void {
    if (!moduleEnabled()) return GpuError.GpuDisabled;
    initialized = true;
}

pub fn deinit() void {
    initialized = false;
}

pub fn moduleEnabled() bool {
    return build_options.enable_gpu;
}

pub fn isInitialized() bool {
    return initialized;
}

pub fn summary() Summary {
    if (!moduleEnabled()) {
        return .{
            .module_enabled = false,
            .enabled_backend_count = 0,
            .device_count = 0,
        };
    }
    const count = countEnabledBackends();
    return .{
        .module_enabled = true,
        .enabled_backend_count = count,
        .device_count = count,
    };
}

pub fn backendName(backend: Backend) []const u8 {
    return switch (backend) {
        .cuda => "cuda",
        .vulkan => "vulkan",
        .metal => "metal",
        .webgpu => "webgpu",
    };
}

pub fn backendDisplayName(backend: Backend) []const u8 {
    return switch (backend) {
        .cuda => "CUDA",
        .vulkan => "Vulkan",
        .metal => "Metal",
        .webgpu => "WebGPU",
    };
}

pub fn backendDescription(backend: Backend) []const u8 {
    return switch (backend) {
        .cuda => "NVIDIA CUDA backend",
        .vulkan => "Cross-platform Vulkan backend",
        .metal => "Apple Metal backend",
        .webgpu => "WebGPU backend",
    };
}

pub fn backendFlag(backend: Backend) []const u8 {
    return switch (backend) {
        .cuda => "-Dgpu-cuda",
        .vulkan => "-Dgpu-vulkan",
        .metal => "-Dgpu-metal",
        .webgpu => "-Dgpu-webgpu",
    };
}

pub fn backendFromString(text: []const u8) ?Backend {
    if (std.ascii.eqlIgnoreCase(text, "cuda")) return .cuda;
    if (std.ascii.eqlIgnoreCase(text, "vulkan")) return .vulkan;
    if (std.ascii.eqlIgnoreCase(text, "metal")) return .metal;
    if (std.ascii.eqlIgnoreCase(text, "webgpu")) return .webgpu;
    return null;
}

pub fn isEnabled(backend: Backend) bool {
    if (!moduleEnabled()) return false;
    return switch (backend) {
        .cuda => build_options.gpu_cuda,
        .vulkan => build_options.gpu_vulkan,
        .metal => build_options.gpu_metal,
        .webgpu => build_options.gpu_webgpu,
    };
}

pub fn listBackendInfo(allocator: std.mem.Allocator) ![]BackendInfo {
    var list = std.ArrayList(BackendInfo).empty;
    errdefer list.deinit(allocator);
    for (std.enums.values(Backend)) |backend| {
        try list.append(allocator, .{
            .backend = backend,
            .name = backendDisplayName(backend),
            .description = backendDescription(backend),
            .enabled = isEnabled(backend),
        });
    }
    return list.toOwnedSlice(allocator);
}

pub fn availableBackends(allocator: std.mem.Allocator) ![]Backend {
    var list = std.ArrayList(Backend).empty;
    errdefer list.deinit(allocator);
    if (!moduleEnabled()) return list.toOwnedSlice(allocator);
    for (std.enums.values(Backend)) |backend| {
        if (isEnabled(backend)) {
            try list.append(allocator, backend);
        }
    }
    return list.toOwnedSlice(allocator);
}

pub fn listDevices(allocator: std.mem.Allocator) ![]DeviceInfo {
    var devices = std.ArrayList(DeviceInfo).empty;
    errdefer devices.deinit(allocator);
    if (!moduleEnabled()) return devices.toOwnedSlice(allocator);

    var id: u32 = 0;
    for (std.enums.values(Backend)) |backend| {
        if (!isEnabled(backend)) continue;
        try devices.append(allocator, .{
            .id = id,
            .backend = backend,
            .name = backendDeviceName(backend),
            .total_memory_bytes = null,
            .is_emulated = true,
        });
        id += 1;
    }
    return devices.toOwnedSlice(allocator);
}

fn countEnabledBackends() usize {
    if (!moduleEnabled()) return 0;
    var count: usize = 0;
    for (std.enums.values(Backend)) |backend| {
        if (isEnabled(backend)) {
            count += 1;
        }
    }
    return count;
}

fn backendDeviceName(backend: Backend) []const u8 {
    return switch (backend) {
        .cuda => "CUDA Adapter (simulated)",
        .vulkan => "Vulkan Adapter (simulated)",
        .metal => "Metal Adapter (simulated)",
        .webgpu => "WebGPU Adapter (simulated)",
    };
}

test "backend parsing helpers" {
    try std.testing.expectEqual(@as(?Backend, .cuda), backendFromString("cuda"));
    try std.testing.expectEqual(@as(?Backend, .vulkan), backendFromString("VULKAN"));
    try std.testing.expectEqual(@as(?Backend, null), backendFromString("unknown"));
    try std.testing.expect(backendName(.metal).len > 0);
    try std.testing.expect(backendDisplayName(.webgpu).len > 0);
}

test "backend flags are stable" {
    try std.testing.expectEqualStrings("-Dgpu-cuda", backendFlag(.cuda));
    try std.testing.expectEqualStrings("-Dgpu-webgpu", backendFlag(.webgpu));
}

test "available backends are enabled" {
    const allocator = std.testing.allocator;
    const backends = try availableBackends(allocator);
    defer allocator.free(backends);
    for (backends) |backend| {
        try std.testing.expect(isEnabled(backend));
    }
}

test "summary matches enabled backends" {
    const details = summary();
    try std.testing.expect(details.module_enabled == moduleEnabled());
    try std.testing.expectEqual(details.enabled_backend_count, details.device_count);
}

test "buffer copy and fill" {
    var buffer = try Buffer.init(std.testing.allocator, 4);
    defer buffer.deinit();

    buffer.fill(0xaa);
    try std.testing.expectEqualSlices(u8, &.{ 0xaa, 0xaa, 0xaa, 0xaa }, buffer.bytes);

    try buffer.copyFrom(&.{ 1, 2, 3, 4 });
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4 }, buffer.bytes);
}
