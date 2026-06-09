const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const gpu = if (build_options.feat_gpu) @import("../gpu/mod.zig") else @import("../gpu/stub.zig");

pub const Platform = enum {
    ios,
    android,
    unknown,
};

pub const PlatformStatus = struct {
    platform: Platform,
    available: bool,
    accelerated: bool,
    message: []const u8,
};

pub const ScreenConfig = struct {
    width: u32,
    height: u32,
    density: f32,
};

pub const DeviceInfo = struct {
    platform: Platform,
    os_version: []const u8,
    screen: ScreenConfig,
    hardware_model: []const u8,
    owned: bool = false,

    pub fn deinit(self: *DeviceInfo, allocator: std.mem.Allocator) void {
        if (!self.owned) return;
        allocator.free(self.os_version);
        allocator.free(self.hardware_model);
    }
};

pub fn platformName(platform: Platform) []const u8 {
    return switch (platform) {
        .ios => "ios",
        .android => "android",
        .unknown => "unknown",
    };
}

pub fn detectPlatform() PlatformStatus {
    if (builtin.target.os.tag == .ios) {
        return .{
            .platform = .ios,
            .available = true,
            .accelerated = false,
            .message = "iOS platform detected; mobile feature active, native dispatch pending",
        };
    }

    if (comptime @hasField(std.Target.Os.Tag, "android")) {
        if (builtin.target.os.tag == .android) {
            return .{
                .platform = .android,
                .available = true,
                .accelerated = false,
                .message = "Android platform detected; mobile feature active, native dispatch pending",
            };
        }
    }

    const gpu_status = gpu.detectBackend();
    return .{
        .platform = .unknown,
        .available = true,
        .accelerated = gpu_status.accelerated,
        .message = "Desktop platform; mobile feature active with simulated mobile profile",
    };
}

pub fn isAvailable() bool {
    return detectPlatform().available;
}

pub fn preferredPlatform() Platform {
    return detectPlatform().platform;
}

pub fn getDeviceInfo(allocator: std.mem.Allocator) !DeviceInfo {
    const status = detectPlatform();
    return .{
        .platform = status.platform,
        .os_version = try allocator.dupe(u8, "simulated"),
        .screen = .{ .width = 375, .height = 812, .density = 3.0 },
        .hardware_model = try allocator.dupe(u8, "simulated-device"),
        .owned = true,
    };
}

pub fn renderMobileView(allocator: std.mem.Allocator, title: []const u8, items: []const []const u8) ![]u8 {
    if (title.len == 0) return error.InvalidMobileView;

    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    const status = detectPlatform();
    try out.print(allocator, "[{s}] {s}\n", .{ platformName(status.platform), title });
    for (items) |item| {
        try out.print(allocator, "  - {s}\n", .{item});
    }
    try out.print(allocator, "status: {s}\n", .{status.message});

    return try out.toOwnedSlice(allocator);
}

pub fn executeMobileTask(allocator: std.mem.Allocator, task_name: []const u8) ![]u8 {
    if (task_name.len == 0) return error.InvalidTaskName;
    const status = detectPlatform();
    return try std.fmt.allocPrint(
        allocator,
        "mobile task '{s}' accepted on {s}; simulated execution completed",
        .{ task_name, platformName(status.platform) },
    );
}

test {
    std.testing.refAllDecls(@This());
}

test "mobile platform detection always provides a safe platform" {
    const status = detectPlatform();
    try std.testing.expect(status.available);
    try std.testing.expect(status.message.len > 0);
}

test "mobile device info returns simulated data" {
    var info = try getDeviceInfo(std.testing.allocator);
    defer info.deinit(std.testing.allocator);
    try std.testing.expect(info.platform == .unknown or info.platform == .ios or info.platform == .android);
    try std.testing.expect(info.screen.width > 0);
    try std.testing.expect(info.screen.height > 0);
}

test "mobile view renders title and items" {
    const rendered = try renderMobileView(std.testing.allocator, "Test", &.{ "item1", "item2" });
    defer std.testing.allocator.free(rendered);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "Test") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "item1") != null);
}

test "mobile task execution validates input" {
    try std.testing.expectError(error.InvalidTaskName, executeMobileTask(std.testing.allocator, ""));
    const result = try executeMobileTask(std.testing.allocator, "sync");
    defer std.testing.allocator.free(result);
    try std.testing.expect(std.mem.indexOf(u8, result, "sync") != null);
}
