const std = @import("std");

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
    return .{
        .platform = .unknown,
        .available = false,
        .accelerated = false,
        .message = "Mobile feature is disabled; no platform detected",
    };
}

pub fn isAvailable() bool {
    return false;
}

pub fn preferredPlatform() Platform {
    return .unknown;
}

pub fn getDeviceInfo(allocator: std.mem.Allocator) !DeviceInfo {
    return .{
        .platform = .unknown,
        .os_version = try allocator.dupe(u8, "disabled"),
        .screen = .{ .width = 0, .height = 0, .density = 0 },
        .hardware_model = try allocator.dupe(u8, "disabled"),
        .owned = true,
    };
}

pub fn renderMobileView(allocator: std.mem.Allocator, title: []const u8, items: []const []const u8) ![]u8 {
    _ = title;
    _ = items;
    return try allocator.dupe(u8, "Mobile feature is disabled");
}

pub fn executeMobileTask(allocator: std.mem.Allocator, task_name: []const u8) ![]u8 {
    _ = task_name;
    return try allocator.dupe(u8, "Mobile feature is disabled");
}
