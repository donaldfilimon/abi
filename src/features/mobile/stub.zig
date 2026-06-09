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

pub const RuntimeMode = enum {
    native_platform,
    simulated_profile,
    disabled,
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

pub const MobileProfile = struct {
    platform: Platform,
    mode: RuntimeMode,
    screen: ScreenConfig,
    hardware_model: []const u8,
    native_dispatch: bool,
    message: []const u8,
};

pub const DeviceProfile = struct {
    platform: Platform,
    mode: RuntimeMode,
    native_dispatch: bool,
    simulated: bool,
    accelerated: bool,
    width: u32,
    height: u32,
    density: f32,
    item_count: usize = 0,
};

pub fn platformName(platform: Platform) []const u8 {
    return switch (platform) {
        .ios => "ios",
        .android => "android",
        .unknown => "unknown",
    };
}

pub fn runtimeModeName(mode: RuntimeMode) []const u8 {
    return switch (mode) {
        .native_platform => "native_platform",
        .simulated_profile => "simulated_profile",
        .disabled => "disabled",
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
    const mobile_profile = profile();
    return .{
        .platform = mobile_profile.platform,
        .os_version = try allocator.dupe(u8, runtimeModeName(mobile_profile.mode)),
        .screen = mobile_profile.screen,
        .hardware_model = try allocator.dupe(u8, mobile_profile.hardware_model),
        .owned = true,
    };
}

pub fn profile() MobileProfile {
    return .{
        .platform = .unknown,
        .mode = .disabled,
        .screen = .{ .width = 0, .height = 0, .density = 0 },
        .hardware_model = "disabled",
        .native_dispatch = false,
        .message = "Mobile feature is disabled; no platform detected",
    };
}

pub fn deviceProfile() DeviceProfile {
    const mobile_profile = profile();
    return .{
        .platform = mobile_profile.platform,
        .mode = mobile_profile.mode,
        .native_dispatch = mobile_profile.native_dispatch,
        .simulated = false,
        .accelerated = false,
        .width = mobile_profile.screen.width,
        .height = mobile_profile.screen.height,
        .density = mobile_profile.screen.density,
    };
}

pub fn layoutSummary(device_profile: DeviceProfile) !DeviceProfile {
    if (device_profile.mode == .disabled) return error.InvalidMobileView;
    if (device_profile.width == 0 or device_profile.height == 0) return error.InvalidMobileView;
    if (device_profile.density <= 0) return error.InvalidMobileView;
    return device_profile;
}

fn validateLabel(value: []const u8, err: anyerror) !void {
    if (value.len == 0) return err;
    if (std.mem.indexOfScalar(u8, value, 0) != null) return err;
}

pub fn renderMobileView(allocator: std.mem.Allocator, title: []const u8, items: []const []const u8) ![]u8 {
    try validateLabel(title, error.InvalidMobileView);
    for (items) |item| {
        try validateLabel(item, error.InvalidMobileView);
    }
    const mobile_profile = profile();
    return try std.fmt.allocPrint(
        allocator,
        "Mobile feature is disabled; mode={s}; native_dispatch={any}",
        .{ runtimeModeName(mobile_profile.mode), mobile_profile.native_dispatch },
    );
}

test {
    std.testing.refAllDecls(@This());
}

pub fn executeMobileTask(allocator: std.mem.Allocator, task_name: []const u8) ![]u8 {
    try validateLabel(task_name, error.InvalidTaskName);
    const mobile_profile = profile();
    return try std.fmt.allocPrint(
        allocator,
        "Mobile feature is disabled; task={s}; mode={s}; native_dispatch={any}",
        .{ task_name, runtimeModeName(mobile_profile.mode), mobile_profile.native_dispatch },
    );
}

test "mobile stub preserves profile and validation contracts" {
    const mobile_profile = profile();
    try std.testing.expectEqual(Platform.unknown, mobile_profile.platform);
    try std.testing.expectEqual(RuntimeMode.disabled, mobile_profile.mode);
    try std.testing.expect(!mobile_profile.native_dispatch);
    const device_profile = deviceProfile();
    try std.testing.expect(!device_profile.simulated);
    try std.testing.expectError(error.InvalidMobileView, layoutSummary(device_profile));
    try std.testing.expectError(error.InvalidMobileView, renderMobileView(std.testing.allocator, "ABI", &.{""}));
    try std.testing.expectError(error.InvalidMobileView, renderMobileView(std.testing.allocator, "bad\x00title", &.{"item"}));
    try std.testing.expectError(error.InvalidTaskName, executeMobileTask(std.testing.allocator, "bad\x00task"));
}
