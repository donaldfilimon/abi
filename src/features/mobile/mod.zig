const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const validation = @import("../../foundation/validation.zig");
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
    /// Reflects the selected GPU backend on this target. This does not imply
    /// native mobile runtime dispatch; `native_dispatch` is the authority for
    /// that capability.
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
    const mobile_profile = profile();
    return .{
        .platform = mobile_profile.platform,
        .os_version = try allocator.dupe(u8, runtimeModeName(mobile_profile.mode)),
        .screen = mobile_profile.screen,
        .hardware_model = try allocator.dupe(u8, mobile_profile.hardware_model),
        .owned = true,
    };
}

fn screenForPlatform(platform: Platform) ScreenConfig {
    return switch (platform) {
        .ios => .{ .width = 390, .height = 844, .density = 3.0 },
        .android => .{ .width = 412, .height = 915, .density = 2.75 },
        .unknown => .{ .width = 375, .height = 812, .density = 3.0 },
    };
}

fn hardwareModelForPlatform(platform: Platform) []const u8 {
    return switch (platform) {
        .ios => "ios-platform",
        .android => "android-platform",
        .unknown => "simulated-mobile-profile",
    };
}

pub fn profile() MobileProfile {
    const status = detectPlatform();
    const mode: RuntimeMode = switch (status.platform) {
        .ios, .android => .native_platform,
        .unknown => .simulated_profile,
    };
    return .{
        .platform = status.platform,
        .mode = mode,
        .screen = screenForPlatform(status.platform),
        .hardware_model = hardwareModelForPlatform(status.platform),
        .native_dispatch = false,
        .message = status.message,
    };
}

pub fn deviceProfile() DeviceProfile {
    const mobile_profile = profile();
    return .{
        .platform = mobile_profile.platform,
        .mode = mobile_profile.mode,
        .native_dispatch = mobile_profile.native_dispatch,
        .simulated = mobile_profile.mode == .simulated_profile,
        .accelerated = detectPlatform().accelerated,
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
    validation.validateNonEmptySlice(value) catch return err;
    validation.validateNoNullBytes(value) catch return err;
}

pub fn renderMobileView(allocator: std.mem.Allocator, title: []const u8, items: []const []const u8) ![]u8 {
    try validateLabel(title, error.InvalidMobileView);
    for (items) |item| {
        try validateLabel(item, error.InvalidMobileView);
    }

    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    const mobile_profile = profile();
    var layout = deviceProfile();
    layout.item_count = items.len;
    layout = try layoutSummary(layout);
    try out.print(
        allocator,
        "[{s}/{s}] {s}\nviewport={d}x{d}@{d:.1} mode={s} simulated={any} native_dispatch={any}\n",
        .{
            platformName(layout.platform),
            runtimeModeName(layout.mode),
            title,
            layout.width,
            layout.height,
            layout.density,
            runtimeModeName(mobile_profile.mode),
            layout.simulated,
            mobile_profile.native_dispatch,
        },
    );
    for (items) |item| {
        try out.print(allocator, "  - {s}\n", .{item});
    }
    try out.print(allocator, "items={d}\nstatus: {s}\n", .{ layout.item_count, mobile_profile.message });

    return try out.toOwnedSlice(allocator);
}

pub fn executeMobileTask(allocator: std.mem.Allocator, task_name: []const u8) ![]u8 {
    try validateLabel(task_name, error.InvalidTaskName);
    const mobile_profile = profile();
    return try std.fmt.allocPrint(
        allocator,
        "mobile task '{s}' accepted on {s}; mode={s}; native_dispatch={any}; execution completed",
        .{ task_name, platformName(mobile_profile.platform), runtimeModeName(mobile_profile.mode), mobile_profile.native_dispatch },
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
    try std.testing.expect(std.mem.indexOf(u8, rendered, "items=2") != null);
    try std.testing.expectError(error.InvalidMobileView, renderMobileView(std.testing.allocator, "Test", &.{""}));
    try std.testing.expectError(error.InvalidMobileView, renderMobileView(std.testing.allocator, "bad\x00title", &.{"item"}));
}

test "mobile task execution validates input" {
    try std.testing.expectError(error.InvalidTaskName, executeMobileTask(std.testing.allocator, ""));
    try std.testing.expectError(error.InvalidTaskName, executeMobileTask(std.testing.allocator, "bad\x00task"));
    const result = try executeMobileTask(std.testing.allocator, "sync");
    defer std.testing.allocator.free(result);
    try std.testing.expect(std.mem.indexOf(u8, result, "sync") != null);
}

test "mobile profile summarizes viewport and dispatch mode" {
    var device_profile = deviceProfile();
    device_profile.item_count = 3;
    const summary = try layoutSummary(device_profile);
    try std.testing.expect(summary.width > 0);
    try std.testing.expect(summary.height > 0);
    try std.testing.expect(summary.density > 0);
    try std.testing.expectEqual(@as(usize, 3), summary.item_count);
    const mobile_profile = @This().profile();
    try std.testing.expect(mobile_profile.message.len > 0);
    try std.testing.expect(mobile_profile.hardware_model.len > 0);
    try std.testing.expect(!mobile_profile.native_dispatch);
    try std.testing.expect(mobile_profile.mode == .native_platform or mobile_profile.mode == .simulated_profile);
    try std.testing.expectError(error.InvalidMobileView, layoutSummary(.{
        .platform = .unknown,
        .mode = .simulated_profile,
        .native_dispatch = false,
        .simulated = true,
        .accelerated = false,
        .width = 0,
        .height = 812,
        .density = 3.0,
    }));
}
