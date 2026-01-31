//! C-compatible framework lifecycle exports.
//! Provides initialization, shutdown, and version functions for FFI.

const std = @import("std");
const errors = @import("errors.zig");

/// Opaque framework handle for C API.
pub const FrameworkHandle = opaque {};

/// Options struct matching C header (abi_options_t).
pub const Options = extern struct {
    enable_ai: bool = true,
    enable_gpu: bool = true,
    enable_database: bool = true,
    enable_network: bool = true,
    enable_web: bool = true,
    enable_profiling: bool = true,
};

/// Version info struct matching C header (abi_version_t).
pub const VersionInfo = extern struct {
    major: c_int,
    minor: c_int,
    patch: c_int,
    full: [*:0]const u8,
};

// Version constants
const VERSION_MAJOR: c_int = 0;
const VERSION_MINOR: c_int = 5;
const VERSION_PATCH: c_int = 0;
const VERSION_STRING: [*:0]const u8 = "0.5.0";

/// Internal framework state stored alongside the handle.
const FrameworkState = struct {
    allocator: std.mem.Allocator,
    options: Options,
    initialized: bool = false,
};

// Global allocator for C bindings
var gpa = std.heap.GeneralPurposeAllocator(.{}){};

// Track active framework to prevent double-init
var active_framework: ?*FrameworkState = null;

/// Initialize framework with defaults.
pub export fn abi_init(out_framework: *?*FrameworkHandle) errors.Error {
    const opts = Options{};
    return abi_init_with_options(&opts, out_framework);
}

/// Initialize framework with options.
pub export fn abi_init_with_options(
    options: *const Options,
    out_framework: *?*FrameworkHandle,
) errors.Error {
    if (active_framework != null) {
        return errors.ALREADY_INITIALIZED;
    }

    const allocator = gpa.allocator();

    const state = allocator.create(FrameworkState) catch {
        return errors.OUT_OF_MEMORY;
    };

    state.* = .{
        .allocator = allocator,
        .options = options.*,
        .initialized = true,
    };

    active_framework = state;
    out_framework.* = @ptrCast(state);
    return errors.OK;
}

/// Shutdown framework.
pub export fn abi_shutdown(handle: ?*FrameworkHandle) void {
    if (handle) |h| {
        const state: *FrameworkState = @ptrCast(@alignCast(h));
        if (state.initialized) {
            state.initialized = false;
            active_framework = null;
            state.allocator.destroy(state);
        }
    }
}

/// Get version string.
pub export fn abi_version() [*:0]const u8 {
    return VERSION_STRING;
}

/// Get detailed version info.
pub export fn abi_version_info(out_version: *VersionInfo) void {
    out_version.* = .{
        .major = VERSION_MAJOR,
        .minor = VERSION_MINOR,
        .patch = VERSION_PATCH,
        .full = VERSION_STRING,
    };
}

/// Check if feature is enabled.
pub export fn abi_is_feature_enabled(
    handle: ?*FrameworkHandle,
    feature: [*:0]const u8,
) bool {
    if (handle) |h| {
        const state: *FrameworkState = @ptrCast(@alignCast(h));
        const feature_str = std.mem.span(feature);

        if (std.mem.eql(u8, feature_str, "ai")) return state.options.enable_ai;
        if (std.mem.eql(u8, feature_str, "gpu")) return state.options.enable_gpu;
        if (std.mem.eql(u8, feature_str, "database")) return state.options.enable_database;
        if (std.mem.eql(u8, feature_str, "network")) return state.options.enable_network;
        if (std.mem.eql(u8, feature_str, "web")) return state.options.enable_web;
        if (std.mem.eql(u8, feature_str, "profiling")) return state.options.enable_profiling;
    }
    return false;
}

test "framework exports" {
    // Reset state for test
    active_framework = null;

    var handle: ?*FrameworkHandle = null;

    // Test version
    const ver = abi_version();
    try std.testing.expectEqualStrings("0.5.0", std.mem.span(ver));

    var info: VersionInfo = undefined;
    abi_version_info(&info);
    try std.testing.expectEqual(@as(c_int, 0), info.major);
    try std.testing.expectEqual(@as(c_int, 5), info.minor);

    // Test init/shutdown
    try std.testing.expectEqual(errors.OK, abi_init(&handle));
    try std.testing.expect(handle != null);

    // Double init should fail
    var handle2: ?*FrameworkHandle = null;
    try std.testing.expectEqual(errors.ALREADY_INITIALIZED, abi_init(&handle2));

    // Test feature check
    try std.testing.expect(abi_is_feature_enabled(handle, "ai"));
    try std.testing.expect(abi_is_feature_enabled(handle, "gpu"));
    try std.testing.expect(!abi_is_feature_enabled(handle, "unknown"));

    // Test shutdown
    abi_shutdown(handle);
    try std.testing.expectEqual(@as(?*FrameworkState, null), active_framework);

    // Shutdown null should be safe
    abi_shutdown(null);
}
