const std = @import("std");
const framework = @import("framework/mod.zig");
const core = @import("shared/core/core.zig");
const lifecycle_mod = @import("shared/core/lifecycle.zig");

pub const features = @import("features/mod.zig");
pub const shared = @import("shared/mod.zig");
pub const utils = @import("shared/utils/mod.zig");

pub const FeatureCategory = framework.feature_manager.FeatureCategory;
pub const Runtime = framework.runtime.Runtime;
pub const RuntimeOptions = Runtime.Options;

const RuntimeInitError = std.mem.Allocator.Error || framework.feature_manager.Error || core.AbiError;

pub const InitError = (error{
    AlreadyInitialized,
    NotInitialized,
}) || RuntimeInitError;

var global_runtime: ?*Runtime = null;

/// Initialize the ABI framework with default runtime options.
pub fn init(allocator: std.mem.Allocator) InitError!void {
    return initWithOptions(allocator, .{});
}

/// Initialize the ABI framework using explicit runtime options.
pub fn initWithOptions(allocator: std.mem.Allocator, options: RuntimeOptions) InitError!void {
    if (global_runtime != null) return InitError.AlreadyInitialized;
    const instance = try allocator.create(Runtime);
    errdefer allocator.destroy(instance);
    instance.* = try Runtime.init(allocator, options);
    global_runtime = instance;
}

/// Retrieve the global runtime instance.
pub fn runtime() InitError!*Runtime {
    if (global_runtime) |instance| {
        return instance;
    }
    return InitError.NotInitialized;
}

/// Shut down the global runtime instance if it exists.
pub fn deinit() void {
    if (global_runtime) |instance| {
        const allocator = instance.gpa;
        instance.deinit();
        allocator.destroy(instance);
        global_runtime = null;
    }
    core.deinit();
}

/// Return framework semantic version.
pub fn version() []const u8 {
    return "2.0.0-alpha";
}

/// Convenience accessor for the feature manager.
pub fn featuresManager() InitError!*framework.feature_manager.FeatureManager {
    return runtime().getFeatureManager();
}

/// Convenience accessor for the lifecycle controller.
pub fn lifecycle() InitError!*lifecycle_mod.Lifecycle {
    return runtime().getLifecycle();
}

/// Ensure all registered features are initialized.
pub fn ensureAllFeatures() InitError!void {
    try runtime().getFeatureManager().ensureAll();
}

/// Ensure a specific feature group is initialized.
pub fn ensureCategory(category: FeatureCategory) InitError!void {
    try runtime().getFeatureManager().ensureCategory(category);
}

test "framework bootstraps runtime" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    try init(arena.allocator());
    defer deinit();

    const manager = try featuresManager();
    try manager.ensure("feature.ai");
    try manager.ensure("feature.web");
}
