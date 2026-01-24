//! Feature Registration
//!
//! Registration APIs for different feature modes: comptime, runtime toggle, and dynamic.

const std = @import("std");
const types = @import("types.zig");

const Feature = types.Feature;
const RegistrationMode = types.RegistrationMode;
const FeatureRegistration = types.FeatureRegistration;
const Error = types.Error;
const isFeatureCompiledIn = types.isFeatureCompiledIn;

/// Register a feature for comptime-only resolution.
/// The feature must be enabled at compile time via build_options.
/// This is zero-overhead - just validates feature exists at comptime.
pub fn registerComptime(
    allocator: std.mem.Allocator,
    registrations: *std.AutoHashMapUnmanaged(Feature, FeatureRegistration),
    comptime feature: Feature,
) Error!void {
    // Compile-time check that feature is enabled
    if (!comptime isFeatureCompiledIn(feature)) {
        @compileError("Feature " ++ @tagName(feature) ++ " not enabled at compile time");
    }

    // Check if already registered
    if (registrations.contains(feature)) {
        return Error.FeatureAlreadyRegistered;
    }

    // Register in map
    try registrations.put(allocator, feature, .{
        .feature = feature,
        .mode = .comptime_only,
        .enabled = true, // Comptime features are always enabled
        .initialized = false,
    });
}

/// Register a feature with runtime toggle capability.
/// Feature must be compiled in, but can be enabled/disabled at runtime.
pub fn registerRuntimeToggle(
    allocator: std.mem.Allocator,
    registrations: *std.AutoHashMapUnmanaged(Feature, FeatureRegistration),
    comptime feature: Feature,
    comptime ContextType: type,
    config_ptr: *const anyopaque,
) Error!void {
    // Compile-time validation
    if (!comptime isFeatureCompiledIn(feature)) {
        @compileError("Feature " ++ @tagName(feature) ++ " not compiled in");
    }

    // Check if already registered
    if (registrations.contains(feature)) {
        return Error.FeatureAlreadyRegistered;
    }

    // Create type-erased init/deinit wrappers
    const Wrapper = struct {
        fn initWrapper(alloc: std.mem.Allocator, cfg_ptr: *const anyopaque) anyerror!*anyopaque {
            _ = cfg_ptr; // Config handled by ContextType.init
            const ctx = try ContextType.init(alloc);
            return @ptrCast(ctx);
        }

        fn deinitWrapper(context_ptr: *anyopaque) void {
            const ctx: *ContextType = @ptrCast(@alignCast(context_ptr));
            ctx.deinit();
        }
    };

    try registrations.put(allocator, feature, .{
        .feature = feature,
        .mode = .runtime_toggle,
        .config_ptr = config_ptr,
        .init_fn = &Wrapper.initWrapper,
        .deinit_fn = &Wrapper.deinitWrapper,
        .enabled = false, // Disabled by default, must explicitly enable
        .initialized = false,
    });
}

/// Register a feature for dynamic loading from a shared library (future).
pub fn registerDynamic(
    allocator: std.mem.Allocator,
    registrations: *std.AutoHashMapUnmanaged(Feature, FeatureRegistration),
    feature: Feature,
    library_path: []const u8,
) Error!void {
    // Check if already registered
    if (registrations.contains(feature)) {
        return Error.FeatureAlreadyRegistered;
    }

    const path_copy = try allocator.dupe(u8, library_path);
    errdefer allocator.free(path_copy);

    try registrations.put(allocator, feature, .{
        .feature = feature,
        .mode = .dynamic,
        .library_path = path_copy,
        .enabled = false,
        .initialized = false,
    });
}
