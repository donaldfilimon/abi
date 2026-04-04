//! Registry Stub — active when registry feature is disabled.
//! Provides the same API surface as mod.zig but with minimal functionality.

const std = @import("std");
const feature_catalog = @import("../feature_catalog.zig");
const types_mod = @import("types.zig");

pub const Feature = feature_catalog.Feature;
pub const RegistrationMode = types_mod.RegistrationMode;
pub const FeatureRegistration = types_mod.FeatureRegistration;

pub fn description(feature: Feature) []const u8 {
    return feature_catalog.description(feature);
}

pub fn isFeatureCompiledIn(comptime feature: Feature) bool {
    _ = feature;
    return false;
}

pub fn getParentFeature(feature: Feature) ?Feature {
    return feature_catalog.parentAsEnum(Feature, feature);
}

/// Stub Registry — all operations return FeatureDisabled or defaults.
pub const Registry = struct {
    pub const Error = error{
        FeatureNotRegistered,
        FeatureAlreadyRegistered,
        FeatureNotCompiled,
        FeatureDisabled,
        InitializationFailed,
        AlreadyInitialized,
        NotInitialized,
        DynamicLoadingNotSupported,
        LibraryLoadFailed,
        SymbolNotFound,
        InvalidMode,
        OutOfMemory,
    };

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Registry {
        return .{ .allocator = allocator };
    }
    pub fn deinit(_: *Registry) void {}
    pub fn registerComptime(_: *Registry, _: Feature) Error!void {
        return error.FeatureDisabled;
    }
    pub fn registerRuntimeToggle(_: *Registry, _: Feature, _: type, _: *const anyopaque) Error!void {
        return error.FeatureDisabled;
    }
    pub fn registerDynamic(_: *Registry, _: Feature, _: []const u8) Error!void {
        return error.FeatureDisabled;
    }
    pub fn initFeature(_: *Registry, _: Feature) Error!void {
        return error.FeatureDisabled;
    }
    pub fn deinitFeature(_: *Registry, _: Feature) Error!void {
        return error.FeatureDisabled;
    }
    pub fn enableFeature(_: *Registry, _: Feature) Error!void {
        return error.FeatureDisabled;
    }
    pub fn disableFeature(_: *Registry, _: Feature) Error!void {
        return error.FeatureDisabled;
    }
    pub fn isRegistered(_: *const Registry, _: Feature) bool {
        return false;
    }
    pub fn isEnabled(_: *const Registry, _: Feature) bool {
        return false;
    }
    pub fn isInitialized(_: *const Registry, _: Feature) bool {
        return false;
    }
    pub fn getMode(_: *const Registry, _: Feature) ?RegistrationMode {
        return null;
    }
    pub fn getContext(_: *const Registry, _: Feature, comptime T: type) Error!*T {
        return error.FeatureDisabled;
    }
    pub fn listFeatures(_: *const Registry, allocator: std.mem.Allocator) Error![]Feature {
        return allocator.alloc(Feature, 0);
    }
    pub fn count(_: *const Registry) usize {
        return 0;
    }
};

// ── Sub-module stubs ───────────────────────────────────────────────────────

pub const types = struct {
    pub const Feature = types_mod.Feature;
    pub const RegistrationMode = types_mod.RegistrationMode;
    pub const FeatureRegistration = types_mod.FeatureRegistration;
    pub const Error = Registry.Error;
    pub fn isFeatureCompiledIn(comptime feature: types_mod.Feature) bool {
        _ = feature;
        return false;
    }
    pub fn getParentFeature(feature: types_mod.Feature) ?types_mod.Feature {
        return feature_catalog.parentAsEnum(types_mod.Feature, feature);
    }
};

pub const registration = struct {
    pub fn registerComptime(_: std.mem.Allocator, _: anytype, _: Feature) Registry.Error!void {
        return error.FeatureDisabled;
    }
    pub fn registerRuntimeToggle(_: std.mem.Allocator, _: anytype, _: Feature, _: type, _: *const anyopaque) Registry.Error!void {
        return error.FeatureDisabled;
    }
    pub fn registerDynamic(_: std.mem.Allocator, _: anytype, _: Feature, _: []const u8) Registry.Error!void {
        return error.FeatureDisabled;
    }
};

pub const plugin = struct {
    pub const PluginCapability = enum {
        ai_provider,
        connector,
        storage_backend,
        gpu_backend,
        inference_engine,
        vector_index,
        auth_provider,
        cache_backend,
        custom,
        pub fn name(self: @This()) []const u8 {
            return @tagName(self);
        }
    };
    pub const PluginState = enum { registered, loading, active, unloading, failed };
    pub const PluginDescriptor = struct {
        name: []const u8,
        version: struct { major: u32, minor: u32, patch: u32 },
        author: []const u8,
        description: []const u8,
        capabilities: []const PluginCapability,
        abi_version: struct { major: u32, minor: u32, patch: u32 },
    };
    pub const PluginCallbacks = struct {
        on_load: ?*const fn () anyerror!void = null,
        on_unload: ?*const fn () void = null,
    };
    pub const PluginEntry = struct {
        descriptor: PluginDescriptor,
        state: PluginState,
        callbacks: PluginCallbacks,
    };
    pub const PluginError = error{ PluginAlreadyRegistered, PluginNotFound, PluginLoadFailed, OutOfMemory };
    pub const PluginRegistry = struct {
        pub fn init() @This() {
            return .{};
        }
        pub fn deinit(_: *@This(), _: std.mem.Allocator) void {}
        pub fn register(_: *@This(), _: std.mem.Allocator, _: PluginDescriptor, _: PluginCallbacks) PluginError!void {}
        pub fn unregister(_: *@This(), _: std.mem.Allocator, _: []const u8) PluginError!void {}
        pub fn get(_: *@This(), _: []const u8) ?*PluginEntry {
            return null;
        }
        pub fn list(_: *const @This(), allocator: std.mem.Allocator) PluginError![]const PluginDescriptor {
            return allocator.alloc(PluginDescriptor, 0) catch return PluginError.OutOfMemory;
        }
        pub fn countByCapability(_: *const @This(), _: PluginCapability) usize {
            return 0;
        }
        pub fn count(_: *const @This()) usize {
            return 0;
        }
    };
};

pub const lifecycle = struct {
    pub fn initFeature(_: std.mem.Allocator, _: anytype, _: Feature) Registry.Error!void {
        return error.FeatureDisabled;
    }
    pub fn deinitFeature(_: anytype, _: Feature) Registry.Error!void {
        return error.FeatureDisabled;
    }
    pub fn enableFeature(_: std.mem.Allocator, _: anytype, _: anytype, _: Feature) Registry.Error!void {
        return error.FeatureDisabled;
    }
    pub fn disableFeature(_: std.mem.Allocator, _: anytype, _: anytype, _: Feature) Registry.Error!void {
        return error.FeatureDisabled;
    }
};

// ── Tests ──────────────────────────────────────────────────────────────────

test "stub Registry init and deinit" {
    var reg = Registry.init(std.testing.allocator);
    defer reg.deinit();
    try std.testing.expectEqual(@as(usize, 0), reg.count());
}

test "stub isEnabled always returns false" {
    var reg = Registry.init(std.testing.allocator);
    defer reg.deinit();
    try std.testing.expect(!reg.isEnabled(.gpu));
    try std.testing.expect(!reg.isEnabled(.ai));
    try std.testing.expect(!reg.isEnabled(.database));
}

test "stub registerComptime returns FeatureDisabled" {
    var reg = Registry.init(std.testing.allocator);
    defer reg.deinit();
    try std.testing.expectError(error.FeatureDisabled, reg.registerComptime(.gpu));
}

test "stub listFeatures returns empty slice" {
    var reg = Registry.init(std.testing.allocator);
    defer reg.deinit();
    const features = try reg.listFeatures(std.testing.allocator);
    defer std.testing.allocator.free(features);
    try std.testing.expectEqual(@as(usize, 0), features.len);
}

test "stub isFeatureCompiledIn always returns false" {
    try std.testing.expect(!comptime isFeatureCompiledIn(.gpu));
    try std.testing.expect(!comptime isFeatureCompiledIn(.ai));
    try std.testing.expect(!comptime isFeatureCompiledIn(.database));
}

test "stub getParentFeature returns correct parent" {
    try std.testing.expectEqual(Feature.ai, getParentFeature(.llm).?);
    try std.testing.expectEqual(Feature.ai, getParentFeature(.embeddings).?);
    try std.testing.expectEqual(Feature.ai, getParentFeature(.agents).?);
    try std.testing.expectEqual(Feature.ai, getParentFeature(.training).?);
    try std.testing.expectEqual(Feature.ai, getParentFeature(.profile).?);
    try std.testing.expectEqual(Feature.ai, getParentFeature(.profiles).?);
    try std.testing.expectEqual(Feature.ai, getParentFeature(.vision).?);
    try std.testing.expectEqual(Feature.ai, getParentFeature(.explore).?);
    try std.testing.expectEqual(Feature.ai, getParentFeature(.pipeline).?);
    try std.testing.expectEqual(Feature.ai, getParentFeature(.ai_database).?);
    try std.testing.expectEqual(Feature.ai, getParentFeature(.ai_documents).?);
    try std.testing.expect(getParentFeature(.gpu) == null);
    try std.testing.expect(getParentFeature(.ai) == null);
}

test {
    std.testing.refAllDecls(@This());
}
