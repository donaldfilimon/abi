//! Registry Stub
//!
//! Placeholder implementation when the registry feature is disabled.
//! Provides the same API surface as mod.zig but with minimal functionality.
//!
//! All feature queries return false, and registration operations return
//! error.FeatureDisabled.

const std = @import("std");
const feature_catalog = @import("../feature_catalog.zig");
const types_mod = @import("types.zig");

// ============================================================================
// Type Exports (mirrors mod.zig)
// ============================================================================

/// Available features in the framework.
pub const Feature = feature_catalog.Feature;

/// Registration mode determines how features are discovered and managed.
pub const RegistrationMode = types_mod.RegistrationMode;

/// Feature registration entry (stub version).
pub const FeatureRegistration = types_mod.FeatureRegistration;

pub fn description(feature: Feature) []const u8 {
    return feature_catalog.description(feature);
}

// ============================================================================
// Compile-time Feature Checking (stub versions)
// ============================================================================

/// Stub: Always returns false when registry is disabled.
pub fn isFeatureCompiledIn(comptime feature: Feature) bool {
    _ = feature;
    return false;
}

/// Get parent feature for sub-features.
pub fn getParentFeature(feature: Feature) ?Feature {
    return feature_catalog.parentAsEnum(Feature, feature);
}

// ============================================================================
// Registry Struct (stub)
// ============================================================================

/// Stub Registry providing the same API surface as the real implementation.
/// All operations either return false or error.FeatureDisabled.
pub const Registry = struct {
    /// Error type for Registry operations.
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

    /// Initialize an empty stub registry.
    pub fn init(allocator: std.mem.Allocator) Registry {
        return .{
            .allocator = allocator,
        };
    }

    /// Cleanup (no-op in stub).
    pub fn deinit(self: *Registry) void {
        _ = self;
    }

    // ========================================================================
    // Registration API (all return FeatureDisabled)
    // ========================================================================

    /// Stub: Returns error.FeatureDisabled.
    pub fn registerComptime(self: *Registry, comptime feature: Feature) Error!void {
        _ = self;
        _ = feature;
        return error.FeatureDisabled;
    }

    /// Stub: Returns error.FeatureDisabled.
    pub fn registerRuntimeToggle(
        self: *Registry,
        comptime feature: Feature,
        comptime ContextType: type,
        config_ptr: *const anyopaque,
    ) Error!void {
        _ = self;
        _ = feature;
        _ = ContextType;
        _ = config_ptr;
        return error.FeatureDisabled;
    }

    /// Stub: Returns error.FeatureDisabled.
    pub fn registerDynamic(
        self: *Registry,
        feature: Feature,
        library_path: []const u8,
    ) Error!void {
        _ = self;
        _ = feature;
        _ = library_path;
        return error.FeatureDisabled;
    }

    // ========================================================================
    // Lifecycle Management (all return FeatureDisabled)
    // ========================================================================

    /// Stub: Returns error.FeatureDisabled.
    pub fn initFeature(self: *Registry, feature: Feature) Error!void {
        _ = self;
        _ = feature;
        return error.FeatureDisabled;
    }

    /// Stub: Returns error.FeatureDisabled.
    pub fn deinitFeature(self: *Registry, feature: Feature) Error!void {
        _ = self;
        _ = feature;
        return error.FeatureDisabled;
    }

    /// Stub: Returns error.FeatureDisabled.
    pub fn enableFeature(self: *Registry, feature: Feature) Error!void {
        _ = self;
        _ = feature;
        return error.FeatureDisabled;
    }

    /// Stub: Returns error.FeatureDisabled.
    pub fn disableFeature(self: *Registry, feature: Feature) Error!void {
        _ = self;
        _ = feature;
        return error.FeatureDisabled;
    }

    // ========================================================================
    // Query API (return defaults indicating disabled state)
    // ========================================================================

    /// Stub: Always returns false.
    pub fn isRegistered(self: *const Registry, feature: Feature) bool {
        _ = self;
        _ = feature;
        return false;
    }

    /// Stub: Always returns false.
    pub fn isEnabled(self: *const Registry, feature: Feature) bool {
        _ = self;
        _ = feature;
        return false;
    }

    /// Stub: Always returns false.
    pub fn isInitialized(self: *const Registry, feature: Feature) bool {
        _ = self;
        _ = feature;
        return false;
    }

    /// Stub: Always returns null.
    pub fn getMode(self: *const Registry, feature: Feature) ?RegistrationMode {
        _ = self;
        _ = feature;
        return null;
    }

    /// Stub: Returns error.FeatureDisabled.
    pub fn getContext(
        self: *const Registry,
        feature: Feature,
        comptime ContextType: type,
    ) Error!*ContextType {
        _ = self;
        _ = feature;
        return error.FeatureDisabled;
    }

    /// Stub: Returns an empty slice.
    pub fn listFeatures(self: *const Registry, allocator: std.mem.Allocator) Error![]Feature {
        _ = self;
        return allocator.alloc(Feature, 0);
    }

    /// Stub: Always returns 0.
    pub fn count(self: *const Registry) usize {
        _ = self;
        return 0;
    }
};

// ============================================================================
// Stub Sub-modules (empty placeholders for API compatibility)
// ============================================================================

/// Stub types module.
pub const types = struct {
    pub const Feature = types_mod.Feature;
    pub const RegistrationMode = types_mod.RegistrationMode;
    pub const FeatureRegistration = types_mod.FeatureRegistration;
    pub const Error = Registry.Error;
    pub const isFeatureCompiledIn = isFeatureCompiledIn;
    pub const getParentFeature = getParentFeature;
};

/// Stub registration module.
pub const registration = struct {
    pub fn registerComptime(
        allocator: std.mem.Allocator,
        registrations: anytype,
        comptime feature: Feature,
    ) Registry.Error!void {
        _ = allocator;
        _ = registrations;
        _ = feature;
        return error.FeatureDisabled;
    }

    pub fn registerRuntimeToggle(
        allocator: std.mem.Allocator,
        registrations: anytype,
        comptime feature: Feature,
        comptime ContextType: type,
        config_ptr: *const anyopaque,
    ) Registry.Error!void {
        _ = allocator;
        _ = registrations;
        _ = feature;
        _ = ContextType;
        _ = config_ptr;
        return error.FeatureDisabled;
    }

    pub fn registerDynamic(
        allocator: std.mem.Allocator,
        registrations: anytype,
        feature: Feature,
        library_path: []const u8,
    ) Registry.Error!void {
        _ = allocator;
        _ = registrations;
        _ = feature;
        _ = library_path;
        return error.FeatureDisabled;
    }
};

/// Stub plugin module.
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

        pub fn name(self: PluginCapability) []const u8 {
            return @tagName(self);
        }
    };

    pub const PluginState = enum {
        registered,
        loading,
        active,
        unloading,
        failed,
    };

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

    pub const PluginError = error{
        PluginAlreadyRegistered,
        PluginNotFound,
        PluginLoadFailed,
        OutOfMemory,
    };

    pub const PluginRegistry = struct {
        pub fn init() PluginRegistry {
            return .{};
        }

        pub fn deinit(self: *PluginRegistry, allocator: std.mem.Allocator) void {
            _ = self;
            _ = allocator;
        }

        pub fn register(self: *PluginRegistry, allocator: std.mem.Allocator, descriptor: PluginDescriptor, callbacks: PluginCallbacks) PluginError!void {
            _ = self;
            _ = allocator;
            _ = descriptor;
            _ = callbacks;
        }

        pub fn unregister(self: *PluginRegistry, allocator: std.mem.Allocator, plugin_name: []const u8) PluginError!void {
            _ = self;
            _ = allocator;
            _ = plugin_name;
        }

        pub fn get(self: *PluginRegistry, plugin_name: []const u8) ?*PluginEntry {
            _ = self;
            _ = plugin_name;
            return null;
        }

        pub fn list(self: *const PluginRegistry, allocator: std.mem.Allocator) PluginError![]const PluginDescriptor {
            _ = self;
            return allocator.alloc(PluginDescriptor, 0) catch return PluginError.OutOfMemory;
        }

        pub fn countByCapability(self: *const PluginRegistry, capability: PluginCapability) usize {
            _ = self;
            _ = capability;
            return 0;
        }

        pub fn count(self: *const PluginRegistry) usize {
            _ = self;
            return 0;
        }
    };
};

/// Stub lifecycle module.
pub const lifecycle = struct {
    pub fn initFeature(
        allocator: std.mem.Allocator,
        registrations: anytype,
        feature: Feature,
    ) Registry.Error!void {
        _ = allocator;
        _ = registrations;
        _ = feature;
        return error.FeatureDisabled;
    }

    pub fn deinitFeature(registrations: anytype, feature: Feature) Registry.Error!void {
        _ = registrations;
        _ = feature;
        return error.FeatureDisabled;
    }

    pub fn enableFeature(
        allocator: std.mem.Allocator,
        registrations: anytype,
        runtime_overrides: anytype,
        feature: Feature,
    ) Registry.Error!void {
        _ = allocator;
        _ = registrations;
        _ = runtime_overrides;
        _ = feature;
        return error.FeatureDisabled;
    }

    pub fn disableFeature(
        allocator: std.mem.Allocator,
        registrations: anytype,
        runtime_overrides: anytype,
        feature: Feature,
    ) Registry.Error!void {
        _ = allocator;
        _ = registrations;
        _ = runtime_overrides;
        _ = feature;
        return error.FeatureDisabled;
    }
};

// ============================================================================
// Tests
// ============================================================================

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
