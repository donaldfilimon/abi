//! Registry Stub
//!
//! Placeholder implementation when the registry feature is disabled.
//! Provides the same API surface as mod.zig but with minimal functionality.
//!
//! All feature queries return false, and registration operations return
//! error.RegistryDisabled.

const std = @import("std");

// ============================================================================
// Type Exports (mirrors mod.zig)
// ============================================================================

/// Available features in the framework.
pub const Feature = enum {
    gpu,
    ai,
    llm,
    embeddings,
    agents,
    training,
    database,
    network,
    observability,
    web,
    personas,
    cloud,
    analytics,
    auth,
    messaging,
    cache,
    storage,
    search,
    mobile,
    gateway,
    pages,
    benchmarks,
    reasoning,
    constitution,

    pub fn name(self: Feature) []const u8 {
        return std.mem.sliceTo(@tagName(self), 0);
    }

    pub fn description(self: Feature) []const u8 {
        return switch (self) {
            .gpu => "GPU acceleration and compute",
            .ai => "AI core functionality",
            .llm => "Local LLM inference",
            .embeddings => "Vector embeddings generation",
            .agents => "AI agent runtime",
            .training => "Model training pipelines",
            .database => "Vector database (WDBX)",
            .network => "Distributed compute network",
            .observability => "Metrics, tracing, profiling",
            .web => "Web/HTTP utilities",
            .personas => "Multi-persona AI assistant",
            .cloud => "Cloud provider integration",
            .analytics => "Analytics event tracking",
            .auth => "Authentication and security",
            .messaging => "Event bus and messaging",
            .cache => "In-memory caching",
            .storage => "Unified file/object storage",
            .search => "Full-text search",
            .mobile => "Mobile platform support",
            .gateway => "API gateway",
            .pages => "Dashboard/UI pages",
            .benchmarks => "Performance benchmarking",
            .reasoning => "AI reasoning (Abbey, eval, RAG)",
            .constitution => "AI safety principles and guardrails",
        };
    }
};

/// Registration mode determines how features are discovered and managed.
pub const RegistrationMode = enum {
    /// Features resolved at compile time only.
    comptime_only,
    /// Features compiled in but can be enabled/disabled at runtime.
    runtime_toggle,
    /// Features can be dynamically loaded from shared libraries at runtime.
    dynamic,
};

/// Feature registration entry (stub version).
pub const FeatureRegistration = struct {
    feature: Feature,
    mode: RegistrationMode,
    context_ptr: ?*anyopaque = null,
    config_ptr: ?*const anyopaque = null,
    init_fn: ?*const fn (std.mem.Allocator, *const anyopaque) anyerror!*anyopaque = null,
    deinit_fn: ?*const fn (*anyopaque) void = null,
    library_handle: ?*anyopaque = null,
    library_path: ?[]const u8 = null,
    enabled: bool = false,
    initialized: bool = false,
};

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
    return switch (feature) {
        .llm, .embeddings, .agents, .training, .personas, .reasoning, .constitution => .ai,
        else => null,
    };
}

// ============================================================================
// Registry Struct (stub)
// ============================================================================

/// Stub Registry providing the same API surface as the real implementation.
/// All operations either return false or error.RegistryDisabled.
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
        RegistryDisabled,
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
    // Registration API (all return RegistryDisabled)
    // ========================================================================

    /// Stub: Returns error.RegistryDisabled.
    pub fn registerComptime(self: *Registry, comptime feature: Feature) Error!void {
        _ = self;
        _ = feature;
        return error.RegistryDisabled;
    }

    /// Stub: Returns error.RegistryDisabled.
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
        return error.RegistryDisabled;
    }

    /// Stub: Returns error.RegistryDisabled.
    pub fn registerDynamic(
        self: *Registry,
        feature: Feature,
        library_path: []const u8,
    ) Error!void {
        _ = self;
        _ = feature;
        _ = library_path;
        return error.RegistryDisabled;
    }

    // ========================================================================
    // Lifecycle Management (all return RegistryDisabled)
    // ========================================================================

    /// Stub: Returns error.RegistryDisabled.
    pub fn initFeature(self: *Registry, feature: Feature) Error!void {
        _ = self;
        _ = feature;
        return error.RegistryDisabled;
    }

    /// Stub: Returns error.RegistryDisabled.
    pub fn deinitFeature(self: *Registry, feature: Feature) Error!void {
        _ = self;
        _ = feature;
        return error.RegistryDisabled;
    }

    /// Stub: Returns error.RegistryDisabled.
    pub fn enableFeature(self: *Registry, feature: Feature) Error!void {
        _ = self;
        _ = feature;
        return error.RegistryDisabled;
    }

    /// Stub: Returns error.RegistryDisabled.
    pub fn disableFeature(self: *Registry, feature: Feature) Error!void {
        _ = self;
        _ = feature;
        return error.RegistryDisabled;
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

    /// Stub: Returns error.RegistryDisabled.
    pub fn getContext(
        self: *const Registry,
        feature: Feature,
        comptime ContextType: type,
    ) Error!*ContextType {
        _ = self;
        _ = feature;
        return error.RegistryDisabled;
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
    pub const Feature = @This().Feature;
    pub const RegistrationMode = @This().RegistrationMode;
    pub const FeatureRegistration = @This().FeatureRegistration;
    pub const Error = Registry.Error;
    pub const isFeatureCompiledIn = @This().isFeatureCompiledIn;
    pub const getParentFeature = @This().getParentFeature;
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
        return error.RegistryDisabled;
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
        return error.RegistryDisabled;
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
        return error.RegistryDisabled;
    }
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
        return error.RegistryDisabled;
    }

    pub fn deinitFeature(registrations: anytype, feature: Feature) Registry.Error!void {
        _ = registrations;
        _ = feature;
        return error.RegistryDisabled;
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
        return error.RegistryDisabled;
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
        return error.RegistryDisabled;
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

test "stub registerComptime returns RegistryDisabled" {
    var reg = Registry.init(std.testing.allocator);
    defer reg.deinit();

    try std.testing.expectError(error.RegistryDisabled, reg.registerComptime(.gpu));
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
    try std.testing.expect(getParentFeature(.gpu) == null);
    try std.testing.expect(getParentFeature(.ai) == null);
}
