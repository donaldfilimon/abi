//! Feature Registry System
//!
//! Provides a unified interface for feature registration and lifecycle management
//! supporting three registration modes:
//!
//! - **Comptime-only**: Zero overhead, features resolved at compile time
//! - **Runtime-toggle**: Compiled in but can be enabled/disabled at runtime
//! - **Dynamic**: Features loaded from shared libraries at runtime (future)
//!
//! ## Usage
//!
//! ```zig
//! const registry = @import("registry/mod.zig");
//!
//! var reg = registry.Registry.init(allocator);
//! defer reg.deinit();
//!
//! // Register features
//! try reg.registerComptime(.gpu);
//! try reg.registerRuntimeToggle(.ai, ai_mod.Context, &ai_config);
//!
//! // Query features
//! if (reg.isEnabled(.gpu)) {
//!     // Use GPU...
//! }
//! ```

const std = @import("std");
const build_options = @import("build_options");

// Import sub-modules
pub const types = @import("types.zig");
pub const registration = @import("registration.zig");
pub const lifecycle = @import("lifecycle.zig");

// Re-export types for backward compatibility
pub const Feature = types.Feature;
pub const RegistrationMode = types.RegistrationMode;
pub const FeatureRegistration = types.FeatureRegistration;

// Re-export compile-time utilities
pub const isFeatureCompiledIn = types.isFeatureCompiledIn;
pub const getParentFeature = types.getParentFeature;

/// Central registry managing feature lifecycle across all registration modes.
pub const Registry = struct {
    /// Error type for Registry operations (alias for backward compatibility).
    pub const Error = types.Error;

    allocator: std.mem.Allocator,

    /// Static registrations (comptime_only, runtime_toggle)
    registrations: std.AutoHashMapUnmanaged(Feature, FeatureRegistration),

    /// Runtime toggles (only used if any features are runtime_toggle)
    runtime_overrides: std.AutoHashMapUnmanaged(Feature, bool),

    /// Initialize empty registry.
    pub fn init(allocator: std.mem.Allocator) Registry {
        return .{
            .allocator = allocator,
            .registrations = .{},
            .runtime_overrides = .{},
        };
    }

    /// Cleanup all registered features and plugin state.
    pub fn deinit(self: *Registry) void {
        // Deinitialize all initialized features
        var iter = self.registrations.valueIterator();
        while (iter.next()) |reg| {
            if (reg.initialized) {
                if (reg.deinit_fn) |deinit_fn| {
                    if (reg.context_ptr) |ptr| {
                        deinit_fn(ptr);
                    }
                }
            }
        }

        self.registrations.deinit(self.allocator);
        self.runtime_overrides.deinit(self.allocator);
    }

    // ========================================================================
    // Registration API (delegates to registration module)
    // ========================================================================

    /// Register a feature for comptime-only resolution.
    /// The feature must be enabled at compile time via build_options.
    /// This is zero-overhead - just validates feature exists at comptime.
    pub fn registerComptime(self: *Registry, comptime feature: Feature) Error!void {
        return registration.registerComptime(self.allocator, &self.registrations, feature);
    }

    /// Register a feature with runtime toggle capability.
    /// Feature must be compiled in, but can be enabled/disabled at runtime.
    pub fn registerRuntimeToggle(
        self: *Registry,
        comptime feature: Feature,
        comptime ContextType: type,
        config_ptr: *const anyopaque,
    ) Error!void {
        return registration.registerRuntimeToggle(self.allocator, &self.registrations, feature, ContextType, config_ptr);
    }

    /// Register a feature for dynamic loading from a shared library (future).
    pub fn registerDynamic(
        self: *Registry,
        feature: Feature,
        library_path: []const u8,
    ) Error!void {
        return registration.registerDynamic(self.allocator, &self.registrations, feature, library_path);
    }

    // ========================================================================
    // Lifecycle Management (delegates to lifecycle module)
    // ========================================================================

    /// Initialize a registered feature. For runtime_toggle and dynamic modes.
    pub fn initFeature(self: *Registry, feature: Feature) Error!void {
        return lifecycle.initFeature(self.allocator, &self.registrations, feature);
    }

    /// Shutdown a feature, releasing resources.
    pub fn deinitFeature(self: *Registry, feature: Feature) Error!void {
        return lifecycle.deinitFeature(&self.registrations, feature);
    }

    /// Enable a runtime-toggleable feature.
    pub fn enableFeature(self: *Registry, feature: Feature) Error!void {
        return lifecycle.enableFeature(self.allocator, &self.registrations, &self.runtime_overrides, feature);
    }

    /// Disable a runtime-toggleable feature. Deinitializes if currently initialized.
    pub fn disableFeature(self: *Registry, feature: Feature) Error!void {
        return lifecycle.disableFeature(self.allocator, &self.registrations, &self.runtime_overrides, feature);
    }

    // ========================================================================
    // Query API
    // ========================================================================

    /// Check if a feature is registered.
    pub fn isRegistered(self: *const Registry, feature: Feature) bool {
        return self.registrations.contains(feature);
    }

    /// Check if a feature is currently enabled.
    /// For comptime_only: always true if registered
    /// For runtime_toggle/dynamic: depends on runtime state
    pub fn isEnabled(self: *const Registry, feature: Feature) bool {
        // Check runtime override first
        if (self.runtime_overrides.get(feature)) |override_val| {
            return override_val;
        }

        if (self.registrations.get(feature)) |reg| {
            return switch (reg.mode) {
                .comptime_only => true, // Always enabled if registered
                .runtime_toggle, .dynamic => reg.enabled,
            };
        }

        return false;
    }

    /// Check if a feature is initialized and ready to use.
    pub fn isInitialized(self: *const Registry, feature: Feature) bool {
        if (self.registrations.get(feature)) |reg| {
            return reg.initialized;
        }
        return false;
    }

    /// Get the registration mode for a feature.
    pub fn getMode(self: *const Registry, feature: Feature) ?RegistrationMode {
        if (self.registrations.get(feature)) |reg| {
            return reg.mode;
        }
        return null;
    }

    /// Get the context for a feature. Returns error if not initialized.
    pub fn getContext(
        self: *const Registry,
        feature: Feature,
        comptime ContextType: type,
    ) Error!*ContextType {
        const reg = self.registrations.get(feature) orelse return Error.FeatureNotRegistered;

        if (!reg.enabled) return Error.FeatureDisabled;
        if (!reg.initialized) return Error.NotInitialized;

        const ptr = reg.context_ptr orelse return Error.NotInitialized;
        return @ptrCast(@alignCast(ptr));
    }

    /// Get list of all registered features.
    pub fn listFeatures(self: *const Registry, allocator: std.mem.Allocator) Error![]Feature {
        var list = std.ArrayList(Feature).init(allocator);
        errdefer list.deinit();

        var iter = self.registrations.keyIterator();
        while (iter.next()) |feature| {
            try list.append(feature.*);
        }

        return list.toOwnedSlice();
    }

    /// Get count of registered features.
    pub fn count(self: *const Registry) usize {
        return self.registrations.count();
    }
};

// ============================================================================
// Tests
// ============================================================================

test "Registry init and deinit" {
    var reg = Registry.init(std.testing.allocator);
    defer reg.deinit();

    try std.testing.expectEqual(@as(usize, 0), reg.count());
}

test "registerComptime adds feature" {
    var reg = Registry.init(std.testing.allocator);
    defer reg.deinit();

    // Register a feature that should be compiled in
    if (comptime build_options.enable_gpu) {
        try reg.registerComptime(.gpu);
        try std.testing.expect(reg.isRegistered(.gpu));
        try std.testing.expect(reg.isEnabled(.gpu));
        try std.testing.expectEqual(RegistrationMode.comptime_only, reg.getMode(.gpu).?);
    }
}

test "registerComptime duplicate returns error" {
    var reg = Registry.init(std.testing.allocator);
    defer reg.deinit();

    if (comptime build_options.enable_gpu) {
        try reg.registerComptime(.gpu);
        try std.testing.expectError(Registry.Error.FeatureAlreadyRegistered, reg.registerComptime(.gpu));
    }
}

test "isEnabled returns false for unregistered" {
    var reg = Registry.init(std.testing.allocator);
    defer reg.deinit();

    try std.testing.expect(!reg.isEnabled(.gpu));
    try std.testing.expect(!reg.isRegistered(.gpu));
}

test "listFeatures returns all registered" {
    var reg = Registry.init(std.testing.allocator);
    defer reg.deinit();

    if (comptime build_options.enable_gpu) {
        try reg.registerComptime(.gpu);
    }
    if (comptime build_options.enable_database) {
        try reg.registerComptime(.database);
    }

    const features = try reg.listFeatures(std.testing.allocator);
    defer std.testing.allocator.free(features);

    var expected_count: usize = 0;
    if (comptime build_options.enable_gpu) expected_count += 1;
    if (comptime build_options.enable_database) expected_count += 1;

    try std.testing.expectEqual(expected_count, features.len);
}

test "isFeatureCompiledIn matches build_options" {
    try std.testing.expectEqual(build_options.enable_gpu, comptime isFeatureCompiledIn(.gpu));
    try std.testing.expectEqual(build_options.enable_ai, comptime isFeatureCompiledIn(.ai));
    try std.testing.expectEqual(build_options.enable_database, comptime isFeatureCompiledIn(.database));
}

test "getParentFeature returns correct parent" {
    try std.testing.expectEqual(Feature.ai, getParentFeature(.llm).?);
    try std.testing.expectEqual(Feature.ai, getParentFeature(.embeddings).?);
    try std.testing.expectEqual(Feature.ai, getParentFeature(.agents).?);
    try std.testing.expectEqual(Feature.ai, getParentFeature(.training).?);
    try std.testing.expect(getParentFeature(.gpu) == null);
    try std.testing.expect(getParentFeature(.ai) == null);
}

// ============================================================================
// Runtime Toggle Tests
// ============================================================================

/// Mock context for testing runtime toggle functionality
const MockContext = struct {
    allocator: std.mem.Allocator,
    initialized: bool = true,
    value: u32 = 42,

    pub fn init(allocator: std.mem.Allocator) !*MockContext {
        const ctx = try allocator.create(MockContext);
        ctx.* = .{
            .allocator = allocator,
            .initialized = true,
            .value = 42,
        };
        return ctx;
    }

    pub fn deinit(self: *MockContext) void {
        self.allocator.destroy(self);
    }
};

test "registerRuntimeToggle creates disabled feature" {
    if (!comptime build_options.enable_gpu) return;

    var reg = Registry.init(std.testing.allocator);
    defer reg.deinit();

    var dummy_config: u8 = 0;
    try reg.registerRuntimeToggle(.gpu, MockContext, &dummy_config);

    try std.testing.expect(reg.isRegistered(.gpu));
    try std.testing.expect(!reg.isEnabled(.gpu)); // Disabled by default
    try std.testing.expect(!reg.isInitialized(.gpu));
    try std.testing.expectEqual(RegistrationMode.runtime_toggle, reg.getMode(.gpu).?);
}

test "enableFeature enables runtime toggle feature" {
    if (!comptime build_options.enable_gpu) return;

    var reg = Registry.init(std.testing.allocator);
    defer reg.deinit();

    var dummy_config: u8 = 0;
    try reg.registerRuntimeToggle(.gpu, MockContext, &dummy_config);

    try std.testing.expect(!reg.isEnabled(.gpu));
    try reg.enableFeature(.gpu);
    try std.testing.expect(reg.isEnabled(.gpu));
}

test "disableFeature disables runtime toggle feature" {
    if (!comptime build_options.enable_gpu) return;

    var reg = Registry.init(std.testing.allocator);
    defer reg.deinit();

    var dummy_config: u8 = 0;
    try reg.registerRuntimeToggle(.gpu, MockContext, &dummy_config);

    try reg.enableFeature(.gpu);
    try std.testing.expect(reg.isEnabled(.gpu));

    try reg.disableFeature(.gpu);
    try std.testing.expect(!reg.isEnabled(.gpu));
}

test "initFeature initializes context" {
    if (!comptime build_options.enable_gpu) return;

    var reg = Registry.init(std.testing.allocator);
    defer reg.deinit();

    var dummy_config: u8 = 0;
    try reg.registerRuntimeToggle(.gpu, MockContext, &dummy_config);

    try reg.enableFeature(.gpu);
    try reg.initFeature(.gpu);

    try std.testing.expect(reg.isInitialized(.gpu));
}

test "getContext returns initialized context" {
    if (!comptime build_options.enable_gpu) return;

    var reg = Registry.init(std.testing.allocator);
    defer reg.deinit();

    var dummy_config: u8 = 0;
    try reg.registerRuntimeToggle(.gpu, MockContext, &dummy_config);

    try reg.enableFeature(.gpu);
    try reg.initFeature(.gpu);

    const ctx = try reg.getContext(.gpu, MockContext);
    try std.testing.expectEqual(@as(u32, 42), ctx.value);
}

test "getContext fails when not enabled" {
    if (!comptime build_options.enable_gpu) return;

    var reg = Registry.init(std.testing.allocator);
    defer reg.deinit();

    var dummy_config: u8 = 0;
    try reg.registerRuntimeToggle(.gpu, MockContext, &dummy_config);

    try std.testing.expectError(Registry.Error.FeatureDisabled, reg.getContext(.gpu, MockContext));
}

test "getContext fails when not initialized" {
    if (!comptime build_options.enable_gpu) return;

    var reg = Registry.init(std.testing.allocator);
    defer reg.deinit();

    var dummy_config: u8 = 0;
    try reg.registerRuntimeToggle(.gpu, MockContext, &dummy_config);

    try reg.enableFeature(.gpu);
    // Don't init

    try std.testing.expectError(Registry.Error.NotInitialized, reg.getContext(.gpu, MockContext));
}

test "deinitFeature cleans up context" {
    if (!comptime build_options.enable_gpu) return;

    var reg = Registry.init(std.testing.allocator);
    defer reg.deinit();

    var dummy_config: u8 = 0;
    try reg.registerRuntimeToggle(.gpu, MockContext, &dummy_config);

    try reg.enableFeature(.gpu);
    try reg.initFeature(.gpu);
    try std.testing.expect(reg.isInitialized(.gpu));

    try reg.deinitFeature(.gpu);
    try std.testing.expect(!reg.isInitialized(.gpu));
}

test "disableFeature auto-deinits if initialized" {
    if (!comptime build_options.enable_gpu) return;

    var reg = Registry.init(std.testing.allocator);
    defer reg.deinit();

    var dummy_config: u8 = 0;
    try reg.registerRuntimeToggle(.gpu, MockContext, &dummy_config);

    try reg.enableFeature(.gpu);
    try reg.initFeature(.gpu);
    try std.testing.expect(reg.isInitialized(.gpu));

    try reg.disableFeature(.gpu);
    try std.testing.expect(!reg.isInitialized(.gpu));
    try std.testing.expect(!reg.isEnabled(.gpu));
}

test "initFeature fails when disabled" {
    if (!comptime build_options.enable_gpu) return;

    var reg = Registry.init(std.testing.allocator);
    defer reg.deinit();

    var dummy_config: u8 = 0;
    try reg.registerRuntimeToggle(.gpu, MockContext, &dummy_config);

    // Don't enable
    try std.testing.expectError(Registry.Error.FeatureDisabled, reg.initFeature(.gpu));
}

test "disableFeature fails for comptime_only" {
    if (!comptime build_options.enable_gpu) return;

    var reg = Registry.init(std.testing.allocator);
    defer reg.deinit();

    try reg.registerComptime(.gpu);

    try std.testing.expectError(Registry.Error.InvalidMode, reg.disableFeature(.gpu));
}
