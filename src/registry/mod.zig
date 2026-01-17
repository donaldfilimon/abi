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
const config_module = @import("../config.zig");

pub const Feature = config_module.Feature;

/// Registration mode determines how features are discovered and managed.
pub const RegistrationMode = enum {
    /// Features resolved at compile time only. Zero runtime overhead.
    /// Enabled features are statically known and cannot be toggled.
    comptime_only,

    /// Features compiled in but can be enabled/disabled at runtime.
    /// Small overhead for state checking and conditional initialization.
    runtime_toggle,

    /// Features can be dynamically loaded from shared libraries at runtime.
    /// Requires platform support (dlopen/LoadLibrary). Most flexible but highest overhead.
    dynamic,
};

/// Feature registration entry with lifecycle management.
pub const FeatureRegistration = struct {
    feature: Feature,
    mode: RegistrationMode,

    // For comptime_only and runtime_toggle
    context_ptr: ?*anyopaque = null,
    config_ptr: ?*const anyopaque = null,
    init_fn: ?*const fn (std.mem.Allocator, *const anyopaque) anyerror!*anyopaque = null,
    deinit_fn: ?*const fn (*anyopaque) void = null,

    // For dynamic mode (future)
    library_handle: ?*anyopaque = null,
    library_path: ?[]const u8 = null,

    // Runtime state
    enabled: bool = false,
    initialized: bool = false,
};

/// Central registry managing feature lifecycle across all registration modes.
pub const Registry = struct {
    allocator: std.mem.Allocator,

    /// Static registrations (comptime_only, runtime_toggle)
    registrations: std.AutoHashMapUnmanaged(Feature, FeatureRegistration),

    /// Runtime toggles (only used if any features are runtime_toggle)
    runtime_overrides: std.AutoHashMapUnmanaged(Feature, bool),

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
    } || std.mem.Allocator.Error;

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
    // Registration API
    // ========================================================================

    /// Register a feature for comptime-only resolution.
    /// The feature must be enabled at compile time via build_options.
    /// This is zero-overhead - just validates feature exists at comptime.
    pub fn registerComptime(self: *Registry, comptime feature: Feature) Error!void {
        // Compile-time check that feature is enabled
        if (!comptime isFeatureCompiledIn(feature)) {
            @compileError("Feature " ++ @tagName(feature) ++ " not enabled at compile time");
        }

        // Check if already registered
        if (self.registrations.contains(feature)) {
            return Error.FeatureAlreadyRegistered;
        }

        // Register in map
        try self.registrations.put(self.allocator, feature, .{
            .feature = feature,
            .mode = .comptime_only,
            .enabled = true, // Comptime features are always enabled
            .initialized = false,
        });
    }

    /// Register a feature with runtime toggle capability.
    /// Feature must be compiled in, but can be enabled/disabled at runtime.
    pub fn registerRuntimeToggle(
        self: *Registry,
        comptime feature: Feature,
        comptime ContextType: type,
        config_ptr: *const anyopaque,
    ) Error!void {
        // Compile-time validation
        if (!comptime isFeatureCompiledIn(feature)) {
            @compileError("Feature " ++ @tagName(feature) ++ " not compiled in");
        }

        // Check if already registered
        if (self.registrations.contains(feature)) {
            return Error.FeatureAlreadyRegistered;
        }

        // Create type-erased init/deinit wrappers
        const Wrapper = struct {
            fn initWrapper(allocator: std.mem.Allocator, cfg_ptr: *const anyopaque) anyerror!*anyopaque {
                _ = cfg_ptr; // Config handled by ContextType.init
                const ctx = try ContextType.init(allocator);
                return @ptrCast(ctx);
            }

            fn deinitWrapper(context_ptr: *anyopaque) void {
                const ctx: *ContextType = @ptrCast(@alignCast(context_ptr));
                ctx.deinit();
            }
        };

        try self.registrations.put(self.allocator, feature, .{
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
        self: *Registry,
        feature: Feature,
        library_path: []const u8,
    ) Error!void {
        // Check if already registered
        if (self.registrations.contains(feature)) {
            return Error.FeatureAlreadyRegistered;
        }

        const path_copy = try self.allocator.dupe(u8, library_path);
        errdefer self.allocator.free(path_copy);

        try self.registrations.put(self.allocator, feature, .{
            .feature = feature,
            .mode = .dynamic,
            .library_path = path_copy,
            .enabled = false,
            .initialized = false,
        });
    }

    // ========================================================================
    // Lifecycle Management
    // ========================================================================

    /// Initialize a registered feature. For runtime_toggle and dynamic modes.
    pub fn initFeature(self: *Registry, feature: Feature) Error!void {
        const reg = self.registrations.getPtr(feature) orelse return Error.FeatureNotRegistered;

        if (reg.initialized) return Error.AlreadyInitialized;

        switch (reg.mode) {
            .comptime_only => {
                // Comptime features don't need explicit init via registry
                reg.initialized = true;
            },

            .runtime_toggle => {
                if (!reg.enabled) return Error.FeatureDisabled;

                const init_fn = reg.init_fn orelse return Error.InitializationFailed;
                const config_ptr = reg.config_ptr orelse return Error.InitializationFailed;
                reg.context_ptr = init_fn(self.allocator, config_ptr) catch return Error.InitializationFailed;
                reg.initialized = true;
            },

            .dynamic => {
                // Dynamic loading not yet implemented
                return Error.DynamicLoadingNotSupported;
            },
        }
    }

    /// Shutdown a feature, releasing resources.
    pub fn deinitFeature(self: *Registry, feature: Feature) Error!void {
        const reg = self.registrations.getPtr(feature) orelse return Error.FeatureNotRegistered;

        if (!reg.initialized) return;

        switch (reg.mode) {
            .comptime_only => {
                reg.initialized = false;
            },

            .runtime_toggle => {
                if (reg.deinit_fn) |deinit_fn| {
                    if (reg.context_ptr) |ptr| {
                        deinit_fn(ptr);
                    }
                }
                reg.context_ptr = null;
                reg.initialized = false;
            },

            .dynamic => {
                return Error.DynamicLoadingNotSupported;
            },
        }
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

    /// Enable a runtime-toggleable feature.
    pub fn enableFeature(self: *Registry, feature: Feature) Error!void {
        const reg = self.registrations.getPtr(feature) orelse return Error.FeatureNotRegistered;

        if (reg.mode == .comptime_only) {
            return; // Already enabled, nothing to do
        }

        reg.enabled = true;
        try self.runtime_overrides.put(self.allocator, feature, true);
    }

    /// Disable a runtime-toggleable feature. Deinitializes if currently initialized.
    pub fn disableFeature(self: *Registry, feature: Feature) Error!void {
        const reg = self.registrations.getPtr(feature) orelse return Error.FeatureNotRegistered;

        if (reg.mode == .comptime_only) {
            return Error.InvalidMode;
        }

        // Deinit if initialized
        if (reg.initialized) {
            try self.deinitFeature(feature);
        }

        reg.enabled = false;
        try self.runtime_overrides.put(self.allocator, feature, false);
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
// Compile-time Feature Checking
// ============================================================================

/// Check if a feature is compiled in via build_options.
pub fn isFeatureCompiledIn(comptime feature: Feature) bool {
    return switch (feature) {
        .gpu => build_options.enable_gpu,
        .ai => build_options.enable_ai,
        .llm => build_options.enable_llm,
        .embeddings => build_options.enable_ai,
        .agents => build_options.enable_ai,
        .training => build_options.enable_ai,
        .database => build_options.enable_database,
        .network => build_options.enable_network,
        .observability => build_options.enable_profiling,
        .web => build_options.enable_web,
    };
}

/// Get parent feature for sub-features.
pub fn getParentFeature(feature: Feature) ?Feature {
    return switch (feature) {
        .llm, .embeddings, .agents, .training => .ai,
        else => null,
    };
}

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
