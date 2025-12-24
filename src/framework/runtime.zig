//! Framework Runtime - Unified Implementation
//!
//! This module provides the core runtime system with proper initialization patterns
//! and memory management compatible with Zig 0.16-dev

const std = @import("std");
const core = @import("../core/mod.zig");
const features = @import("../features/mod.zig");

const feature_tag_count = features.feature_count;

/// Framework runtime configuration
pub const RuntimeConfig = struct {
    max_plugins: u32 = 128,
    enable_hot_reload: bool = false,
    enable_profiling: bool = false,
    memory_limit_mb: ?u32 = null,
    log_level: LogLevel = .info,
    enabled_features: []const features.FeatureTag = &[_]features.FeatureTag{ .ai, .database, .web, .monitoring, .simd },
    disabled_features: []const features.FeatureTag = &[_]features.FeatureTag{},
    plugin_paths: []const []const u8 = &[_][]const u8{},
    auto_discover_plugins: bool = false,
    auto_register_plugins: bool = false,
    auto_start_plugins: bool = false,
    feature_storage: FeatureStorage = .{},

    pub const LogLevel = enum {
        debug,
        info,
        warn,
        err,
    };

    pub const FeatureStorage = struct {
        enabled: [feature_tag_count]features.FeatureTag = undefined,
        disabled: [feature_tag_count]features.FeatureTag = undefined,
        enabled_len: usize = 0,
        disabled_len: usize = 0,

        pub fn setEnabled(self: *FeatureStorage, feature_list: []const features.FeatureTag) void {
            std.debug.assert(feature_list.len <= feature_tag_count);
            std.mem.copyForwards(features.FeatureTag, self.enabled[0..feature_list.len], feature_list);
            self.enabled_len = feature_list.len;
        }

        pub fn setDisabled(self: *FeatureStorage, feature_list: []const features.FeatureTag) void {
            std.debug.assert(feature_list.len <= feature_tag_count);
            std.mem.copyForwards(features.FeatureTag, self.disabled[0..feature_list.len], feature_list);
            self.disabled_len = feature_list.len;
        }

        pub fn enabledSlice(self: *const FeatureStorage) []const features.FeatureTag {
            return self.enabled[0..self.enabled_len];
        }

        pub fn disabledSlice(self: *const FeatureStorage) []const features.FeatureTag {
            return self.disabled[0..self.disabled_len];
        }
    };

    pub fn rebaseFeatureSlices(self: *RuntimeConfig) void {
        if (self.enabled_features.ptr == self.feature_storage.enabled[0..].ptr) {
            self.enabled_features = self.feature_storage.enabledSlice();
        }

        if (self.disabled_features.ptr == self.feature_storage.disabled[0..].ptr) {
            self.disabled_features = self.feature_storage.disabledSlice();
        }
    }
};

/// Creates a deep copy of RuntimeConfig with owned slices
///
/// This function ensures that the feature slices and plugin path list in the
/// returned config are owned by the allocator and can be safely freed. This
/// prevents use-after-free errors when the original config points to const
/// literals.
///
/// # Parameters
/// - allocator: The allocator to use for copying slices
/// - config: The source configuration to copy
///
/// # Returns
/// A RuntimeConfig with owned slices that must be freed by the caller
///
/// # Errors
/// - OutOfMemory: If allocation fails for copying slices
fn normalizeConfig(allocator: std.mem.Allocator, config: RuntimeConfig) !RuntimeConfig {
    const enabled_features = try allocator.dupe(features.FeatureTag, config.enabled_features);
    errdefer allocator.free(enabled_features);

    const disabled_features = try allocator.dupe(features.FeatureTag, config.disabled_features);
    errdefer allocator.free(disabled_features);

    const plugin_paths = try allocator.alloc([]const u8, config.plugin_paths.len);
    errdefer allocator.free(plugin_paths);
    std.mem.copyForwards([]const u8, plugin_paths, config.plugin_paths);

    return RuntimeConfig{
        .max_plugins = config.max_plugins,
        .enable_hot_reload = config.enable_hot_reload,
        .enable_profiling = config.enable_profiling,
        .memory_limit_mb = config.memory_limit_mb,
        .log_level = config.log_level,
        .enabled_features = enabled_features,
        .disabled_features = disabled_features,
        .plugin_paths = plugin_paths,
        .auto_discover_plugins = config.auto_discover_plugins,
        .auto_register_plugins = config.auto_register_plugins,
        .auto_start_plugins = config.auto_start_plugins,
    };
}

/// Component interface for the runtime system
pub const Component = struct {
    name: []const u8,
    version: []const u8,
    init_fn: ?*const fn (std.mem.Allocator, *const RuntimeConfig) anyerror!void = null,
    deinit_fn: ?*const fn () anyerror!void = null,
    update_fn: ?*const fn (f64) anyerror!void = null,

    pub fn init(self: *const Component, allocator: std.mem.Allocator, config: *const RuntimeConfig) !void {
        if (self.init_fn) |init_func| {
            try init_func(allocator, config);
        }
    }

    pub fn deinit(self: *const Component) !void {
        if (self.deinit_fn) |deinit_func| {
            try deinit_func();
        }
    }

    pub fn update(self: *const Component, delta_time: f64) !void {
        if (self.update_fn) |update_func| {
            try update_func(delta_time);
        }
    }
};

/// Runtime statistics
pub const RuntimeStats = struct {
    start_time: i64,
    total_components: u32,
    active_components: u32,
    memory_usage_bytes: usize,
    update_count: u64,
    last_update_duration_ns: u64,
    enabled_features: usize,
    error_count: u64 = 0,

    pub fn init(enabled_features: usize) RuntimeStats {
        return .{
            .start_time = 0,
            .total_components = 0,
            .active_components = 0,
            .memory_usage_bytes = 0,
            .update_count = 0,
            .last_update_duration_ns = 0,
            .enabled_features = enabled_features,
            .error_count = 0,
        };
    }

    pub fn uptime(self: *const RuntimeStats) i64 {
        if (self.start_time == 0) return 0;

        const now_ms = @divFloor(std.time.nanoTimestamp(), std.time.ns_per_ms);
        return now_ms - self.start_time;
    }
};

/// Main framework runtime system
pub const Framework = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    config: RuntimeConfig,
    components: core.ArrayList(Component),
    component_registry: core.StringHashMap(Component),
    stats: RuntimeStats,
    running: std.atomic.Value(bool),
    enabled_features: features.config.FeatureFlags,

    /// Initializes a new Framework instance with proper memory management
    ///
    /// This function creates a Framework with owned configuration slices and
    /// initialized component management systems. The input config is normalized
    /// to prevent use-after-free errors when dealing with const literal slices.
    ///
    /// # Parameters
    /// - allocator: Memory allocator for the framework and all its components
    /// - config: Runtime configuration for the framework
    ///
    /// # Returns
    /// An initialized Framework instance ready for component registration
    ///
    /// # Errors
    /// - OutOfMemory: If memory allocation fails for configuration or components
    ///
    /// # Safety
    /// The caller must call deinit() on the returned Framework to free all resources
    pub fn init(allocator: std.mem.Allocator, config: RuntimeConfig) !Self {
        const normalized_config = try normalizeConfig(allocator, config);
        errdefer {
            allocator.free(normalized_config.enabled_features);
            allocator.free(normalized_config.disabled_features);
            allocator.free(normalized_config.plugin_paths);
        }

        // Calculate enabled features
        var enabled_features = features.config.createFlags(normalized_config.enabled_features);

        // Remove disabled features
        for (normalized_config.disabled_features) |feature| {
            enabled_features.unset(features.config.tagIndex(feature));
        }

        // Initialize enabled features
        var enabled_feature_list = std.ArrayList(features.FeatureTag).initCapacity(allocator, enabled_features.count()) catch unreachable;
        defer enabled_feature_list.deinit(allocator);

        var iter = enabled_features.iterator(.{});
        while (iter.next()) |index| {
            const feature = @as(features.FeatureTag, @enumFromInt(index));
            enabled_feature_list.appendAssumeCapacity(feature);
        }

        // Initialize all enabled features
        features.lifecycle.initFeatures(allocator, enabled_feature_list.items) catch |err| {
            std.log.err("Failed to initialize features: {}", .{err});
            return err;
        };

        return Self{
            .allocator = allocator,
            .config = normalized_config,
            .components = core.ArrayList(Component).initCapacity(allocator, 0) catch unreachable,
            .component_registry = core.StringHashMap(Component).init(allocator),
            .stats = RuntimeStats.init(enabled_features.count()),
            .running = std.atomic.Value(bool).init(false),
            .enabled_features = enabled_features,
        };
    }

    pub fn deinit(self: *Self) void {
        // Stop runtime if still running
        self.stop();

        // Deinitialize all components in reverse order
        var i = self.components.items.len;
        while (i > 0) {
            i -= 1;
            const component = &self.components.items[i];
            component.deinit() catch |err| {
                std.log.err("Failed to deinitialize component '{s}': {}", .{ component.name, err });
            };
        }

        // Deinitialize all enabled features
        var enabled_feature_list = std.ArrayList(features.FeatureTag).initCapacity(self.allocator, self.enabled_features.count()) catch unreachable;
        defer enabled_feature_list.deinit(self.allocator);

        var iter = self.enabled_features.iterator(.{});
        while (iter.next()) |index| {
            const feature = @as(features.FeatureTag, @enumFromInt(index));
            enabled_feature_list.appendAssumeCapacity(feature);
        }

        features.lifecycle.deinitFeatures(enabled_feature_list.items);

        self.components.deinit(self.allocator);
        self.component_registry.deinit();
        self.allocator.free(self.config.enabled_features);
        self.allocator.free(self.config.disabled_features);
        self.allocator.free(self.config.plugin_paths);
    }

    pub fn registerComponent(self: *Self, component: Component) !void {
        // Validate component name
        if (component.name.len == 0) return error.InvalidComponentName;
        if (std.mem.indexOf(u8, component.name, &[_]u8{0}) != null) return error.InvalidComponentName;

        // Validate version
        if (component.version.len == 0) return error.InvalidVersion;

        if (self.component_registry.contains(component.name)) {
            return core.Error.AlreadyExists;
        }

        try self.components.append(self.allocator, component);
        try self.component_registry.put(component.name, component);

        self.stats.total_components += 1;
    }

    pub fn initializeComponent(self: *Self, name: []const u8) !void {
        var component = self.component_registry.getPtr(name) orelse return core.Error.NotFound;
        component.init(self.allocator, &self.config) catch |err| {
            std.log.err("Failed to initialize component '{s}': {}", .{ name, err });
            return err;
        };
        self.stats.active_components += 1;
    }

    pub fn initializeAllComponents(self: *Self) !void {
        for (self.components.items) |*component| {
            component.init(self.allocator, &self.config) catch |err| {
                std.log.err("Failed to initialize component '{s}': {}", .{ component.name, err });
                return err;
            };
        }
        self.stats.active_components = @intCast(self.components.items.len);
    }

    pub fn start(self: *Self) !void {
        if (self.running.load(.acquire)) {
            return core.Error.AlreadyExists;
        }

        errdefer self.running.store(false, .release);
        self.running.store(true, .release);

        // Initialize all components first
        try self.initializeAllComponents();

        self.stats.start_time = @divFloor(std.time.nanoTimestamp(), std.time.ns_per_ms);
        std.log.info("Framework started with {} components and {} features", .{ self.stats.total_components, self.stats.enabled_features });
    }

    pub fn stop(self: *Self) void {
        if (!self.running.load(.acquire)) return;

        self.running.store(false, .release);
        std.log.info("Framework stopped after {} updates", .{self.stats.update_count});
    }

    pub fn update(self: *Self, delta_time: f64) void {
        if (!self.running.load(.acquire)) return;

        const start_time = std.time.nanoTimestamp();

        for (self.components.items) |component| {
            component.update(delta_time) catch |err| {
                std.log.err("Component '{s}' update failed: {}", .{ component.name, err });
                self.stats.error_count += 1;
                // Continue with other components even if one fails
            };
        }

        const end_time = std.time.nanoTimestamp();
        self.stats.last_update_duration_ns = @intCast(end_time - start_time);
        self.stats.update_count += 1;
    }

    pub fn isRunning(self: *const Self) bool {
        return self.running.load(.acquire);
    }

    pub fn getStats(self: *const Self) RuntimeStats {
        return self.stats;
    }

    pub fn getComponent(self: *Self, name: []const u8) ?*Component {
        return self.component_registry.getPtr(name);
    }

    pub fn isFeatureEnabled(self: *const Self, feature: features.FeatureTag) bool {
        return self.enabled_features.isSet(features.config.tagIndex(feature));
    }

    pub fn enableFeature(self: *Self, feature: features.FeatureTag) void {
        self.enabled_features.set(features.config.tagIndex(feature));
    }

    pub fn disableFeature(self: *Self, feature: features.FeatureTag) void {
        self.enabled_features.unset(features.config.tagIndex(feature));
    }

    /// Write framework summary to a writer interface
    pub fn writeSummary(self: *const Self, writer: anytype) !void {
        try writer.print("ABI Framework Summary\n");
        try writer.print("=====================\n");
        try writer.print("Status: {s}\n", .{if (self.isRunning()) "Running" else "Stopped"});
        try writer.print("Components: {}/{}\n", .{ self.stats.active_components, self.stats.total_components });
        const enabled_feature_count = self.enabled_features.count();
        if (self.stats.enabled_features != enabled_feature_count) {
            // Keep stats in sync with runtime feature toggles for accurate summaries.
            @constCast(&self.stats).enabled_features = enabled_feature_count;
        }
        try writer.print("Features: {}\n", .{self.stats.enabled_features});
        try writer.print("Uptime: {}ms\n", .{self.stats.uptime()});
        try writer.print("Updates: {}\n", .{self.stats.update_count});

        if (self.stats.update_count > 0) {
            try writer.print("Last Update: {}ns\n", .{self.stats.last_update_duration_ns});
        }

        // List enabled features
        try writer.print("Enabled Features:\n");
        const feature_tags = features.config.allTags();
        for (feature_tags, 0..) |feature, idx| {
            if (self.enabled_features.isSet(idx)) {
                try writer.print("  - {s}: {s}\n", .{ features.config.getName(feature), features.config.getDescription(feature) });
            }
        }
    }
};

/// Factory function for creating framework instances
pub fn createFramework(allocator: std.mem.Allocator, config: RuntimeConfig) !Framework {
    return try Framework.init(allocator, config);
}

/// Default framework configuration
pub fn defaultConfig() RuntimeConfig {
    return RuntimeConfig{};
}

test "framework runtime - basic operations" {
    const testing = std.testing;

    var framework = try createFramework(testing.allocator, defaultConfig());
    defer framework.deinit();

    // Test component registration
    const test_component = Component{
        .name = "test",
        .version = "1.0.0",
    };

    try framework.registerComponent(test_component);
    try testing.expectEqual(@as(u32, 1), framework.stats.total_components);

    // Test feature management
    try testing.expect(framework.isFeatureEnabled(.ai));
    try testing.expect(framework.isFeatureEnabled(.simd));
    try testing.expect(!framework.isFeatureEnabled(.gpu));

    framework.enableFeature(.gpu);
    try testing.expect(framework.isFeatureEnabled(.gpu));
    framework.disableFeature(.simd);
    try testing.expect(!framework.isFeatureEnabled(.simd));
    framework.enableFeature(.simd);
    try testing.expect(framework.isFeatureEnabled(.simd));

    // Test runtime start/stop
    try framework.start();
    try testing.expect(framework.isRunning());

    framework.stop();
    try testing.expect(!framework.isRunning());
}

test "framework - feature configuration" {
    const testing = std.testing;

    const config = RuntimeConfig{
        .enabled_features = &[_]features.FeatureTag{ .ai, .gpu, .simd },
        .disabled_features = &[_]features.FeatureTag{.gpu},
    };

    var framework = try createFramework(testing.allocator, config);
    defer framework.deinit();

    try testing.expect(framework.isFeatureEnabled(.ai));
    try testing.expect(!framework.isFeatureEnabled(.gpu)); // Disabled overrides enabled
    try testing.expect(framework.isFeatureEnabled(.simd));
    try testing.expect(!framework.isFeatureEnabled(.database));
    try testing.expect(!framework.isFeatureEnabled(.simd));
}

test "runtime feature bitset stays aligned with feature flags mapping" {
    const testing = std.testing;
    const all_features = [_]features.FeatureTag{
        .ai,
        .gpu,
        .database,
        .web,
        .monitoring,
        .connectors,
    };

    var framework = try createFramework(
        testing.allocator,
        .{ .enabled_features = &all_features },
    );
    defer framework.deinit();

    const expected_flags = features.config.createFlags(&all_features);
    const bit_length = expected_flags.bit_length;
    var idx: usize = 0;
    while (idx < bit_length) : (idx += 1) {
        try testing.expectEqual(
            expected_flags.isSet(idx),
            framework.enabled_features.isSet(idx),
        );
    }

    framework.disableFeature(.web);
    try testing.expect(!framework.isFeatureEnabled(.web));
    framework.enableFeature(.web);
    try testing.expect(framework.isFeatureEnabled(.web));
}
