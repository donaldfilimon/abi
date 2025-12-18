//! Framework Runtime - Unified Implementation
//!
//! This module provides the core runtime system with proper initialization patterns
//! and memory management compatible with Zig 0.16-dev

const std = @import("std");
const core = @import("../core/mod.zig");
const features = @import("../features/mod.zig");

/// Framework runtime configuration
pub const RuntimeConfig = struct {
    max_plugins: u32 = 128,
    enable_hot_reload: bool = false,
    enable_profiling: bool = false,
    memory_limit_mb: ?u32 = null,
    log_level: LogLevel = .info,
    enabled_features: []const features.FeatureTag = &[_]features.FeatureTag{ .ai, .database, .web, .monitoring, .simd },
    disabled_features: []const features.FeatureTag = &[_]features.FeatureTag{},

    pub const LogLevel = enum {
        debug,
        info,
        warn,
        err,
    };
};

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

    pub fn init(enabled_features: usize) RuntimeStats {
        return .{
            .start_time = 0,
            .total_components = 0,
            .active_components = 0,
            .memory_usage_bytes = 0,
            .update_count = 0,
            .last_update_duration_ns = 0,
            .enabled_features = enabled_features,
        };
    }

    pub fn uptime(self: *const RuntimeStats) i64 {
        return 0 - self.start_time;
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

    pub fn init(allocator: std.mem.Allocator, config: RuntimeConfig) !Self {
        // Calculate enabled features
        var enabled_features = features.config.FeatureFlags.initEmpty();
        for (config.enabled_features) |feature| {
            enabled_features.set(features.config.tagIndex(feature));
        }

        // Remove disabled features
        for (config.disabled_features) |feature| {
            enabled_features.unset(features.config.tagIndex(feature));
        }

        return Self{
            .allocator = allocator,
            .config = config,
            .components = core.ArrayList(Component).init(allocator),
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
            component.deinit() catch {};
        }

        self.components.deinit();
        self.component_registry.deinit();
    }

    pub fn registerComponent(self: *Self, component: Component) !void {
        if (self.component_registry.contains(component.name)) {
            return core.Error.AlreadyExists;
        }

        try self.components.append(component);
        try self.component_registry.put(component.name, component);

        self.stats.total_components += 1;
    }

    pub fn initializeComponent(self: *Self, name: []const u8) !void {
        var component = self.component_registry.getPtr(name) orelse return core.Error.NotFound;
        try component.init(self.allocator, &self.config);
        self.stats.active_components += 1;
    }

    pub fn initializeAllComponents(self: *Self) !void {
        for (self.components.items) |*component| {
            try component.init(self.allocator, &self.config);
        }
        self.stats.active_components = @intCast(self.components.items.len);
    }

    pub fn start(self: *Self) !void {
        if (self.running.load(.acquire)) {
            return core.Error.AlreadyExists;
        }

        self.running.store(true, .release);

        // Initialize all components first
        try self.initializeAllComponents();

        std.log.info("Framework started with {} components and {} features", .{ self.stats.total_components, self.stats.enabled_features });
    }

    pub fn stop(self: *Self) void {
        if (!self.running.load(.acquire)) return;

        self.running.store(false, .release);
        std.log.info("Framework stopped after {} updates", .{self.stats.update_count});
    }

    pub fn update(self: *Self, delta_time: f64) void {
        if (!self.running.load(.acquire)) return;

        const start_time = std.time.nanoTimestamp;

        for (self.components.items) |component| {
            component.update(delta_time) catch {};
        }

        const end_time = std.time.nanoTimestamp;
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

    // Test runtime start/stop
    try framework.start();
    try testing.expect(framework.isRunning());

    framework.stop();
    try testing.expect(!framework.isRunning());
}

test "framework - feature configuration" {
    const testing = std.testing;

    const config = RuntimeConfig{
        .enabled_features = &[_]features.FeatureTag{ .ai, .gpu },
        .disabled_features = &[_]features.FeatureTag{.gpu},
    };

    var framework = try createFramework(testing.allocator, config);
    defer framework.deinit();

    try testing.expect(framework.isFeatureEnabled(.ai));
    try testing.expect(!framework.isFeatureEnabled(.gpu)); // Disabled overrides enabled
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
