//! Framework Runtime - Modernized for Zig 0.16
//!
//! This module provides the core runtime system with proper initialization patterns
//! and memory management compatible with Zig 0.16-dev

const std = @import("std");
const collections = @import("../core/collections.zig");
const utils = @import("../shared/utils_modern.zig");

/// Framework runtime configuration
pub const RuntimeConfig = struct {
    max_plugins: u32 = 128,
    enable_hot_reload: bool = false,
    enable_profiling: bool = false,
    memory_limit_mb: ?u32 = null,
    log_level: LogLevel = .info,

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

pub const ComponentRegistry = struct {
    const Self = @This();

    components: collections.StringHashMap(Component),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .components = collections.StringHashMap(Component).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.components.deinit();
    }

    pub fn register(self: *Self, name: []const u8, component: Component) !void {
        try self.components.put(name, component);
    }

    pub fn get(self: *const Self, name: []const u8) ?Component {
        return self.components.get(name);
    }
};

pub const AtomicState = struct {
    const Self = @This();

    map: collections.StringHashMap(u64),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .map = collections.StringHashMap(u64).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.map.deinit();
    }

    pub fn set(self: *Self, key: []const u8, value: u64) !void {
        try self.map.put(key, value);
    }

    pub fn get(self: *const Self, key: []const u8) u64 {
        return self.map.get(key) orelse 0;
    }
};

pub const Stats = struct {
    const Self = @This();

    counters: collections.StringHashMap(u64),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .counters = collections.StringHashMap(u64).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.counters.deinit();
    }

    pub fn increment(self: *Self, key: []const u8) !void {
        const value = self.counters.get(key) orelse 0;
        try self.counters.put(key, value + 1);
    }

    pub fn get(self: *const Self, key: []const u8) u64 {
        return self.counters.get(key) orelse 0;
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

    pub fn init() RuntimeStats {
        return .{
            .start_time = std.time.milliTimestamp(),
            .total_components = 0,
            .active_components = 0,
            .memory_usage_bytes = 0,
            .update_count = 0,
            .last_update_duration_ns = 0,
        };
    }

    pub fn uptime(self: *const RuntimeStats) i64 {
        return std.time.milliTimestamp() - self.start_time;
    }
};

/// Main runtime system
pub const Runtime = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    config: RuntimeConfig,
    components: collections.ArrayList(Component),
    component_registry: collections.StringHashMap(Component), // name -> index
    stats: RuntimeStats,
    running: std.atomic.Value(bool),

    pub fn init(allocator: std.mem.Allocator, config: RuntimeConfig) !Self {
        return Self{
            .allocator = allocator,
            .config = config,
            .components = collections.ArrayList(Component){},
            .component_registry = collections.StringHashMap(Component).init(allocator),
            .stats = RuntimeStats.init(),
            .running = std.atomic.Value(bool).init(false),
        };
    }

    pub fn deinit(self: *Self) void {
        // Stop runtime if still running
        self.stop();

        // Deinitialize all components in reverse order
        var i = self.components.items.len;
        while (i > 0) {
            i -= 1;
            self.components.items[i].deinit() catch {};
        }

        self.components.deinit(self.allocator);
        self.component_registry.deinit();
    }

    pub fn registerComponent(self: *Self, component: Component) !void {
        if (self.component_registry.contains(component.name)) {
            return error.ComponentAlreadyRegistered;
        }

        try self.components.append(self.allocator, component);
        try self.component_registry.put(component.name, component);

        self.stats.total_components += 1;
    }

    pub fn initializeComponent(self: *Self, name: []const u8) !void {
        var component = self.component_registry.getPtr(name) orelse return error.ComponentNotFound;
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
            return error.AlreadyRunning;
        }

        self.running.store(true, .release);

        // Initialize all components first
        try self.initializeAllComponents();

        std.log.info("Runtime started with {} components", .{self.stats.total_components});
    }

    pub fn stop(self: *Self) void {
        if (!self.running.load(.acquire)) return;

        self.running.store(false, .release);
        std.log.info("Runtime stopped after {} updates", .{self.stats.update_count});
    }

    pub fn update(self: *Self, delta_time: f64) void {
        if (!self.running.load(.acquire)) return;

        const start_time = std.time.nanoTimestamp();

        for (self.components.items()) |component| {
            component.update(delta_time) catch {};
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

    /// Write runtime summary to a writer interface
    pub fn writeSummary(self: *const Self, writer: anytype) !void {
        try writer.print("ABI Runtime Summary\n");
        try writer.print("==================\n");
        try writer.print("Status: {s}\n", .{if (self.isRunning()) "Running" else "Stopped"});
        try writer.print("Components: {}/{}\n", .{ self.stats.active_components, self.stats.total_components });
        try writer.print("Uptime: {}ms\n", .{self.stats.uptime()});
        try writer.print("Updates: {}\n", .{self.stats.update_count});
        if (self.stats.update_count > 0) {
            try writer.print("Last Update: {}ns\n", .{self.stats.last_update_duration_ns});
        }
    }
};

/// Factory function for creating runtime instances
pub fn createRuntime(allocator: std.mem.Allocator, config: RuntimeConfig) !Runtime {
    return try Runtime.init(allocator, config);
}

/// Default runtime configuration
pub fn defaultConfig() RuntimeConfig {
    return RuntimeConfig{};
}

test "framework runtime - basic operations" {
    const testing = std.testing;

    var runtime = try createRuntime(testing.allocator, defaultConfig());
    defer runtime.deinit();

    // Test component registration
    const test_component = Component{
        .name = "test",
        .version = "1.0.0",
    };

    try runtime.registerComponent(test_component);
    try testing.expectEqual(@as(u32, 1), runtime.stats.total_components);

    // Test runtime start/stop
    try runtime.start();
    try testing.expect(runtime.isRunning());

    runtime.stop();
    try testing.expect(!runtime.isRunning());
}

test "framework runtime - component lifecycle" {
    const testing = std.testing;

    var runtime = try createRuntime(testing.allocator, defaultConfig());
    defer runtime.deinit();

    const TestState = struct {
        var init_called: bool = false;
        var deinit_called: bool = false;
    };

    TestState.init_called = false;
    TestState.deinit_called = false;

    const TestComponent = struct {
        fn testInit(allocator: std.mem.Allocator, config: *const RuntimeConfig) !void {
            _ = allocator;
            _ = config;
            TestState.init_called = true;
        }

        fn testDeinit() !void {
            TestState.deinit_called = true;
        }
    };

    const component = Component{
        .name = "lifecycle_test",
        .version = "1.0.0",
        .init_fn = TestComponent.testInit,
        .deinit_fn = TestComponent.testDeinit,
    };

    try runtime.registerComponent(component);
    try runtime.initializeComponent("lifecycle_test");

    try testing.expect(TestState.init_called);

    runtime.deinit();
    try testing.expect(TestState.deinit_called);
}

test "component registry - basic operations" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var registry = ComponentRegistry.init(allocator);
    defer registry.deinit();

    const dummy_component = Component{ .name = "dummy", .version = "1.0" };
    try registry.register("dummy", dummy_component);

    const retrieved = registry.get("dummy");
    try std.testing.expect(retrieved != null);
    try std.testing.expect(std.mem.eql(u8, "dummy", retrieved.?.name));
}

test "atomic state" {
    const allocator = std.testing.allocator;
    var state = AtomicState.init(allocator);
    defer state.deinit();

    try state.set("key1", 123);
    try std.testing.expect(state.get("key1") == 123);
}

test "stats" {
    const allocator = std.testing.allocator;
    var stats = Stats.init(allocator);
    defer stats.deinit();

    try stats.increment("counter1");
    try stats.increment("counter1");
    try std.testing.expect(stats.get("counter1") == 2);
}
