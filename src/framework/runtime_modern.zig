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
    init_fn: ?*const fn (allocator: std.mem.Allocator, config: *const RuntimeConfig) anyerror!void = null,
    deinit_fn: ?*const fn () void = null,
    update_fn: ?*const fn (delta_time: f64) void = null,

    pub fn init(self: *const Component, allocator: std.mem.Allocator, config: *const RuntimeConfig) !void {
        if (self.init_fn) |func| {
            try func(allocator, config);
        }
    }

    pub fn deinit(self: *const Component) void {
        if (self.deinit_fn) |func| {
            func();
        }
    }

    pub fn update(self: *const Component, delta_time: f64) void {
        if (self.update_fn) |func| {
            func(delta_time);
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
    component_registry: collections.StringHashMap(u32), // name -> index
    stats: RuntimeStats,
    running: std.atomic.Value(bool),

    pub fn init(allocator: std.mem.Allocator, config: RuntimeConfig) !Self {
        return Self{
            .allocator = allocator,
            .config = config,
            .components = collections.ArrayList(Component).init(allocator),
            .component_registry = collections.StringHashMap(u32).init(allocator),
            .stats = RuntimeStats.init(),
            .running = std.atomic.Value(bool).init(false),
        };
    }

    pub fn deinit(self: *Self) void {
        // Stop runtime if still running
        self.stop();

        // Deinitialize all components in reverse order
        var i = self.components.len();
        while (i > 0) {
            i -= 1;
            self.components.items()[i].deinit();
        }

        self.components.deinit();
        self.component_registry.deinit();
    }

    pub fn registerComponent(self: *Self, component: Component) !void {
        if (self.component_registry.contains(component.name)) {
            return error.ComponentAlreadyRegistered;
        }

        const index = @as(u32, @intCast(self.components.len()));
        try self.components.append(self.allocator, component);
        try self.component_registry.put(self.allocator, component.name, index);

        self.stats.total_components += 1;
    }

    pub fn initializeComponent(self: *Self, name: []const u8) !void {
        const index = self.component_registry.get(name) orelse return error.ComponentNotFound;
        const component = &self.components.itemsMut()[index];
        try component.init(self.allocator, &self.config);
        self.stats.active_components += 1;
    }

    pub fn initializeAllComponents(self: *Self) !void {
        for (self.components.itemsMut()) |*component| {
            try component.init(self.allocator, &self.config);
        }
        self.stats.active_components = self.stats.total_components;
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

        for (self.components.items()) |*component| {
            component.update(delta_time);
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
        const index = self.component_registry.get(name) orelse return null;
        return &self.components.itemsMut()[index];
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

        fn testDeinit() void {
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

    // Note: deinit_called will be true after runtime.deinit() is called
}
