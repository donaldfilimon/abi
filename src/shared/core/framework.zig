//! Core Framework - Main framework initialization and lifecycle management
//!
//! This module provides the main framework initialization, configuration,
//! and lifecycle management for the Abi AI Framework.

const std = @import("std");
const builtin = @import("builtin");

const config = @import("config.zig");
const errors = @import("errors.zig");
const lifecycle = @import("lifecycle.zig");

const FrameworkError = errors.FrameworkError;
const FrameworkConfig = config.FrameworkConfig;
const FrameworkState = lifecycle.FrameworkState;

/// Main framework instance
pub const Framework = struct {
    allocator: std.mem.Allocator,
    config: FrameworkConfig,
    state: FrameworkState,
    components: ComponentRegistry,
    metrics: *MetricsCollector,
    logger: *Logger,

    const Self = @This();

    /// Initialize the framework with the given configuration
    pub fn init(allocator: std.mem.Allocator, framework_config: FrameworkConfig) FrameworkError!*Self {
        // Validate configuration
        try framework_config.validate();

        // Create framework instance
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        self.* = .{
            .allocator = allocator,
            .config = framework_config,
            .state = .initializing,
            .components = ComponentRegistry.init(allocator),
            .metrics = try MetricsCollector.init(allocator),
            .logger = try Logger.init(allocator, framework_config.log_level),
        };

        // Initialize core components
        try self.initializeCoreComponents();

        // Initialize optional components
        if (framework_config.enable_gpu) {
            try self.initializeGPUComponents();
        }

        if (framework_config.enable_simd) {
            try self.initializeSIMDComponents();
        }

        if (framework_config.enable_memory_tracking) {
            try self.initializeMemoryTracking();
        }

        if (framework_config.enable_performance_profiling) {
            try self.initializePerformanceProfiling();
        }

        // Set state to ready
        self.state = .ready;

        self.logger.info("Framework initialized successfully", .{});

        return self;
    }

    /// Deinitialize the framework and clean up resources
    pub fn deinit(self: *Self) void {
        self.logger.info("Framework shutting down", .{});

        // Set state to shutting down
        self.state = .shutting_down;

        // Clean up components in reverse order
        self.components.deinit();

        // Clean up core components
        self.metrics.deinit();
        self.logger.deinit();

        // Set state to stopped
        self.state = .stopped;

        self.allocator.destroy(self);
    }

    /// Get the current framework state
    pub fn getState(self: *const Self) FrameworkState {
        return self.state;
    }

    /// Get framework configuration
    pub fn getConfig(self: *const Self) FrameworkConfig {
        return self.config;
    }

    /// Get component registry
    pub fn getComponents(self: *Self) *ComponentRegistry {
        return &self.components;
    }

    /// Get metrics collector
    pub fn getMetrics(self: *Self) *MetricsCollector {
        return self.metrics;
    }

    /// Get logger
    pub fn getLogger(self: *Self) *Logger {
        return self.logger;
    }

    /// Register a component with the framework
    pub fn registerComponent(self: *Self, name: []const u8, component: anytype) !void {
        try self.components.register(name, component);
        self.logger.debug("Component '{s}' registered", .{name});
    }

    /// Get a registered component
    pub fn getComponent(self: *Self, name: []const u8) ?anyopaque {
        return self.components.get(name);
    }

    /// Health check for the framework
    pub fn healthCheck(self: *const Self) HealthStatus {
        var status = HealthStatus{
            .overall = .healthy,
            .components = std.StringHashMap(ComponentHealth).init(self.allocator),
        };

        // Check core components
        if (self.state != .ready) {
            status.overall = .unhealthy;
            status.message = "Framework not in ready state";
            return status;
        }

        // Check component health
        var iterator = self.components.iterator();
        while (iterator.next()) |entry| {
            const component_health = self.checkComponentHealth(entry.key_ptr.*, entry.value_ptr.*);
            status.components.put(entry.key_ptr.*, component_health) catch continue;

            if (component_health.status != .healthy) {
                status.overall = .degraded;
            }
        }

        return status;
    }

    // Private methods

    fn initializeCoreComponents(self: *Self) !void {
        // Initialize error handling
        try self.registerComponent("error_handler", ErrorHandler.init(self.allocator));

        // Initialize configuration manager
        try self.registerComponent("config_manager", ConfigManager.init(self.allocator, self.config));

        // Initialize lifecycle manager
        try self.registerComponent("lifecycle_manager", LifecycleManager.init(self.allocator));

        self.logger.debug("Core components initialized", .{});
    }

    fn initializeGPUComponents(self: *Self) !void {
        // Initialize GPU backend manager
        const gpu_manager = try GPUBackendManager.init(self.allocator);
        try self.registerComponent("gpu_manager", gpu_manager);

        self.logger.debug("GPU components initialized", .{});
    }

    fn initializeSIMDComponents(self: *Self) !void {
        // Initialize SIMD operations
        const simd_ops = try SIMDOperations.init(self.allocator);
        try self.registerComponent("simd_operations", simd_ops);

        self.logger.debug("SIMD components initialized", .{});
    }

    fn initializeMemoryTracking(self: *Self) !void {
        // Initialize memory tracker
        const memory_tracker = try MemoryTracker.init(self.allocator);
        try self.registerComponent("memory_tracker", memory_tracker);

        self.logger.debug("Memory tracking initialized", .{});
    }

    fn initializePerformanceProfiling(self: *Self) !void {
        // Initialize performance profiler
        const profiler = try PerformanceProfiler.init(self.allocator);
        try self.registerComponent("performance_profiler", profiler);

        self.logger.debug("Performance profiling initialized", .{});
    }

    fn checkComponentHealth(self: *const Self, name: []const u8, component: anytype) ComponentHealth {
        _ = self;
        _ = name;
        _ = component;

        // Default health check - components can override this
        return ComponentHealth{
            .status = .healthy,
            .message = "Component is healthy",
            .last_check = std.time.microTimestamp(),
        };
    }
};

/// Component registry for managing framework components
pub const ComponentRegistry = struct {
    components: std.StringHashMap(*anyopaque),
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .components = std.StringHashMap(*anyopaque).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.components.deinit();
    }

    pub fn register(self: *Self, name: []const u8, component: anytype) !void {
        try self.components.put(name, component);
    }

    pub fn get(self: *Self, name: []const u8) ?*anyopaque {
        return self.components.get(name);
    }

    pub fn unregister(self: *Self, name: []const u8) bool {
        return self.components.remove(name);
    }

    pub fn list(self: *const Self) []const []const u8 {
        var names = std.ArrayList([]const u8).initCapacity(self.allocator, 0) catch return &[_][]const u8{};
        defer names.deinit(self.allocator);

        var iterator = self.components.iterator();
        while (iterator.next()) |entry| {
            names.append(self.allocator, entry.key_ptr.*) catch continue;
        }

        return names.toOwnedSlice(self.allocator) catch &[_][]const u8{};
    }
};

/// Health status for the framework
pub const HealthStatus = struct {
    overall: HealthLevel,
    message: ?[]const u8 = null,
    components: std.StringHashMap(ComponentHealth),
    timestamp: i64 = std.time.microTimestamp(),

    pub fn deinit(self: *HealthStatus) void {
        self.components.deinit();
    }
};

/// Component health status
pub const ComponentHealth = struct {
    status: HealthLevel,
    message: []const u8,
    last_check: i64,
    metrics: ?std.StringHashMap(f64) = null,
};

/// Health levels
pub const HealthLevel = enum {
    healthy,
    degraded,
    unhealthy,
};

/// Metrics collector for framework metrics
pub const MetricsCollector = struct {
    metrics: std.StringHashMap(Metric),
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) !*Self {
        const self = try allocator.create(Self);
        self.* = .{
            .metrics = std.StringHashMap(Metric).init(allocator),
            .allocator = allocator,
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.metrics.deinit();
        self.allocator.destroy(self);
    }

    pub fn recordMetric(self: *Self, name: []const u8, value: f64, tags: ?[]const []const u8) !void {
        const metric = Metric{
            .name = try self.allocator.dupe(u8, name),
            .value = value,
            .tags = if (tags) |t| try self.allocator.dupe([]const u8, t) else null,
            .timestamp = std.time.microTimestamp(),
        };

        try self.metrics.put(name, metric);
    }

    pub fn getMetric(self: *const Self, name: []const u8) ?Metric {
        return self.metrics.get(name);
    }

    pub fn getAllMetrics(self: *const Self) []const Metric {
        var metrics = std.ArrayList(Metric).initCapacity(self.allocator, 0) catch return &[_]Metric{};
        defer metrics.deinit(self.allocator);

        var iterator = self.metrics.iterator();
        while (iterator.next()) |entry| {
            metrics.append(self.allocator, entry.value_ptr.*) catch continue;
        }

        return metrics.toOwnedSlice(self.allocator) catch &[_]Metric{};
    }
};

/// Individual metric
pub const Metric = struct {
    name: []const u8,
    value: f64,
    tags: ?[]const []const u8,
    timestamp: i64,

    pub fn deinit(self: *Metric, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        if (self.tags) |tags| {
            allocator.free(tags);
        }
    }
};

/// Logger for framework logging
pub const Logger = struct {
    level: std.log.Level,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, level: std.log.Level) !*Self {
        const self = try allocator.create(Self);
        self.* = .{
            .level = level,
            .allocator = allocator,
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.destroy(self);
    }

    pub fn info(self: *const Self, comptime format: []const u8, args: anytype) void {
        if (@intFromEnum(self.level) <= @intFromEnum(std.log.Level.info)) {
            std.log.info(format, args);
        }
    }

    pub fn debug(self: *const Self, comptime format: []const u8, args: anytype) void {
        if (@intFromEnum(self.level) <= @intFromEnum(std.log.Level.debug)) {
            std.log.debug(format, args);
        }
    }

    pub fn warn(self: *const Self, comptime format: []const u8, args: anytype) void {
        if (@intFromEnum(self.level) <= @intFromEnum(std.log.Level.warn)) {
            std.log.warn(format, args);
        }
    }

    pub fn err(self: *const Self, comptime format: []const u8, args: anytype) void {
        if (@intFromEnum(self.level) <= @intFromEnum(std.log.Level.err)) {
            std.log.err(format, args);
        }
    }
};

// Placeholder types for components (to be implemented in separate modules)
const ErrorHandler = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) ErrorHandler {
        return .{ .allocator = allocator };
    }
};

const ConfigManager = struct {
    allocator: std.mem.Allocator,
    config: FrameworkConfig,

    pub fn init(allocator: std.mem.Allocator, _config: FrameworkConfig) ConfigManager {
        return .{ .allocator = allocator, .config = _config };
    }
};

const LifecycleManager = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) LifecycleManager {
        return .{ .allocator = allocator };
    }
};

const GPUBackendManager = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*GPUBackendManager {
        const self = try allocator.create(GPUBackendManager);
        self.* = .{ .allocator = allocator };
        return self;
    }
};

const SIMDOperations = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*SIMDOperations {
        const self = try allocator.create(SIMDOperations);
        self.* = .{ .allocator = allocator };
        return self;
    }
};

const MemoryTracker = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*MemoryTracker {
        const self = try allocator.create(MemoryTracker);
        self.* = .{ .allocator = allocator };
        return self;
    }
};

const PerformanceProfiler = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*PerformanceProfiler {
        const self = try allocator.create(PerformanceProfiler);
        self.* = .{ .allocator = allocator };
        return self;
    }
};

test "framework initialization" {
    const testing = std.testing;

    const framework_config = FrameworkConfig{
        .enable_gpu = false,
        .enable_simd = false,
        .enable_memory_tracking = false,
        .enable_performance_profiling = false,
        .log_level = .debug,
    };

    const framework = try Framework.init(testing.allocator, framework_config);
    defer framework.deinit();

    try testing.expectEqual(FrameworkState.ready, framework.getState());
    try testing.expectEqual(framework_config, framework.getConfig());
}

test "component registration" {
    const testing = std.testing;

    const framework_config = FrameworkConfig{
        .enable_gpu = false,
        .enable_simd = false,
        .enable_memory_tracking = false,
        .enable_performance_profiling = false,
        .log_level = .debug,
    };

    const framework = try Framework.init(testing.allocator, framework_config);
    defer framework.deinit();

    const test_component = ErrorHandler.init(testing.allocator);
    try framework.registerComponent("test_component", test_component);

    const retrieved = framework.getComponent("test_component");
    try testing.expect(retrieved != null);
}

test "health check" {
    const testing = std.testing;

    const framework_config = FrameworkConfig{
        .enable_gpu = false,
        .enable_simd = false,
        .enable_memory_tracking = false,
        .enable_performance_profiling = false,
        .log_level = .debug,
    };

    const framework = try Framework.init(testing.allocator, framework_config);
    defer framework.deinit();

    const health = framework.healthCheck();
    defer health.deinit();

    try testing.expectEqual(HealthLevel.healthy, health.overall);
}
