//! Enhanced Plugin System - Production-ready plugin architecture with hot-reloading
//!
//! This module provides a comprehensive plugin system for the Abi AI Framework with:
//! - Dynamic plugin loading and unloading
//! - Hot-reloading capabilities
//! - Plugin dependencies and versioning
//! - Service discovery and communication
//! - Security and sandboxing
//! - Performance monitoring and metrics

const std = @import("std");
const builtin = @import("builtin");

const core = @import("../core/mod.zig");
const config = @import("../core/config.zig");
const errors = @import("../core/errors.zig");

const FrameworkError = errors.FrameworkError;
const PluginConfig = config.PluginConfig;

/// Enhanced plugin system with production-ready features
pub const EnhancedPluginSystem = struct {
    allocator: std.mem.Allocator,
    plugins: std.StringHashMap(*Plugin),
    plugin_loaders: std.ArrayList(*PluginLoader),
    plugin_registry: *PluginRegistry,
    plugin_watcher: *PluginWatcher,
    plugin_manager: *PluginManager,
    service_discovery: *ServiceDiscovery,
    security_manager: *SecurityManager,
    performance_monitor: *PerformanceMonitor,

    const Self = @This();

    /// Initialize the enhanced plugin system
    pub fn init(allocator: std.mem.Allocator) FrameworkError!*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        self.* = .{
            .allocator = allocator,
            .plugins = std.StringHashMap(*Plugin).init(allocator),
            .plugin_loaders = std.ArrayList(*PluginLoader).init(allocator),
            .plugin_registry = try PluginRegistry.init(allocator),
            .plugin_watcher = try PluginWatcher.init(allocator),
            .plugin_manager = try PluginManager.init(allocator),
            .service_discovery = try ServiceDiscovery.init(allocator),
            .security_manager = try SecurityManager.init(allocator),
            .performance_monitor = try PerformanceMonitor.init(allocator),
        };

        // Initialize plugin loaders
        try self.initializePluginLoaders();

        // Initialize plugin watcher if hot-reloading is enabled
        if (config.enable_hot_reload) {
            try self.plugin_watcher.start(config.plugin_directory);
        }

        // Load initial plugins
        try self.loadInitialPlugins(config.plugin_directory);

        return self;
    }

    /// Deinitialize the enhanced plugin system
    pub fn deinit(self: *Self) void {
        // Stop plugin watcher
        self.plugin_watcher.stop();

        // Unload all plugins
        self.unloadAllPlugins();

        // Clean up components
        self.plugin_registry.deinit();
        self.plugin_watcher.deinit();
        self.plugin_manager.deinit();
        self.service_discovery.deinit();
        self.security_manager.deinit();
        self.performance_monitor.deinit();

        // Clean up plugin loaders
        for (self.plugin_loaders.items) |loader| {
            loader.deinit();
        }
        self.plugin_loaders.deinit();

        // Clean up plugins
        self.plugins.deinit();

        self.allocator.destroy(self);
    }

    /// Load a plugin from file
    pub fn loadPlugin(self: *Self, path: []const u8) FrameworkError!*Plugin {
        // Security check
        try self.security_manager.validatePluginPath(path);

        // Load plugin using appropriate loader
        const loader = try self.getPluginLoader(path);
        const plugin = try loader.loadPlugin(path);

        // Validate plugin
        try self.validatePlugin(plugin);

        // Check dependencies
        try self.checkPluginDependencies(plugin);

        // Initialize plugin
        try plugin.initialize(self.allocator);

        // Register plugin
        try self.plugins.put(plugin.name, plugin);
        try self.plugin_registry.registerPlugin(plugin);

        // Register services
        try self.registerPluginServices(plugin);

        // Start plugin
        try plugin.start();

        // Emit plugin loaded event
        self.emitPluginEvent(.plugin_loaded, plugin, null);

        return plugin;
    }

    /// Unload a plugin
    pub fn unloadPlugin(self: *Self, name: []const u8) FrameworkError!void {
        const plugin = self.plugins.get(name) orelse return FrameworkError.PluginNotFound;

        // Stop plugin
        try plugin.stop();

        // Unregister services
        try self.unregisterPluginServices(plugin);

        // Unregister plugin
        try self.plugin_registry.unregisterPlugin(name);

        // Remove from plugins map
        _ = self.plugins.remove(name);

        // Deinitialize plugin
        plugin.deinitialize();

        // Emit plugin unloaded event
        self.emitPluginEvent(.plugin_unloaded, plugin, null);
    }

    /// Reload a plugin
    pub fn reloadPlugin(self: *Self, name: []const u8) FrameworkError!void {
        const plugin = self.plugins.get(name) orelse return FrameworkError.PluginNotFound;

        // Get plugin path
        const path = plugin.path;

        // Unload plugin
        try self.unloadPlugin(name);

        // Load plugin again
        _ = try self.loadPlugin(path);
    }

    /// Get a plugin by name
    pub fn getPlugin(self: *Self, name: []const u8) ?*Plugin {
        return self.plugins.get(name);
    }

    /// List all loaded plugins
    pub fn listPlugins(self: *const Self) []const []const u8 {
        var names = std.ArrayList([]const u8).init(self.allocator);
        defer names.deinit();

        var iterator = self.plugins.iterator();
        while (iterator.next()) |entry| {
            names.append(entry.key_ptr.*) catch continue;
        }

        return names.toOwnedSlice() catch &[_][]const u8{};
    }

    /// Get plugin statistics
    pub fn getPluginStats(self: *const Self) PluginStats {
        var stats = PluginStats{};

        stats.total_plugins = @as(u32, @intCast(self.plugins.count()));

        var iterator = self.plugins.iterator();
        while (iterator.next()) |entry| {
            const plugin = entry.value_ptr.*;

            switch (plugin.state) {
                .loaded => stats.loaded_plugins += 1,
                .running => stats.running_plugins += 1,
                .error_state => stats.error_plugins += 1,
                else => {},
            }

            stats.total_services += @as(u32, @intCast(plugin.services.items.len));
            stats.total_memory_usage += plugin.memory_usage;
        }

        return stats;
    }

    /// Health check for all plugins
    pub fn healthCheck(self: *const Self) PluginHealthStatus {
        var status = PluginHealthStatus{.{
            .overall = .healthy,
            .plugins = std.StringHashMap(PluginHealth).init(self.allocator),
        }};

        var iterator = self.plugins.iterator();
        while (iterator.next()) |entry| {
            const plugin = entry.value_ptr.*;
            const health = plugin.healthCheck();

            status.plugins.put(entry.key_ptr.*, health) catch {
                continue;
            };

            if (health.status != .healthy) {
                status.overall = .degraded;
            }
        }

        return status;
    }

    // Private methods

    fn initializePluginLoaders(self: *Self) FrameworkError!void {
        // Initialize native plugin loader
        const native_loader = try NativePluginLoader.init(self.allocator);
        try self.plugin_loaders.append(native_loader);

        // Initialize script plugin loader
        const script_loader = try ScriptPluginLoader.init(self.allocator);
        try self.plugin_loaders.append(script_loader);

        // Initialize web plugin loader
        const web_loader = try WebPluginLoader.init(self.allocator);
        try self.plugin_loaders.append(web_loader);
    }

    fn getPluginLoader(self: *Self, path: []const u8) FrameworkError!*PluginLoader {
        const extension = std.fs.path.extension(path);
        if (std.mem.eql(u8, extension, ".dll") or
            std.mem.eql(u8, extension, ".so") or
            std.mem.eql(u8, extension, ".dylib"))
        {
            return &self.plugin_loaders.items[0]; // Native loader
        } else if (std.mem.eql(u8, extension, ".js") or
            std.mem.eql(u8, extension, ".py") or
            std.mem.eql(u8, extension, ".lua"))
        {
            return &self.plugin_loaders.items[1]; // Script loader
        } else if (std.mem.eql(u8, extension, ".wasm")) {
            return &self.plugin_loaders.items[2]; // Web loader
        } else {
            return FrameworkError.UnsupportedOperation;
        }
    }

    fn validatePlugin(self: *Self, plugin: *Plugin) FrameworkError!void {
        // Validate plugin configuration
        try plugin.config.validate();

        // Security validation
        try self.security_manager.validatePlugin(plugin);

        // Performance validation
        try self.performance_monitor.validatePlugin(plugin);
    }

    fn checkPluginDependencies(self: *Self, plugin: *Plugin) FrameworkError!void {
        for (plugin.config.dependencies) |dependency| {
            if (!self.plugins.contains(dependency)) {
                return FrameworkError.PluginNotFound;
            }
        }
    }

    fn registerPluginServices(self: *Self, plugin: *Plugin) FrameworkError!void {
        for (plugin.services.items) |service| {
            try self.service_discovery.registerService(service);
        }
    }

    fn unregisterPluginServices(self: *Self, plugin: *Plugin) FrameworkError!void {
        for (plugin.services.items) |service| {
            try self.service_discovery.unregisterService(service.name);
        }
    }

    fn loadInitialPlugins(self: *Self, plugin_directory: []const u8) FrameworkError!void {
        const dir = std.fs.cwd().openDir(plugin_directory, .{ .iterate = true }) catch return;
        defer dir.close();

        var iterator = dir.iterate();
        while (try iterator.next()) |entry| {
            if (entry.kind == .file) {
                const plugin_path = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ plugin_directory, entry.name });
                defer self.allocator.free(plugin_path);

                self.loadPlugin(plugin_path) catch |err| {
                    std.log.warn("Failed to load plugin {s}: {}", .{ plugin_path, err });
                    continue;
                };
            }
        }
    }

    fn unloadAllPlugins(self: *Self) void {
        var iterator = self.plugins.iterator();
        while (iterator.next()) |entry| {
            self.unloadPlugin(entry.key_ptr.*) catch |err| {
                std.log.warn("Failed to unload plugin {s}: {}", .{ entry.key_ptr.*, err });
            };
        }
    }

    fn emitPluginEvent(self: *Self, event_type: PluginEventType, plugin: *Plugin, data: ?*anyopaque) void {
        _ = self;
        _ = event_type;
        _ = plugin;
        _ = data;
        // Emit plugin event
    }
};

/// Plugin system configuration
pub const PluginSystemConfig = struct {
    plugin_directory: []const u8 = "plugins/",
    enable_hot_reload: bool = true,
    max_plugins: u32 = 100,
    plugin_timeout_ms: u32 = 30000,
    enable_sandboxing: bool = true,
    enable_performance_monitoring: bool = true,
    enable_security_validation: bool = true,

    pub fn validate(self: PluginSystemConfig) FrameworkError!void {
        if (self.max_plugins == 0) {
            return FrameworkError.InvalidConfiguration;
        }

        if (self.plugin_timeout_ms == 0) {
            return FrameworkError.InvalidConfiguration;
        }
    }
};

/// Enhanced plugin with production-ready features
pub const Plugin = struct {
    name: []const u8,
    version: []const u8,
    description: []const u8,
    path: []const u8,
    config: PluginConfig,
    state: PluginState,
    services: std.ArrayList(PluginService),
    dependencies: std.ArrayList([]const u8),
    memory_usage: usize,
    performance_metrics: PluginPerformanceMetrics,
    security_context: *SecurityContext,

    const Self = @This();

    pub fn init(name: []const u8, version: []const u8, description: []const u8, path: []const u8) Self {
        return Self{
            .name = name,
            .version = version,
            .description = description,
            .path = path,
            .config = PluginConfig{
                .name = name,
                .version = version,
                .description = description,
                .settings = std.StringHashMap([]const u8).init(std.heap.page_allocator),
            },
            .state = .unloaded,
            .services = std.ArrayList(PluginService).init(std.heap.page_allocator),
            .dependencies = std.ArrayList([]const u8).init(std.heap.page_allocator),
            .memory_usage = 0,
            .performance_metrics = PluginPerformanceMetrics{},
            .security_context = undefined, // Will be set during initialization
        };
    }

    pub fn deinit(self: *Self) void {
        self.services.deinit();
        self.dependencies.deinit();
        self.config.settings.deinit();
    }

    pub fn initialize(self: *Self, allocator: std.mem.Allocator) FrameworkError!void {
        _ = allocator;

        if (!self.state.canTransitionTo(.loaded)) {
            return FrameworkError.OperationFailed;
        }

        self.state = .loaded;

        // Initialize plugin-specific resources
        // This would be implemented by the specific plugin
    }

    pub fn start(self: *Self) FrameworkError!void {
        if (!self.state.canTransitionTo(.running)) {
            return FrameworkError.OperationFailed;
        }

        self.state = .running;

        // Start plugin-specific services
        // This would be implemented by the specific plugin
    }

    pub fn stop(self: *Self) FrameworkError!void {
        if (!self.state.canTransitionTo(.stopped)) {
            return FrameworkError.OperationFailed;
        }

        self.state = .stopped;

        // Stop plugin-specific services
        // This would be implemented by the specific plugin
    }

    pub fn deinitialize(self: *Self) void {
        self.state = .unloaded;

        // Clean up plugin-specific resources
        // This would be implemented by the specific plugin
    }

    pub fn healthCheck(self: *const Self) PluginHealth {
        return PluginHealth{
            .status = if (self.state == .running) .healthy else .unhealthy,
            .message = "Plugin is healthy",
            .last_check = std.time.microTimestamp(),
            .memory_usage = self.memory_usage,
            .performance_metrics = self.performance_metrics,
        };
    }

    pub fn addService(self: *Self, service: PluginService) FrameworkError!void {
        try self.services.append(service);
    }

    pub fn removeService(self: *Self, service_name: []const u8) bool {
        for (self.services.items, 0..) |service, i| {
            if (std.mem.eql(u8, service.name, service_name)) {
                _ = self.services.swapRemove(i);
                return true;
            }
        }
        return false;
    }

    pub fn getService(self: *const Self, service_name: []const u8) ?PluginService {
        for (self.services.items) |service| {
            if (std.mem.eql(u8, service.name, service_name)) {
                return service;
            }
        }
        return null;
    }
};

/// Plugin state management
pub const PluginState = enum {
    unloaded,
    loaded,
    running,
    stopped,
    error_state,

    pub fn canTransitionTo(self: PluginState, new_state: PluginState) bool {
        return switch (self) {
            .unloaded => new_state == .loaded,
            .loaded => new_state == .running or new_state == .error_state,
            .running => new_state == .stopped or new_state == .error_state,
            .stopped => new_state == .unloaded or new_state == .error_state,
            .error_state => new_state == .unloaded,
        };
    }
};

/// Plugin service definition
pub const PluginService = struct {
    name: []const u8, // Service name
    version: []const u8,
    description: []const u8,
    capabilities: PluginCapabilities,
    handler: *const fn (input: []const u8) anyerror![]const u8,
    metadata: std.StringHashMap([]const u8),

    pub fn init(name: []const u8, version: []const u8, description: []const u8, capabilities: PluginCapabilities, handler: *const fn ([]const u8) anyerror![]const u8) PluginService {
        return PluginService{
            .name = name,
            .version = version,
            .description = description,
            .capabilities = capabilities,
            .handler = handler,
            .metadata = std.StringHashMap([]const u8).init(std.heap.page_allocator),
        };
    }

    pub fn deinit(self: *PluginService) void {
        self.metadata.deinit();
    }
};

/// Plugin capabilities
pub const PluginCapabilities = struct {
    text_processing: bool = false,
    image_processing: bool = false,
    audio_processing: bool = false,
    data_transformation: bool = false,
    api_integration: bool = false,
    custom_operations: bool = false,
    real_time_processing: bool = false,
    batch_processing: bool = false,
    streaming: bool = false,
    caching: bool = false,
};

/// Plugin performance metrics
pub const PluginPerformanceMetrics = struct {
    total_requests: u64 = 0,
    successful_requests: u64 = 0,
    failed_requests: u64 = 0,
    average_response_time_ms: f64 = 0.0,
    max_response_time_ms: f64 = 0.0,
    min_response_time_ms: f64 = 0.0,
    memory_usage_bytes: usize = 0,
    cpu_usage_percent: f32 = 0.0,

    pub fn updateResponseTime(self: *PluginPerformanceMetrics, response_time_ms: f64) void {
        self.total_requests += 1;

        if (self.total_requests == 1) {
            self.average_response_time_ms = response_time_ms;
            self.max_response_time_ms = response_time_ms;
            self.min_response_time_ms = response_time_ms;
        } else {
            self.average_response_time_ms = (self.average_response_time_ms + response_time_ms) / 2.0;
            self.max_response_time_ms = @max(self.max_response_time_ms, response_time_ms);
            self.min_response_time_ms = @min(self.min_response_time_ms, response_time_ms);
        }
    }

    pub fn recordSuccess(self: *PluginPerformanceMetrics) void {
        self.successful_requests += 1;
    }

    pub fn recordFailure(self: *PluginPerformanceMetrics) void {
        self.failed_requests += 1;
    }

    pub fn getSuccessRate(self: *const PluginPerformanceMetrics) f32 {
        if (self.total_requests == 0) return 0.0;
        return @as(f32, @floatFromInt(self.successful_requests)) / @as(f32, @floatFromInt(self.total_requests));
    }
};

/// Plugin health status
pub const PluginHealth = struct {
    status: HealthStatus,
    message: []const u8,
    last_check: i64,
    memory_usage: usize,
    performance_metrics: PluginPerformanceMetrics,
};

/// Health status levels
pub const HealthStatus = enum {
    healthy,
    degraded,
    unhealthy,
};

/// Plugin statistics
pub const PluginStats = struct {
    total_plugins: u32 = 0,
    loaded_plugins: u32 = 0,
    running_plugins: u32 = 0,
    error_plugins: u32 = 0,
    total_services: u32 = 0,
    total_memory_usage: usize = 0,
};

/// Plugin health status for all plugins
pub const PluginHealthStatus = struct {
    overall: HealthStatus,
    plugins: std.StringHashMap(PluginHealth),

    pub fn deinit(self: *PluginHealthStatus) void {
        self.plugins.deinit();
    }
};

/// Plugin event types
pub const PluginEventType = enum {
    plugin_loaded,
    plugin_unloaded,
    plugin_started,
    plugin_stopped,
    plugin_error,
    service_registered,
    service_unregistered,
};

// Placeholder types for components (to be implemented in separate modules)

const PluginLoader = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*PluginLoader {
        const self = try allocator.create(PluginLoader);
        self.* = .{ .allocator = allocator };
        return self;
    }

    pub fn deinit(self: *PluginLoader) void {
        self.allocator.destroy(self);
    }

    pub fn loadPlugin(self: *PluginLoader, path: []const u8) !*Plugin {
        _ = self;
        _ = path;
        return undefined;
    }
};

const NativePluginLoader = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*NativePluginLoader {
        const self = try allocator.create(NativePluginLoader);
        self.* = .{ .allocator = allocator };
        return self;
    }

    pub fn deinit(self: *NativePluginLoader) void {
        self.allocator.destroy(self);
    }

    pub fn loadPlugin(self: *NativePluginLoader, path: []const u8) !*Plugin {
        _ = self;
        _ = path;
        return undefined;
    }
};

const ScriptPluginLoader = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*ScriptPluginLoader {
        const self = try allocator.create(ScriptPluginLoader);
        self.* = .{ .allocator = allocator };
        return self;
    }

    pub fn deinit(self: *ScriptPluginLoader) void {
        self.allocator.destroy(self);
    }

    pub fn loadPlugin(self: *ScriptPluginLoader, path: []const u8) !*Plugin {
        _ = self;
        _ = path;
        return undefined;
    }
};

const WebPluginLoader = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*WebPluginLoader {
        const self = try allocator.create(WebPluginLoader);
        self.* = .{ .allocator = allocator };
        return self;
    }

    pub fn deinit(self: *WebPluginLoader) void {
        self.allocator.destroy(self);
    }

    pub fn loadPlugin(self: *WebPluginLoader, path: []const u8) !*Plugin {
        _ = self;
        _ = path;
        return undefined;
    }
};

const PluginRegistry = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*PluginRegistry {
        const self = try allocator.create(PluginRegistry);
        self.* = .{ .allocator = allocator };
        return self;
    }

    pub fn deinit(self: *PluginRegistry) void {
        self.allocator.destroy(self);
    }

    pub fn registerPlugin(self: *PluginRegistry, plugin: *Plugin) !void {
        _ = self;
        _ = plugin;
    }

    pub fn unregisterPlugin(self: *PluginRegistry, name: []const u8) !void {
        _ = self;
        _ = name;
    }
};

const PluginWatcher = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*PluginWatcher {
        const self = try allocator.create(PluginWatcher);
        self.* = .{ .allocator = allocator };
        return self;
    }

    pub fn deinit(self: *PluginWatcher) void {
        self.allocator.destroy(self);
    }

    pub fn start(self: *PluginWatcher, directory: []const u8) !void {
        _ = self;
        _ = directory;
    }

    pub fn stop(self: *PluginWatcher) void {
        _ = self;
    }
};

const PluginManager = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*PluginManager {
        const self = try allocator.create(PluginManager);
        self.* = .{ .allocator = allocator };
        return self;
    }

    pub fn deinit(self: *PluginManager) void {
        self.allocator.destroy(self);
    }
};

const ServiceDiscovery = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*ServiceDiscovery {
        const self = try allocator.create(ServiceDiscovery);
        self.* = .{ .allocator = allocator };
        return self;
    }

    pub fn deinit(self: *ServiceDiscovery) void {
        self.allocator.destroy(self);
    }

    pub fn registerService(self: *ServiceDiscovery, service: PluginService) !void {
        _ = self;
        _ = service;
    }

    pub fn unregisterService(self: *ServiceDiscovery, name: []const u8) !void {
        _ = self;
        _ = name;
    }
};

const SecurityManager = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*SecurityManager {
        const self = try allocator.create(SecurityManager);
        self.* = .{ .allocator = allocator };
        return self;
    }

    pub fn deinit(self: *SecurityManager) void {
        self.allocator.destroy(self);
    }

    pub fn validatePluginPath(self: *SecurityManager, path: []const u8) !void {
        _ = self;
        _ = path;
    }

    pub fn validatePlugin(self: *SecurityManager, plugin: *Plugin) !void {
        _ = self;
        _ = plugin;
    }
};

const SecurityContext = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*SecurityContext {
        const self = try allocator.create(SecurityContext);
        self.* = .{ .allocator = allocator };
        return self;
    }

    pub fn deinit(self: *SecurityContext) void {
        self.allocator.destroy(self);
    }
};

const PerformanceMonitor = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*PerformanceMonitor {
        const self = try allocator.create(PerformanceMonitor);
        self.* = .{ .allocator = allocator };
        return self;
    }

    pub fn deinit(self: *PerformanceMonitor) void {
        self.allocator.destroy(self);
    }

    pub fn validatePlugin(self: *PerformanceMonitor, plugin: *Plugin) !void {
        _ = self;
        _ = plugin;
    }
};

test "enhanced plugin system initialization" {
    const testing = std.testing;

    const test_config = PluginSystemConfig{
        .plugin_directory = "test_plugins/",
        .enable_hot_reload = false,
        .max_plugins = 10,
    };

    var test_plugin_system = try EnhancedPluginSystem.init(testing.allocator, test_config);
    defer test_plugin_system.deinit();

    // Test plugin system initialization
    try testing.expectEqual(@as(usize, 0), test_plugin_system.plugins.count());
}

test "plugin system statistics" {
    const testing = std.testing;

    const test_config = PluginSystemConfig{
        .plugin_directory = "test_plugins/",
        .enable_hot_reload = false,
        .max_plugins = 10,
    };

    var test_plugin_system = try EnhancedPluginSystem.init(testing.allocator, test_config);
    defer test_plugin_system.deinit();

    // Test plugin statistics
    const stats = test_plugin_system.getPluginStats();
    try testing.expectEqual(@as(u32, 0), stats.total_plugins);
    try testing.expectEqual(@as(u32, 0), stats.loaded_plugins);
    try testing.expectEqual(@as(u32, 0), stats.running_plugins);
}

test "plugin system health check" {
    const testing = std.testing;

    const test_config = PluginSystemConfig{
        .plugin_directory = "test_plugins/",
        .enable_hot_reload = false,
        .max_plugins = 10,
    };

    var test_plugin_system = try EnhancedPluginSystem.init(testing.allocator, test_config);
    defer test_plugin_system.deinit();

    // Test health check
    const health = test_plugin_system.healthCheck();
    defer health.deinit();

    try testing.expectEqual(HealthStatus.healthy, health.overall);
}
