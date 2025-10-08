const std = @import("std");
const config = @import("../core/config.zig");
const errors = @import("../core/errors.zig");

const FrameworkError = errors.FrameworkError;
const PluginConfig = config.PluginConfig;

/// Plugin system configuration shared across the runtime.
pub const PluginSystemConfig = struct {
    plugin_directory: []const u8 = "plugins/",
    enable_hot_reload: bool = true,
    max_plugins: u32 = 100,
    plugin_timeout_ms: u32 = 30_000,
    enable_sandboxing: bool = true,
    enable_performance_monitoring: bool = true,
    enable_security_validation: bool = true,

    pub fn validate(self: PluginSystemConfig) FrameworkError!void {
        if (self.max_plugins == 0) return FrameworkError.InvalidConfiguration;
        if (self.plugin_timeout_ms == 0) return FrameworkError.InvalidConfiguration;
        if (self.plugin_directory.len == 0) return FrameworkError.InvalidConfiguration;
    }
};

/// Primary entry point for managing plugins at runtime.
pub const EnhancedPluginSystem = struct {
    allocator: std.mem.Allocator,
    config: PluginSystemConfig,
    plugins: std.StringHashMap(*Plugin),
    plugin_registry: PluginRegistry,
    plugin_watcher: PluginWatcher,
    plugin_manager: PluginManager,
    service_discovery: ServiceDiscovery,
    security_manager: SecurityManager,
    performance_monitor: PerformanceMonitor,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, system_config: PluginSystemConfig) FrameworkError!*Self {
        try system_config.validate();

        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        self.* = .{
            .allocator = allocator,
            .config = system_config,
            .plugins = std.StringHashMap(*Plugin).init(allocator),
            .plugin_registry = PluginRegistry.init(allocator),
            .plugin_watcher = PluginWatcher.init(allocator),
            .plugin_manager = PluginManager.init(system_config.max_plugins),
            .service_discovery = ServiceDiscovery.init(allocator),
            .security_manager = SecurityManager.init(system_config.plugin_directory),
            .performance_monitor = PerformanceMonitor.init(allocator),
        };

        if (system_config.enable_hot_reload) {
            try self.plugin_watcher.start(system_config.plugin_directory);
        }

        self.loadInitialPlugins(system_config.plugin_directory) catch |err| {
            std.log.warn("plugin bootstrap deferred: {}", .{err});
        };

        return self;
    }

    pub fn deinit(self: *Self) void {
        if (self.config.enable_hot_reload) {
            self.plugin_watcher.stop();
        }

        self.unloadAllPlugins();

        self.plugin_registry.deinit();
        self.service_discovery.deinit();
        self.security_manager.deinit();
        self.performance_monitor.deinit();
        self.plugins.deinit();

        self.allocator.destroy(self);
    }

    pub fn loadPlugin(self: *Self, path: []const u8) FrameworkError!*Plugin {
        try self.security_manager.validatePluginPath(path);

        const name_hint = derivePluginName(path);
        if (self.plugins.contains(name_hint)) {
            return FrameworkError.InvalidOperation;
        }

        try self.plugin_manager.register();
        errdefer self.plugin_manager.unregister();

        const plugin = try Plugin.create(self.allocator, name_hint, "0.1.0", "auto discovered plugin", path);
        var plugin_ready = false;
        errdefer {
            if (!plugin_ready) plugin.destroy();
        }

        try self.validatePlugin(plugin);
        try self.checkPluginDependencies(plugin);
        try plugin.initialize(self.allocator);

        var inserted = false;
        errdefer {
            if (inserted) {
                _ = self.plugins.remove(plugin.name);
            }
        }

        try self.plugins.put(plugin.name, plugin);
        inserted = true;

        try self.plugin_registry.registerPlugin(plugin);
        try self.registerPluginServices(plugin);
        try plugin.start();

        self.emitPluginEvent(.plugin_loaded, plugin, null);

        plugin_ready = true;
        return plugin;
    }

    pub fn unloadPlugin(self: *Self, name: []const u8) FrameworkError!void {
        const plugin = self.plugins.get(name) orelse return FrameworkError.InvalidOperation;

        if (plugin.state == .running) {
            try plugin.stop();
        }

        try self.unregisterPluginServices(plugin);
        try self.plugin_registry.unregisterPlugin(name);

        if (!self.plugins.remove(name)) {
            return FrameworkError.InvalidOperation;
        }

        self.plugin_manager.unregister();

        plugin.deinitialize();
        self.emitPluginEvent(.plugin_unloaded, plugin, null);
        plugin.destroy();
    }

    pub fn reloadPlugin(self: *Self, name: []const u8) FrameworkError!void {
        const plugin = self.plugins.get(name) orelse return FrameworkError.InvalidOperation;
        const path_copy = try self.allocator.dupe(u8, plugin.path);
        defer self.allocator.free(path_copy);

        try self.unloadPlugin(name);
        _ = try self.loadPlugin(path_copy);
    }

    pub fn getPlugin(self: *Self, name: []const u8) ?*Plugin {
        return self.plugins.get(name);
    }

    pub fn listPlugins(self: *const Self) []const []const u8 {
        var names = std.ArrayList([]const u8).init(self.allocator);
        defer names.deinit(self.allocator);

        var iterator = self.plugins.iterator();
        while (iterator.next()) |entry| {
            names.append(self.allocator, entry.key_ptr.*) catch {
                std.log.warn("failed to collect plugin name {s}", .{entry.key_ptr.*});
                break;
            };
        }

        return names.toOwnedSlice(self.allocator) catch &[_][]const u8{};
    }

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

    pub fn healthCheck(self: *const Self) PluginHealthStatus {
        var status = PluginHealthStatus{
            .overall = .healthy,
            .plugins = std.StringHashMap(PluginHealth).init(self.allocator),
        };

        var iterator = self.plugins.iterator();
        while (iterator.next()) |entry| {
            const plugin = entry.value_ptr.*;
            const health = plugin.healthCheck();

            status.plugins.put(entry.key_ptr.*, health) catch {
                status.overall = .degraded;
                continue;
            };

            if (health.status != .healthy and status.overall == .healthy) {
                status.overall = .degraded;
            }
        }

        return status;
    }

    fn validatePlugin(self: *Self, plugin: *Plugin) FrameworkError!void {
        try plugin.config.validate();

        if (self.config.enable_security_validation) {
            try self.security_manager.validatePlugin(plugin);
        }

        if (self.config.enable_performance_monitoring) {
            try self.performance_monitor.validatePlugin(plugin);
        }
    }

    fn checkPluginDependencies(self: *Self, plugin: *Plugin) FrameworkError!void {
        for (plugin.config.dependencies) |dependency| {
            if (!self.plugins.contains(dependency)) {
                return FrameworkError.InvalidOperation;
            }
        }
    }

    fn registerPluginServices(self: *Self, plugin: *Plugin) FrameworkError!void {
        var index: usize = 0;
        while (index < plugin.services.items.len) : (index += 1) {
            const service_ptr = &plugin.services.items[index];
            try self.service_discovery.registerService(plugin, service_ptr);
        }
    }

    fn unregisterPluginServices(self: *Self, plugin: *Plugin) FrameworkError!void {
        var index: usize = 0;
        while (index < plugin.services.items.len) : (index += 1) {
            const name = plugin.services.items[index].name;
            try self.service_discovery.unregisterService(name);
        }
    }

    fn loadInitialPlugins(self: *Self, plugin_directory: []const u8) FrameworkError!void {
        const dir = std.fs.cwd().openDir(plugin_directory, .{ .iterate = true }) catch return;
        defer dir.close();

        var iterator = dir.iterate();
        while (try iterator.next()) |entry| {
            if (entry.kind != .file) continue;

            const plugin_path = std.fs.path.join(self.allocator, &.{ plugin_directory, entry.name }) catch {
                continue;
            };
            defer self.allocator.free(plugin_path);

            self.loadPlugin(plugin_path) catch |err| {
                std.log.warn("failed to load plugin {s}: {}", .{ plugin_path, err });
            };
        }
    }

    fn unloadAllPlugins(self: *Self) void {
        if (self.plugins.count() == 0) return;

        var names = std.ArrayList([]const u8).init(self.allocator);
        defer names.deinit(self.allocator);

        var iterator = self.plugins.iterator();
        while (iterator.next()) |entry| {
            names.append(self.allocator, entry.key_ptr.*) catch {
                std.log.warn("failed to queue plugin {s} for shutdown", .{entry.key_ptr.*});
                break;
            };
        }

        for (names.items) |plugin_name| {
            self.unloadPlugin(plugin_name) catch |err| {
                std.log.warn("failed to unload plugin {s}: {}", .{ plugin_name, err });
            };
        }
    }

    fn emitPluginEvent(self: *Self, event_type: PluginEventType, plugin: *Plugin, data: ?*anyopaque) void {
        _ = self;
        _ = event_type;
        _ = plugin;
        _ = data;
    }
};

fn derivePluginName(path: []const u8) []const u8 {
    const base = std.fs.path.basename(path);
    if (std.mem.lastIndexOfScalar(u8, base, '.')) |dot_index| {
        return base[0..dot_index];
    }
    return base;
}

pub const Plugin = struct {
    allocator: std.mem.Allocator,
    name: []const u8,
    version: []const u8,
    description: []const u8,
    path: []const u8,
    config: PluginConfig,
    state: PluginState = .unloaded,
    services: std.ArrayList(PluginService),
    dependencies: std.ArrayList([]const u8),
    memory_usage: usize = 0,
    performance_metrics: PluginPerformanceMetrics = .{},
    security_context: ?*SecurityContext = null,

    pub fn create(allocator: std.mem.Allocator, name: []const u8, version: []const u8, description: []const u8, path: []const u8) FrameworkError!*Plugin {
        const self = try allocator.create(Plugin);
        errdefer allocator.destroy(self);

        const name_copy = try allocator.dupe(u8, name);
        errdefer allocator.free(name_copy);

        const version_copy = try allocator.dupe(u8, version);
        errdefer allocator.free(version_copy);

        const description_copy = try allocator.dupe(u8, description);
        errdefer allocator.free(description_copy);

        const path_copy = try allocator.dupe(u8, path);
        errdefer allocator.free(path_copy);

        self.* = .{
            .allocator = allocator,
            .name = name_copy,
            .version = version_copy,
            .description = description_copy,
            .path = path_copy,
            .config = PluginConfig{
                .name = name_copy,
                .version = version_copy,
                .description = description_copy,
                .settings = std.StringHashMap([]const u8).init(allocator),
            },
            .services = std.ArrayList(PluginService).init(allocator),
            .dependencies = std.ArrayList([]const u8).init(allocator),
            .memory_usage = 0,
            .performance_metrics = PluginPerformanceMetrics{},
            .security_context = null,
        };

        errdefer self.config.settings.deinit();

        return self;
    }

    pub fn destroy(self: *Plugin) void {
        var index: usize = 0;
        while (index < self.services.items.len) : (index += 1) {
            var service = &self.services.items[index];
            service.deinit();
        }
        self.services.deinit(self.allocator);

        for (self.dependencies.items) |dependency| {
            self.allocator.free(dependency);
        }
        self.dependencies.deinit(self.allocator);

        self.config.settings.deinit();

        if (self.security_context) |ctx| {
            ctx.deinit();
        }

        self.allocator.free(self.name);
        self.allocator.free(self.version);
        self.allocator.free(self.description);
        self.allocator.free(self.path);

        self.allocator.destroy(self);
    }

    pub fn initialize(self: *Plugin, allocator: std.mem.Allocator) FrameworkError!void {
        _ = allocator;
        if (!self.state.canTransitionTo(.loaded)) return FrameworkError.InvalidState;

        if (self.security_context == null) {
            self.security_context = try SecurityContext.create(self.allocator);
        }

        self.state = .loaded;
    }

    pub fn start(self: *Plugin) FrameworkError!void {
        if (!self.state.canTransitionTo(.running)) return FrameworkError.InvalidState;
        self.state = .running;
    }

    pub fn stop(self: *Plugin) FrameworkError!void {
        if (!self.state.canTransitionTo(.stopped)) return FrameworkError.InvalidState;
        self.state = .stopped;
    }

    pub fn deinitialize(self: *Plugin) void {
        self.state = .unloaded;
        if (self.security_context) |ctx| {
            ctx.deinit();
            self.security_context = null;
        }
    }

    pub fn healthCheck(self: *const Plugin) PluginHealth {
        const timestamp = std.time.microTimestamp();
        return PluginHealth{
            .status = if (self.state == .running) .healthy else .degraded,
            .message = "plugin heartbeat",
            .last_check = @as(i64, @intCast(timestamp)),
            .memory_usage = self.memory_usage,
            .performance_metrics = self.performance_metrics,
        };
    }

    pub fn addService(self: *Plugin, service: PluginService) FrameworkError!void {
        try self.services.append(self.allocator, service);
    }

    pub fn removeService(self: *Plugin, name: []const u8) bool {
        for (self.services.items, 0..) |service, idx| {
            if (std.mem.eql(u8, service.name, name)) {
                const removed = self.services.swapRemove(idx);
                removed.deinit();
                return true;
            }
        }
        return false;
    }

    pub fn getService(self: *Plugin, name: []const u8) ?*PluginService {
        for (self.services.items, 0..) |service, idx| {
            if (std.mem.eql(u8, service.name, name)) {
                return &self.services.items[idx];
            }
        }
        return null;
    }
};

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

pub const PluginService = struct {
    name: []const u8,
    version: []const u8,
    description: []const u8,
    capabilities: PluginCapabilities,
    handler: *const fn ([]const u8) anyerror![]const u8,

    pub fn init(name: []const u8, version: []const u8, description: []const u8, capabilities: PluginCapabilities, handler: *const fn ([]const u8) anyerror![]const u8) PluginService {
        return .{
            .name = name,
            .version = version,
            .description = description,
            .capabilities = capabilities,
            .handler = handler,
        };
    }

    pub fn deinit(self: *PluginService) void {
        _ = self;
    }
};

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

pub const PluginHealth = struct {
    status: HealthStatus,
    message: []const u8,
    last_check: i64,
    memory_usage: usize,
    performance_metrics: PluginPerformanceMetrics,
};

pub const HealthStatus = enum {
    healthy,
    degraded,
    unhealthy,
};

pub const PluginStats = struct {
    total_plugins: u32 = 0,
    loaded_plugins: u32 = 0,
    running_plugins: u32 = 0,
    error_plugins: u32 = 0,
    total_services: u32 = 0,
    total_memory_usage: usize = 0,
};

pub const PluginHealthStatus = struct {
    overall: HealthStatus,
    plugins: std.StringHashMap(PluginHealth),

    pub fn deinit(self: *PluginHealthStatus) void {
        self.plugins.deinit();
    }
};

pub const PluginEventType = enum {
    plugin_loaded,
    plugin_unloaded,
    plugin_started,
    plugin_stopped,
    plugin_error,
    service_registered,
    service_unregistered,
};

const PluginRegistry = struct {
    entries: std.StringHashMap(RegistryEntry),
    allocator: std.mem.Allocator,

    const RegistryEntry = struct {
        version: []const u8,
        path: []const u8,
    };

    pub fn init(allocator: std.mem.Allocator) PluginRegistry {
        return .{
            .entries = std.StringHashMap(RegistryEntry).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *PluginRegistry) void {
        self.entries.deinit();
    }

    pub fn registerPlugin(self: *PluginRegistry, plugin: *Plugin) FrameworkError!void {
        try self.entries.put(plugin.name, .{ .version = plugin.version, .path = plugin.path });
    }

    pub fn unregisterPlugin(self: *PluginRegistry, name: []const u8) FrameworkError!void {
        if (!self.entries.remove(name)) {
            return FrameworkError.InvalidOperation;
        }
    }
};

const PluginWatcher = struct {
    allocator: std.mem.Allocator,
    running: bool = false,

    pub fn init(allocator: std.mem.Allocator) PluginWatcher {
        return .{ .allocator = allocator, .running = false };
    }

    pub fn start(self: *PluginWatcher, directory: []const u8) FrameworkError!void {
        _ = directory;
        self.running = true;
    }

    pub fn stop(self: *PluginWatcher) void {
        self.running = false;
    }
};

const PluginManager = struct {
    max_plugins: u32,
    active_plugins: u32 = 0,

    pub fn init(max_plugins: u32) PluginManager {
        return .{ .max_plugins = max_plugins, .active_plugins = 0 };
    }

    pub fn register(self: *PluginManager) FrameworkError!void {
        if (self.active_plugins >= self.max_plugins) {
            return FrameworkError.ResourceLimitExceeded;
        }
        self.active_plugins += 1;
    }

    pub fn unregister(self: *PluginManager) void {
        if (self.active_plugins > 0) {
            self.active_plugins -= 1;
        }
    }
};

const ServiceDiscovery = struct {
    allocator: std.mem.Allocator,
    services: std.StringHashMap(ServiceRecord),

    const ServiceRecord = struct {
        plugin_name: []const u8,
        version: []const u8,
        capabilities: PluginCapabilities,
    };

    pub fn init(allocator: std.mem.Allocator) ServiceDiscovery {
        return .{
            .allocator = allocator,
            .services = std.StringHashMap(ServiceRecord).init(allocator),
        };
    }

    pub fn deinit(self: *ServiceDiscovery) void {
        self.services.deinit();
    }

    pub fn registerService(self: *ServiceDiscovery, plugin: *Plugin, service: *const PluginService) FrameworkError!void {
        try self.services.put(service.name, .{
            .plugin_name = plugin.name,
            .version = service.version,
            .capabilities = service.capabilities,
        });
    }

    pub fn unregisterService(self: *ServiceDiscovery, name: []const u8) FrameworkError!void {
        if (!self.services.remove(name)) {
            return FrameworkError.InvalidOperation;
        }
    }
};

const SecurityManager = struct {
    root_directory: []const u8,

    pub fn init(root_directory: []const u8) SecurityManager {
        return .{ .root_directory = root_directory };
    }

    pub fn deinit(self: *SecurityManager) void {
        _ = self;
    }

    pub fn validatePluginPath(self: *SecurityManager, path: []const u8) FrameworkError!void {
        if (path.len == 0) return FrameworkError.InvalidInput;
        if (std.fs.path.isAbsolute(path)) return FrameworkError.InvalidInput;
        if (std.mem.indexOf(u8, path, "..")) |_| {
            return FrameworkError.InvalidInput;
        }

        _ = self.root_directory; // Future directory scoping.
    }

    pub fn validatePlugin(self: *SecurityManager, plugin: *Plugin) FrameworkError!void {
        try self.validatePluginPath(plugin.path);
        if (plugin.config.name.len == 0) {
            return FrameworkError.InvalidConfiguration;
        }
    }
};

const SecurityContext = struct {
    allocator: std.mem.Allocator,

    pub fn create(allocator: std.mem.Allocator) FrameworkError!*SecurityContext {
        const ctx = try allocator.create(SecurityContext);
        ctx.* = .{ .allocator = allocator };
        return ctx;
    }

    pub fn deinit(self: *SecurityContext) void {
        self.allocator.destroy(self);
    }
};

const PerformanceMonitor = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) PerformanceMonitor {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *PerformanceMonitor) void {
        _ = self;
    }

    pub fn validatePlugin(self: *PerformanceMonitor, plugin: *Plugin) FrameworkError!void {
        _ = self;
        if (plugin.performance_metrics.cpu_usage_percent > 100.0) {
            return FrameworkError.InvalidState;
        }
    }
};

test "enhanced plugin system initialization" {
    const testing = std.testing;

    const test_config = PluginSystemConfig{
        .plugin_directory = "test_plugins/",
        .enable_hot_reload = false,
        .max_plugins = 10,
    };

    var system = try EnhancedPluginSystem.init(testing.allocator, test_config);
    defer system.deinit();

    try testing.expectEqual(@as(usize, 0), system.plugins.count());
}

test "plugin system statistics" {
    const testing = std.testing;

    const test_config = PluginSystemConfig{
        .plugin_directory = "test_plugins/",
        .enable_hot_reload = false,
        .max_plugins = 10,
    };

    var system = try EnhancedPluginSystem.init(testing.allocator, test_config);
    defer system.deinit();

    const stats = system.getPluginStats();
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

    var system = try EnhancedPluginSystem.init(testing.allocator, test_config);
    defer system.deinit();

    const health = system.healthCheck();
    defer health.deinit();

    try testing.expectEqual(HealthStatus.healthy, health.overall);
}
