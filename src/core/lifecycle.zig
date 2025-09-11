//! Lifecycle Management - Framework lifecycle and component management
//!
//! This module provides lifecycle management for the Abi AI Framework,
//! including component initialization, shutdown, and state management.

const std = @import("std");

const errors = @import("errors.zig");

const FrameworkError = errors.FrameworkError;

/// Framework lifecycle states
pub const FrameworkState = enum {
    uninitialized,
    initializing,
    ready,
    running,
    shutting_down,
    stopped,
    @"error",
};

/// Framework lifecycle state methods
pub const FrameworkStateMethods = struct {
    /// Check if the framework can transition to a new state
    pub fn canTransitionTo(self: FrameworkState, new_state: FrameworkState) bool {
        return switch (self) {
            .uninitialized => new_state == .initializing,
            .initializing => new_state == .ready or new_state == .@"error",
            .ready => new_state == .running or new_state == .shutting_down or new_state == .@"error",
            .running => new_state == .shutting_down or new_state == .@"error",
            .shutting_down => new_state == .stopped or new_state == .@"error",
            .stopped => new_state == .uninitialized,
            .@"error" => new_state == .uninitialized or new_state == .shutting_down,
        };
    }

    /// Get human-readable state description
    pub fn getDescription(self: FrameworkState) []const u8 {
        return switch (self) {
            .uninitialized => "Framework is not initialized",
            .initializing => "Framework is initializing",
            .ready => "Framework is ready to start",
            .running => "Framework is running",
            .shutting_down => "Framework is shutting down",
            .stopped => "Framework has stopped",
            .@"error" => "Framework is in error state",
        };
    }
};

/// Component lifecycle states
pub const ComponentState = enum {
    uninitialized,
    initializing,
    ready,
    running,
    stopping,
    stopped,
    @"error",

    /// Check if the component can transition to a new state
    pub fn canTransitionTo(self: ComponentState, new_state: ComponentState) bool {
        return switch (self) {
            .uninitialized => new_state == .initializing,
            .initializing => new_state == .ready or new_state == .@"error",
            .ready => new_state == .running or new_state == .stopping or new_state == .@"error",
            .running => new_state == .stopping or new_state == .@"error",
            .stopping => new_state == .stopped or new_state == .@"error",
            .stopped => new_state == .uninitialized,
            .@"error" => new_state == .uninitialized or new_state == .stopping,
        };
    }

    /// Get human-readable state description
    pub fn getDescription(self: ComponentState) []const u8 {
        return switch (self) {
            .uninitialized => "Component is not initialized",
            .initializing => "Component is initializing",
            .ready => "Component is ready to start",
            .running => "Component is running",
            .stopping => "Component is stopping",
            .stopped => "Component has stopped",
            .@"error" => "Component is in error state",
        };
    }
};

/// Component lifecycle interface
pub const ComponentLifecycle = struct {
    name: []const u8,
    state: ComponentState,
    dependencies: []const []const u8,
    init_fn: ?*const fn (allocator: std.mem.Allocator) anyerror!void = null,
    start_fn: ?*const fn () anyerror!void = null,
    stop_fn: ?*const fn () anyerror!void = null,
    deinit_fn: ?*const fn () void = null,
    health_check_fn: ?*const fn () bool = null,

    const Self = @This();

    pub fn init(name: []const u8, dependencies: []const []const u8) Self {
        return Self{
            .name = name,
            .state = .uninitialized,
            .dependencies = dependencies,
        };
    }

    pub fn withInit(self: Self, fn_init: *const fn (allocator: std.mem.Allocator) anyerror!void) Self {
        return Self{
            .name = self.name,
            .state = self.state,
            .dependencies = self.dependencies,
            .init_fn = fn_init,
            .start_fn = self.start_fn,
            .stop_fn = self.stop_fn,
            .deinit_fn = self.deinit_fn,
            .health_check_fn = self.health_check_fn,
        };
    }

    pub fn withStart(self: Self, fn_start: *const fn () anyerror!void) Self {
        return Self{
            .name = self.name,
            .state = self.state,
            .dependencies = self.dependencies,
            .init_fn = self.init_fn,
            .start_fn = fn_start,
            .stop_fn = self.stop_fn,
            .deinit_fn = self.deinit_fn,
            .health_check_fn = self.health_check_fn,
        };
    }

    pub fn withStop(self: Self, fn_stop: *const fn () anyerror!void) Self {
        return Self{
            .name = self.name,
            .state = self.state,
            .dependencies = self.dependencies,
            .init_fn = self.init_fn,
            .start_fn = self.start_fn,
            .stop_fn = fn_stop,
            .deinit_fn = self.deinit_fn,
            .health_check_fn = self.health_check_fn,
        };
    }

    pub fn withDeinit(self: Self, fn_deinit: *const fn () void) Self {
        return Self{
            .name = self.name,
            .state = self.state,
            .dependencies = self.dependencies,
            .init_fn = self.init_fn,
            .start_fn = self.start_fn,
            .stop_fn = self.stop_fn,
            .deinit_fn = fn_deinit,
            .health_check_fn = self.health_check_fn,
        };
    }

    pub fn withHealthCheck(self: Self, fn_health_check: *const fn () bool) Self {
        return Self{
            .name = self.name,
            .state = self.state,
            .dependencies = self.dependencies,
            .init_fn = self.init_fn,
            .start_fn = self.start_fn,
            .stop_fn = self.stop_fn,
            .deinit_fn = self.deinit_fn,
            .health_check_fn = fn_health_check,
        };
    }

    pub fn initialize(self: *Self, allocator: std.mem.Allocator) FrameworkError!void {
        if (!self.state.canTransitionTo(.initializing)) {
            return FrameworkError.OperationFailed;
        }

        self.state = .initializing;

        if (self.init_fn) |init_fn| {
            init_fn(allocator) catch |err| {
                self.state = .@"error";
                return err;
            };
        }

        self.state = .ready;
    }

    pub fn start(self: *Self) FrameworkError!void {
        if (!self.state.canTransitionTo(.running)) {
            return FrameworkError.OperationFailed;
        }

        if (self.start_fn) |start_fn| {
            start_fn() catch |err| {
                self.state = .@"error";
                return err;
            };
        }

        self.state = .running;
    }

    pub fn stop(self: *Self) FrameworkError!void {
        if (!self.state.canTransitionTo(.stopping)) {
            return FrameworkError.OperationFailed;
        }

        self.state = .stopping;

        if (self.stop_fn) |stop_fn| {
            stop_fn() catch |err| {
                self.state = .@"error";
                return err;
            };
        }

        self.state = .stopped;
    }

    pub fn deinitialize(self: *Self) void {
        if (self.deinit_fn) |deinit_fn| {
            deinit_fn();
        }

        self.state = .uninitialized;
    }

    pub fn healthCheck(self: *const Self) bool {
        if (self.health_check_fn) |health_check_fn| {
            return health_check_fn();
        }

        return self.state == .running;
    }
};

/// Lifecycle manager for managing component lifecycles
pub const LifecycleManager = struct {
    allocator: std.mem.Allocator,
    components: std.StringHashMap(ComponentLifecycle),
    initialization_order: std.ArrayList([]const u8),
    shutdown_order: std.ArrayList([]const u8),

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .components = std.StringHashMap(ComponentLifecycle).init(allocator),
            .initialization_order = std.ArrayList([]const u8).init(allocator),
            .shutdown_order = std.ArrayList([]const u8).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        self.components.deinit();
        self.initialization_order.deinit();
        self.shutdown_order.deinit();
    }

    /// Register a component with the lifecycle manager
    pub fn registerComponent(self: *Self, component: ComponentLifecycle) FrameworkError!void {
        try self.components.put(component.name, component);

        // Recalculate initialization order
        try self.calculateInitializationOrder();
    }

    /// Unregister a component from the lifecycle manager
    pub fn unregisterComponent(self: *Self, name: []const u8) bool {
        return self.components.remove(name);
    }

    /// Get a component by name
    pub fn getComponent(self: *Self, name: []const u8) ?*ComponentLifecycle {
        return self.components.getPtr(name);
    }

    /// Initialize all components in dependency order
    pub fn initializeAll(self: *Self) FrameworkError!void {
        for (self.initialization_order.items) |component_name| {
            if (self.components.getPtr(component_name)) |component| {
                try component.initialize(self.allocator);
            }
        }
    }

    /// Start all components
    pub fn startAll(self: *Self) FrameworkError!void {
        for (self.initialization_order.items) |component_name| {
            if (self.components.getPtr(component_name)) |component| {
                try component.start();
            }
        }
    }

    /// Stop all components in reverse order
    pub fn stopAll(self: *Self) FrameworkError!void {
        // Stop components in reverse order
        var i = self.shutdown_order.items.len;
        while (i > 0) {
            i -= 1;
            const component_name = self.shutdown_order.items[i];
            if (self.components.getPtr(component_name)) |component| {
                try component.stop();
            }
        }
    }

    /// Deinitialize all components
    pub fn deinitializeAll(self: *Self) void {
        for (self.shutdown_order.items) |component_name| {
            if (self.components.getPtr(component_name)) |component| {
                component.deinitialize();
            }
        }
    }

    /// Get component states
    pub fn getComponentStates(self: *const Self) std.StringHashMap(ComponentState) {
        var states = std.StringHashMap(ComponentState).init(self.allocator);

        var iterator = self.components.iterator();
        while (iterator.next()) |entry| {
            states.put(entry.key_ptr.*, entry.value_ptr.state) catch continue;
        }

        return states;
    }

    /// Health check all components
    pub fn healthCheckAll(self: *const Self) std.StringHashMap(bool) {
        var health_status = std.StringHashMap(bool).init(self.allocator);

        var iterator = self.components.iterator();
        while (iterator.next()) |entry| {
            health_status.put(entry.key_ptr.*, entry.value_ptr.healthCheck()) catch {
                continue;
            };
        }

        return health_status;
    }

    // Private methods

    fn calculateInitializationOrder(self: *Self) FrameworkError!void {
        self.initialization_order.clearRetainingCapacity();
        self.shutdown_order.clearRetainingCapacity();

        var visited = std.StringHashMap(bool).init(self.allocator);
        defer visited.deinit();

        var temp_visited = std.StringHashMap(bool).init(self.allocator);
        defer temp_visited.deinit();

        // Topological sort to determine initialization order
        var iterator = self.components.iterator();
        while (iterator.next()) |entry| {
            const component_name = entry.key_ptr.*;
            if (!visited.contains(component_name)) {
                try self.visitComponent(component_name, &visited, &temp_visited);
            }
        }

        // Shutdown order is reverse of initialization order
        for (self.initialization_order.items) |component_name| {
            try self.shutdown_order.append(component_name);
        }
    }

    fn visitComponent(self: *Self, component_name: []const u8, visited: *std.StringHashMap(bool), temp_visited: *std.StringHashMap(bool)) FrameworkError!void {
        if (temp_visited.contains(component_name)) {
            return FrameworkError.OperationFailed; // Circular dependency
        }

        if (visited.contains(component_name)) {
            return;
        }

        temp_visited.put(component_name, true) catch return;

        if (self.components.get(component_name)) |component| {
            // Visit dependencies first
            for (component.dependencies) |dependency| {
                try self.visitComponent(dependency, visited, temp_visited);
            }
        }

        temp_visited.remove(component_name);
        visited.put(component_name, true) catch return;

        try self.initialization_order.append(component_name);
    }
};

/// Lifecycle event types
pub const LifecycleEvent = enum {
    component_initialized,
    component_started,
    component_stopped,
    component_deinitialized,
    component_error,
    framework_initialized,
    framework_started,
    framework_stopped,
    framework_error,
};

/// Lifecycle event handler
pub const LifecycleEventHandler = struct {
    event_type: LifecycleEvent,
    handler: *const fn (event: LifecycleEvent, component_name: ?[]const u8, data: ?*anyopaque) void,
    context: ?*anyopaque = null,
};

/// Lifecycle event manager
pub const LifecycleEventManager = struct {
    allocator: std.mem.Allocator,
    event_handlers: std.ArrayList(LifecycleEventHandler),

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .event_handlers = std.ArrayList(LifecycleEventHandler).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        self.event_handlers.deinit();
    }

    /// Register an event handler
    pub fn registerHandler(self: *Self, handler: LifecycleEventHandler) !void {
        try self.event_handlers.append(handler);
    }

    /// Unregister an event handler
    pub fn unregisterHandler(self: *Self, index: usize) void {
        if (index < self.event_handlers.items.len) {
            _ = self.event_handlers.swapRemove(index);
        }
    }

    /// Emit a lifecycle event
    pub fn emitEvent(self: *const Self, event: LifecycleEvent, component_name: ?[]const u8, data: ?*anyopaque) void {
        for (self.event_handlers.items) |handler| {
            if (handler.event_type == event) {
                handler.handler(event, component_name, data);
            }
        }
    }
};

/// Lifecycle statistics
pub const LifecycleStats = struct {
    total_components: u32 = 0,
    initialized_components: u32 = 0,
    running_components: u32 = 0,
    stopped_components: u32 = 0,
    error_components: u32 = 0,
    initialization_time_ms: u64 = 0,
    startup_time_ms: u64 = 0,
    shutdown_time_ms: u64 = 0,

    pub fn update(self: *LifecycleStats, components: std.StringHashMap(ComponentLifecycle)) void {
        self.total_components = @as(u32, @intCast(components.count()));
        self.initialized_components = 0;
        self.running_components = 0;
        self.stopped_components = 0;
        self.error_components = 0;

        var iterator = components.iterator();
        while (iterator.next()) |entry| {
            switch (entry.value_ptr.state) {
                .ready => self.initialized_components += 1,
                .running => self.running_components += 1,
                .stopped => self.stopped_components += 1,
                .@"error" => self.error_components += 1,
                else => {},
            }
        }
    }
};

test "component lifecycle" {
    const testing = std.testing;

    // Test component lifecycle
    var component = ComponentLifecycle.init("test_component", &[_][]const u8{});
    try testing.expectEqual(ComponentState.uninitialized, component.state);

    // Test state transitions
    try testing.expect(component.state.canTransitionTo(.initializing));
    try testing.expect(!component.state.canTransitionTo(.running));

    // Test initialization
    try component.initialize(testing.allocator);
    try testing.expectEqual(ComponentState.ready, component.state);

    // Test start
    try component.start();
    try testing.expectEqual(ComponentState.running, component.state);

    // Test stop
    try component.stop();
    try testing.expectEqual(ComponentState.stopped, component.state);

    // Test deinitialize
    component.deinitialize();
    try testing.expectEqual(ComponentState.uninitialized, component.state);
}

test "lifecycle manager" {
    const testing = std.testing;

    // Test lifecycle manager
    var manager = LifecycleManager.init(testing.allocator);
    defer manager.deinit();

    // Test component registration
    const component1 = ComponentLifecycle.init("component1", &[_][]const u8{});
    const component2 = ComponentLifecycle.init("component2", &[_][]const u8{"component1"});

    try manager.registerComponent(component1);
    try manager.registerComponent(component2);

    // Test initialization order
    try testing.expectEqual(@as(usize, 2), manager.initialization_order.items.len);
    try testing.expectEqualStrings("component1", manager.initialization_order.items[0]);
    try testing.expectEqualStrings("component2", manager.initialization_order.items[1]);

    // Test shutdown order
    try testing.expectEqual(@as(usize, 2), manager.shutdown_order.items.len);
    try testing.expectEqualStrings("component2", manager.shutdown_order.items[0]);
    try testing.expectEqualStrings("component1", manager.shutdown_order.items[1]);
}

test "lifecycle event manager" {
    const testing = std.testing;

    // Test lifecycle event manager
    var event_manager = LifecycleEventManager.init(testing.allocator);
    defer event_manager.deinit();

    // Test event handler registration
    const handler = LifecycleEventHandler{
        .event_type = .component_initialized,
        .handler = testEventHandler,
    };

    try event_manager.registerHandler(handler);
    try testing.expectEqual(@as(usize, 1), event_manager.event_handlers.items.len);

    // Test event emission
    event_manager.emitEvent(.component_initialized, "test_component", null);
}

fn testEventHandler(event: LifecycleEvent, component_name: ?[]const u8, data: ?*anyopaque) void {
    _ = event;
    _ = component_name;
    _ = data;
    // Test event handler
}
