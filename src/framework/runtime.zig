const std = @import("std");
const feature_manager = @import("feature_manager.zig");
const catalog = @import("catalog.zig");
const state = @import("state.zig");
const lifecycle_mod = @import("../shared/core/lifecycle.zig");
const core_logging = @import("../shared/core/logging.zig");

const FeatureManager = feature_manager.FeatureManager;
const Lifecycle = lifecycle_mod.Lifecycle;
const RuntimeOptions = state.RuntimeOptions;
const RuntimeState = state.RuntimeState;

/// Aggregated runtime responsible for configuring core services and feature modules.
pub const Runtime = struct {
    gpa: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    allocator: std.mem.Allocator,
    lifecycle: Lifecycle,
    features: FeatureManager,
    state: *RuntimeState,

    pub const Options = RuntimeOptions;

    /// Instantiate a runtime using the provided allocator and options.
    pub fn init(allocator: std.mem.Allocator, options: Options) !Runtime {
        var arena = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();

        const scoped_allocator = arena.allocator();
        var lifecycle = try Lifecycle.init(scoped_allocator, .{ .reserve_observers = 8 });
        errdefer lifecycle.deinit();

        const state_ptr = try scoped_allocator.create(RuntimeState);
        state_ptr.* = RuntimeState.init(scoped_allocator, options);

        var manager = try FeatureManager.init(
            scoped_allocator,
            .{ .allocator = scoped_allocator, .context = state_ptr },
            &catalog.descriptors,
        );
        errdefer manager.deinit();

        var runtime = Runtime{
            .gpa = allocator,
            .arena = arena,
            .allocator = scoped_allocator,
            .lifecycle = lifecycle,
            .features = manager,
            .state = state_ptr,
        };

        try runtime.lifecycle.addObserver(.{
            .name = "lifecycle-logger",
            .stages = lifecycle_mod.StageMask{ .running = true, .shutting_down = true, .terminated = true },
            .priority = 5,
            .callback = lifecycleEventLogger,
        });

        try runtime.lifecycle.advance(.bootstrapping);

        if (options.ensure_core) try runtime.features.ensure("core.kernel");
        try runtime.features.ensure("core.lifecycle");
        if (options.ensure_logging) {
            try runtime.features.ensure("core.logging.bootstrap");
            try runtime.features.ensure("core.logging.structured");
        }
        if (options.ensure_plugin_system) {
            try runtime.features.ensure("shared.plugins");
        }

        for (options.enable_features) |feature_name| {
            try runtime.features.ensure(feature_name);
        }

        for (options.enable_categories) |category| {
            try runtime.features.ensureCategory(category);
        }

        if (options.enable_all_features) {
            try runtime.features.ensureAll();
        }

        try runtime.lifecycle.advance(.running);

        return runtime;
    }

    /// Shut down the runtime, reversing feature initialization order.
    pub fn deinit(self: *Runtime) void {
        if (self.lifecycle.currentStage() == .running) {
            self.lifecycle.advance(.shutting_down) catch {};
        }

        self.features.shutdown();
        self.features.deinit();

        if (self.lifecycle.currentStage() != .terminated) {
            self.lifecycle.advance(.terminated) catch {};
        }

        self.lifecycle.deinit();
        self.state.deinit();
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn getLifecycle(self: *Runtime) *Lifecycle {
        return &self.lifecycle;
    }

    pub fn getFeatureManager(self: *Runtime) *FeatureManager {
        return &self.features;
    }

    pub fn options(self: Runtime) Options {
        return self.state.options;
    }

    pub fn allocator(self: Runtime) std.mem.Allocator {
        return self.allocator;
    }

    fn lifecycleEventLogger(transition: lifecycle_mod.Transition, _: *Lifecycle) anyerror!void {
        core_logging.log.info(
            "lifecycle transition {s} -> {s}",
            .{ @tagName(transition.from), @tagName(transition.to) },
        );
    }
};

