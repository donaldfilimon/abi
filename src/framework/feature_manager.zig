const std = @import("std");

/// Categories used to group framework features.
pub const FeatureCategory = enum {
    core,
    logging,
    ai,
    gpu,
    database,
    web,
    monitoring,
    connectors,
    plugins,
    utilities,
};

/// Execution context passed to feature callbacks.
pub const Environment = struct {
    allocator: std.mem.Allocator,
    context: ?*anyopaque = null,

    /// Attempt to reinterpret the opaque context pointer as the requested type.
    pub fn contextAs(self: Environment, comptime T: type) ?*T {
        if (self.context) |ptr| {
            return @ptrCast(*T, @alignCast(@alignOf(T), ptr));
        }
        return null;
    }
};

/// Initialization routine for a feature.
pub const InitFn = *const fn (Environment) anyerror!void;

/// Shutdown routine for a feature.
pub const DeinitFn = *const fn (Environment) void;

/// Static metadata describing an available feature.
pub const FeatureDescriptor = struct {
    name: []const u8,
    display_name: []const u8,
    category: FeatureCategory,
    description: []const u8 = "",
    dependencies: []const []const u8 = &.{},
    init: ?InitFn = null,
    deinit: ?DeinitFn = null,
};

/// Tracks the initialization status for a descriptor.
const InitState = enum { dormant, initializing, initialized };

/// Error set returned by the feature manager.
pub const Error = error{
    UnknownFeature,
    DuplicateFeature,
    CyclicDependency,
};

/// Orchestrates feature initialization and shutdown with dependency management.
pub const FeatureManager = struct {
    allocator: std.mem.Allocator,
    environment: Environment,
    descriptors: []const FeatureDescriptor,
    index_by_name: std.StringHashMap(usize),
    states: []InitState,
    init_order: std.ArrayList(usize),

    /// Construct a manager for a static descriptor set.
    pub fn init(
        allocator: std.mem.Allocator,
        environment: Environment,
        descriptors: []const FeatureDescriptor,
    ) !FeatureManager {
        var index_by_name = std.StringHashMap(usize).init(allocator);
        errdefer index_by_name.deinit();

        var states = try allocator.alloc(InitState, descriptors.len);
        errdefer allocator.free(states);
        @memset(states, InitState.dormant);

        for (descriptors, 0..) |descriptor, idx| {
            if (index_by_name.contains(descriptor.name)) {
                return Error.DuplicateFeature;
            }
            try index_by_name.put(descriptor.name, idx);
        }

        return .{
            .allocator = allocator,
            .environment = environment,
            .descriptors = descriptors,
            .index_by_name = index_by_name,
            .states = states,
            .init_order = std.ArrayList(usize).init(allocator),
        };
    }

    /// Release memory owned by the manager. The caller must call `shutdown` first.
    pub fn deinit(self: *FeatureManager) void {
        self.index_by_name.deinit();
        self.init_order.deinit();
        self.allocator.free(self.states);
        self.* = undefined;
    }

    /// Resolve a descriptor by name.
    fn descriptorIndex(self: *FeatureManager, name: []const u8) Error!usize {
        if (self.index_by_name.get(name)) |idx| {
            return idx;
        }
        return Error.UnknownFeature;
    }

    /// Ensure a feature and its dependencies are initialized.
    pub fn ensure(self: *FeatureManager, name: []const u8) anyerror!void {
        const index = try self.descriptorIndex(name);
        try self.ensureByIndex(index);
    }

    /// Ensure an entire category of features is initialized.
    pub fn ensureCategory(self: *FeatureManager, category: FeatureCategory) anyerror!void {
        for (self.descriptors, 0..) |_, idx| {
            if (self.descriptors[idx].category == category) {
                try self.ensureByIndex(idx);
            }
        }
    }

    /// Ensure all descriptors are initialized in dependency order.
    pub fn ensureAll(self: *FeatureManager) anyerror!void {
        for (self.descriptors, 0..) |_, idx| {
            try self.ensureByIndex(idx);
        }
    }

    /// Return whether a feature has already been initialized.
    pub fn isInitialized(self: FeatureManager, name: []const u8) bool {
        if (self.index_by_name.get(name)) |idx| {
            return self.states[idx] == .initialized;
        }
        return false;
    }

    /// Number of features that have been initialized.
    pub fn initializedCount(self: FeatureManager) usize {
        var count: usize = 0;
        for (self.states) |state| {
            if (state == .initialized) count += 1;
        }
        return count;
    }

    /// Shut down initialized features in reverse order.
    pub fn shutdown(self: *FeatureManager) void {
        while (self.init_order.popOrNull()) |idx| {
            const descriptor = self.descriptors[idx];
            if (self.states[idx] == .initialized) {
                if (descriptor.deinit) |deinit_fn| {
                    deinit_fn(self.environment);
                }
                self.states[idx] = .dormant;
            }
        }
        self.init_order.shrinkRetainingCapacity(0);
    }

    fn ensureByIndex(self: *FeatureManager, index: usize) anyerror!void {
        return switch (self.states[index]) {
            .initialized => {},
            .initializing => Error.CyclicDependency,
            .dormant => {
                self.states[index] = .initializing;
                const descriptor = self.descriptors[index];
                for (descriptor.dependencies) |dependency| {
                    try self.ensure(dependency);
                }
                if (descriptor.init) |init_fn| {
                    try init_fn(self.environment);
                }
                self.states[index] = .initialized;
                try self.init_order.append(index);
            },
        };
    }
};

const testing = std.testing;

test "feature manager initializes dependencies once" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();

    var init_sequence = std.ArrayList([]const u8).init(arena.allocator());

    const descriptors = [_]FeatureDescriptor{
        .{
            .name = "core",
            .display_name = "Core",
            .category = .core,
            .init = initTracker("core", &init_sequence),
            .deinit = deinitTracker("core", &init_sequence),
        },
        .{
            .name = "logging",
            .display_name = "Logging",
            .category = .logging,
            .dependencies = &.{"core"},
            .init = initTracker("logging", &init_sequence),
            .deinit = deinitTracker("logging", &init_sequence),
        },
    };

    var manager = try FeatureManager.init(arena.allocator(), .{ .allocator = arena.allocator() }, &descriptors);
    defer manager.deinit();

    try manager.ensure("logging");
    try testing.expectEqualSlices(u8, "core", init_sequence.items[0]);
    try testing.expectEqualSlices(u8, "logging", init_sequence.items[1]);

    manager.shutdown();
    try testing.expectEqual(@as(usize, 0), init_sequence.items.len);
}

fn initTracker(name: []const u8, sequence: *std.ArrayList([]const u8)) InitFn {
    return struct {
        fn init(env: Environment) anyerror!void {
            _ = env;
            try sequence.append(name);
        }
    }.init;
}

fn deinitTracker(name: []const u8, sequence: *std.ArrayList([]const u8)) DeinitFn {
    return struct {
        fn deinit(env: Environment) void {
            _ = env;
            _ = name;
            _ = sequence.popOrNull();
        }
    }.deinit;
}

