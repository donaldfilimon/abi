const std = @import("std");

/// High-level phases the runtime travels through during its lifetime.
pub const Stage = enum(u3) {
    cold,
    bootstrapping,
    running,
    shutting_down,
    terminated,

    pub fn next(self: Stage) ?Stage {
        return switch (self) {
            .cold => .bootstrapping,
            .bootstrapping => .running,
            .running => .shutting_down,
            .shutting_down => .terminated,
            .terminated => null,
        };
    }
};

/// Transition metadata provided to lifecycle observers.
pub const Transition = struct {
    from: Stage,
    to: Stage,
};

/// Callback executed when a lifecycle transition occurs.
pub const Observer = struct {
    name: []const u8,
    stages: StageMask = StageMask.all(),
    priority: i32 = 0,
    callback: *const fn (Transition, *Lifecycle, ?*anyopaque) anyerror!void,
    context: ?*anyopaque = null,
};

/// Bit mask describing the stages an observer is interested in.
pub const StageMask = packed struct(u5) {
    cold: bool = false,
    bootstrapping: bool = false,
    running: bool = false,
    shutting_down: bool = false,
    terminated: bool = false,

    pub fn all() StageMask {
        return StageMask{
            .cold = true,
            .bootstrapping = true,
            .running = true,
            .shutting_down = true,
            .terminated = true,
        };
    }

    pub fn contains(self: StageMask, stage: Stage) bool {
        return switch (stage) {
            .cold => self.cold,
            .bootstrapping => self.bootstrapping,
            .running => self.running,
            .shutting_down => self.shutting_down,
            .terminated => self.terminated,
        };
    }
};

pub const Error = error{
    InvalidTransition,
    AlreadyTerminated,
};

/// Lightweight lifecycle coordinator used by the runtime.
pub const Lifecycle = struct {
    allocator: std.mem.Allocator,
    stage: Stage = .cold,
    observers: std.ArrayList(Observer),

    pub const Options = struct {
        reserve_observers: usize = 8,
    };

    pub fn init(allocator: std.mem.Allocator, options: Options) !Lifecycle {
        var observers = std.ArrayList(Observer).init(allocator);
        if (options.reserve_observers > 0) {
            try observers.ensureTotalCapacity(options.reserve_observers);
        }
        return .{
            .allocator = allocator,
            .observers = observers,
        };
    }

    pub fn deinit(self: *Lifecycle) void {
        self.observers.deinit();
        self.* = undefined;
    }

    pub fn currentStage(self: Lifecycle) Stage {
        return self.stage;
    }

    pub fn addObserver(self: *Lifecycle, observer: Observer) !void {
        try self.observers.append(observer);
        std.sort.heap(Observer, self.observers.items, {}, observerLessThan);
    }

    pub fn advance(self: *Lifecycle, to: Stage) Error!void {
        if (self.stage == .terminated) return Error.AlreadyTerminated;
        const expected = self.stage.next() orelse return Error.InvalidTransition;
        if (to != expected) return Error.InvalidTransition;

        const transition = Transition{ .from = self.stage, .to = to };
        for (self.observers.items) |observer| {
            if (!observer.stages.contains(to)) continue;
            observer.callback(transition, self, observer.context) catch |err| {
                std.log.err("lifecycle observer '{s}' failed: {s}", .{ observer.name, @errorName(err) });
            };
        }

        self.stage = to;
    }

    fn observerLessThan(_: void, lhs: Observer, rhs: Observer) bool {
        if (lhs.priority == rhs.priority) return std.mem.lessThan(u8, lhs.name, rhs.name);
        return lhs.priority > rhs.priority;
    }
};

const testing = std.testing;

test "lifecycle enforces forward-only transitions" {
    var lifecycle = try Lifecycle.init(testing.allocator, .{});
    defer lifecycle.deinit();

    try testing.expectEqual(Stage.cold, lifecycle.currentStage());
    try lifecycle.advance(.bootstrapping);
    try testing.expectEqual(Stage.bootstrapping, lifecycle.currentStage());
    try lifecycle.advance(.running);
    try testing.expectEqual(Stage.running, lifecycle.currentStage());
    try lifecycle.advance(.shutting_down);
    try lifecycle.advance(.terminated);
    try testing.expectError(Error.InvalidTransition, lifecycle.advance(.running));
}

test "lifecycle observers receive ordered callbacks" {
    var lifecycle = try Lifecycle.init(testing.allocator, .{ .reserve_observers = 2 });
    defer lifecycle.deinit();

    var calls = std.ArrayList([]const u8).init(testing.allocator);
    defer calls.deinit();

    try lifecycle.addObserver(.{
        .name = "first",
        .priority = 10,
        .callback = struct {
            fn call(transition: Transition, context: *Lifecycle, calls_ctx_opaque: ?*anyopaque) anyerror!void {
                _ = context;
                const calls_ctx = @as(*std.ArrayList([]const u8), @alignCast(calls_ctx_opaque.?));
                try calls_ctx.append(switch (transition.to) {
                    .bootstrapping => "first:boot",
                    .running => "first:run",
                    .shutting_down => "first:down",
                    .terminated => "first:end",
                    .cold => "first:cold",
                });
            }
        }.call,
        .context = &calls,
    });

    try lifecycle.addObserver(.{
        .name = "second",
        .priority = 0,
        .stages = StageMask{ .running = true, .shutting_down = true },
        .callback = struct {
            fn call(transition: Transition, context: *Lifecycle, calls_ctx_opaque: ?*anyopaque) anyerror!void {
                _ = context;
                const calls_ctx = @as(*std.ArrayList([]const u8), @alignCast(calls_ctx_opaque.?));
                try calls_ctx.append(switch (transition.to) {
                    .running => "second:run",
                    .shutting_down => "second:down",
                    else => "second:other",
                });
            }
        }.call,
        .context = &calls,
    });

    try lifecycle.advance(.bootstrapping);
    try lifecycle.advance(.running);
    try lifecycle.advance(.shutting_down);
    try lifecycle.advance(.terminated);

    try testing.expectEqual(@as(usize, 6), calls.items.len);
    try testing.expectEqualSlices(u8, "first:boot", calls.items[0]);
    try testing.expectEqualSlices(u8, "first:run", calls.items[1]);
    try testing.expectEqualSlices(u8, "second:run", calls.items[2]);
    try testing.expectEqualSlices(u8, "first:down", calls.items[3]);
    try testing.expectEqualSlices(u8, "second:down", calls.items[4]);
    try testing.expectEqualSlices(u8, "first:end", calls.items[5]);
}
