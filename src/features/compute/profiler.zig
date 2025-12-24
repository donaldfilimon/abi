const std = @import("std");

pub const Sample = struct {
    label: []const u8,
    elapsed_ns: u64,
};

pub const Profiler = struct {
    allocator: std.mem.Allocator,
    timer: std.time.Timer,
    samples: std.ArrayList(Sample),

    pub fn init(allocator: std.mem.Allocator) !Profiler {
        return .{
            .allocator = allocator,
            .timer = std.time.Timer.start() catch unreachable,
            .samples = try std.ArrayList(Sample).initCapacity(allocator, 0),
        };
    }

    pub fn deinit(self: *Profiler) void {
        self.samples.deinit(self.allocator);
    }

    pub fn reset(self: *Profiler) void {
        self.timer.reset();
    }

    pub fn record(self: *Profiler, label: []const u8) !void {
        const elapsed = self.timer.read();
        try self.samples.append(self.allocator, .{ .label = label, .elapsed_ns = elapsed });
    }

    pub fn clear(self: *Profiler) void {
        self.samples.clearRetainingCapacity();
    }
};

test "profiler records samples" {
    var profiler = try Profiler.init(std.testing.allocator);
    defer profiler.deinit();

    profiler.reset();
    try profiler.record("start");
    try std.testing.expect(profiler.samples.items.len == 1);
}
