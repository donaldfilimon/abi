const std = @import("std");

var global_timer: ?std.time.Timer = null;

pub fn unixSeconds() i64 {
    return std.time.timestamp();
}

pub fn unixMilliseconds() i64 {
    return std.time.milliTimestamp();
}

pub fn nowSeconds() i64 {
    const divisor: i128 = std.time.ns_per_s;
    return @intCast(@divTrunc(nowNanoseconds(), divisor));
}

pub fn nowMilliseconds() i64 {
    const divisor: i128 = std.time.ns_per_ms;
    return @intCast(@divTrunc(nowNanoseconds(), divisor));
}

pub fn nowNanoseconds() i128 {
    const timer = getTimer() orelse return 0;
    return @intCast(timer.read());
}

pub fn sleepSeconds(seconds: u64) void {
    sleepMs(seconds * 1000);
}

pub fn sleepMs(milliseconds: u64) void {
    std.time.sleep(milliseconds * std.time.ns_per_ms);
}

pub fn formatDurationNs(allocator: std.mem.Allocator, duration_ns: u64) ![]u8 {
    if (duration_ns < 1_000) {
        return std.fmt.allocPrint(allocator, "{d}ns", .{duration_ns});
    }
    if (duration_ns < 1_000_000) {
        return std.fmt.allocPrint(allocator, "{d}us", .{duration_ns / 1_000});
    }
    if (duration_ns < 1_000_000_000) {
        return std.fmt.allocPrint(allocator, "{d}ms", .{duration_ns / 1_000_000});
    }
    const seconds = @as(f64, @floatFromInt(duration_ns)) / 1_000_000_000.0;
    return std.fmt.allocPrint(allocator, "{d:.3}s", .{seconds});
}

pub const Stopwatch = struct {
    timer: std.time.Timer,

    pub fn start() !Stopwatch {
        return .{ .timer = try std.time.Timer.start() };
    }

    pub fn reset(self: *Stopwatch) void {
        self.timer.reset();
    }

    pub fn elapsedNs(self: *Stopwatch) u64 {
        return self.timer.read();
    }

    pub fn elapsedUs(self: *Stopwatch) u64 {
        return self.timer.read() / std.time.ns_per_us;
    }

    pub fn elapsedMs(self: *Stopwatch) u64 {
        return self.timer.read() / std.time.ns_per_ms;
    }

    pub fn elapsedSeconds(self: *Stopwatch) f64 {
        return @as(f64, @floatFromInt(self.timer.read())) / @as(f64, std.time.ns_per_s);
    }
};

fn getTimer() ?*std.time.Timer {
    if (global_timer == null) {
        global_timer = std.time.Timer.start() catch null;
    }
    if (global_timer) |*timer| {
        return timer;
    }
    return null;
}

test "stopwatch measures elapsed time" {
    var watch = try Stopwatch.start();
    const before = watch.elapsedNs();
    sleepMs(1);
    const after = watch.elapsedNs();
    try std.testing.expect(after >= before);
}

test "unix time helpers are monotonic enough" {
    const seconds = unixSeconds();
    const millis = unixMilliseconds();
    try std.testing.expect(seconds > 0);
    const millis_as_seconds = millis / 1000;
    try std.testing.expect(millis_as_seconds >= seconds - 1);
    try std.testing.expect(millis_as_seconds <= seconds + 1);
}

test "format duration helper" {
    const allocator = std.testing.allocator;
    const ns_text = try formatDurationNs(allocator, 999);
    defer allocator.free(ns_text);
    try std.testing.expectEqualStrings("999ns", ns_text);

    const us_text = try formatDurationNs(allocator, 1_000);
    defer allocator.free(us_text);
    try std.testing.expectEqualStrings("1us", us_text);

    const ms_text = try formatDurationNs(allocator, 1_000_000);
    defer allocator.free(ms_text);
    try std.testing.expectEqualStrings("1ms", ms_text);
}
