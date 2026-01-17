const std = @import("std");
const builtin = @import("builtin");
const posix = std.posix;
const windows = std.os.windows;

var global_timer: ?std.time.Timer = null;
var global_io_backend: ?std.Io.Threaded = null;
var io_once = std.once(initIoBackend);

/// Get current Unix timestamp in seconds.
/// @return Unix timestamp as seconds since epoch
pub fn unixSeconds() i64 {
    return @intCast(@divTrunc(unixEpochNanoseconds(), std.time.ns_per_s));
}

/// Get current Unix timestamp in milliseconds.
/// @return Unix timestamp as milliseconds since epoch
pub fn unixMilliseconds() i64 {
    return @intCast(@divTrunc(unixEpochNanoseconds(), std.time.ns_per_ms));
}

/// Get current time in seconds using monotonic timer.
/// @return Seconds since program start
pub fn nowSeconds() i64 {
    const divisor: i128 = std.time.ns_per_s;
    return @intCast(@divTrunc(nowNanoseconds(), divisor));
}

/// Get current time in milliseconds using monotonic timer.
/// @return Milliseconds since program start
pub fn nowMilliseconds() i64 {
    const divisor: i128 = std.time.ns_per_ms;
    return @intCast(@divTrunc(nowNanoseconds(), divisor));
}

/// Get current time in nanoseconds using monotonic timer.
/// @return Nanoseconds since program start
pub fn nowNanoseconds() i128 {
    const timer = getTimer() orelse return 0;
    return @intCast(timer.read());
}

/// Sleep for specified number of seconds.
/// @param seconds Number of seconds to sleep
pub fn sleepSeconds(seconds: u64) void {
    sleepMs(seconds * 1000);
}

/// Sleep for specified number of milliseconds.
/// @param milliseconds Number of milliseconds to sleep
pub fn sleepMs(milliseconds: u64) void {
    sleepNs(milliseconds * std.time.ns_per_ms);
}

/// Sleep for specified number of nanoseconds.
/// @param nanoseconds Number of nanoseconds to sleep
pub fn sleepNs(nanoseconds: u64) void {
    if (nanoseconds == 0) return;
    const io = getIo() orelse return;
    const duration = std.Io.Clock.Duration{
        .clock = .awake,
        .raw = .fromNanoseconds(@intCast(nanoseconds)),
    };
    std.Io.Clock.Duration.sleep(duration, io) catch {};
}

/// Format a duration in nanoseconds to human-readable string.
/// @param allocator Memory allocator for result string
/// @param duration_ns Duration in nanoseconds to format
/// @return Formatted duration string (e.g., "1.234s", "500ms", "50us", "500ns")
pub fn formatDurationNs(allocator: std.mem.Allocator, duration_ns: u64) ![]u8 {
    if (duration_ns < 1_000) {
        return std.fmt.allocPrint(allocator, "{}ns", .{duration_ns});
    }
    if (duration_ns < 1_000_000) {
        return std.fmt.allocPrint(allocator, "{}us", .{duration_ns / 1_000});
    }
    if (duration_ns < 1_000_000_000) {
        return std.fmt.allocPrint(allocator, "{}ms", .{duration_ns / 1_000_000});
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

fn initIoBackend() void {
    global_io_backend = std.Io.Threaded.init(std.heap.page_allocator, .{ .environ = std.process.Environ.empty });
}

fn getIo() ?std.Io {
    io_once.call();
    if (global_io_backend) |*backend| {
        return backend.io();
    }
    return null;
}

fn unixEpochNanoseconds() i128 {
    if (builtin.os.tag == .windows) {
        const ticks_100ns: i64 = windows.ntdll.RtlGetSystemTimePrecise();
        const unix_epoch_ticks_100ns: i64 = 11644473600 * 10_000_000;
        const unix_ticks_100ns = ticks_100ns - unix_epoch_ticks_100ns;
        return @as(i128, unix_ticks_100ns) * 100;
    }

    const ts = posix.clock_gettime(posix.CLOCK.REALTIME) catch return 0;
    return @as(i128, @intCast(ts.sec)) * std.time.ns_per_s +
        @as(i128, @intCast(ts.nsec));
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
