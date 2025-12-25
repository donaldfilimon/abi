const std = @import("std");

var global_timer: ?std.time.Timer = null;

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

pub fn sleepMs(milliseconds: u64) void {
    std.time.sleep(milliseconds * std.time.ns_per_ms);
}

fn getTimer() ?*std.time.Timer {
    if (global_timer == null) {
        global_timer = std.time.Timer.start() catch null;
    }
    if (global_timer) |*timer| {
        return timer;
    }
    return null;
}
