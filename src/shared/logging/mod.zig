//! Simple logging helpers with scoped timing.
const std = @import("std");

pub const Level = enum {
    trace,
    debug,
    info,
    warn,
    err,
};

var log_mutex: std.Thread.Mutex = .{};

pub fn log(level: Level, comptime fmt: []const u8, args: anytype) void {
    log_mutex.lock();
    defer log_mutex.unlock();
    std.debug.print("[{t}] ", .{level});
    std.debug.print(fmt, args);
    std.debug.print("\n", .{});
}

pub fn trace(comptime fmt: []const u8, args: anytype) void {
    log(.trace, fmt, args);
}

pub fn debug(comptime fmt: []const u8, args: anytype) void {
    log(.debug, fmt, args);
}

pub fn info(comptime fmt: []const u8, args: anytype) void {
    log(.info, fmt, args);
}

pub fn warn(comptime fmt: []const u8, args: anytype) void {
    log(.warn, fmt, args);
}

pub fn err(comptime fmt: []const u8, args: anytype) void {
    log(.err, fmt, args);
}

pub const ScopedTimer = struct {
    label: []const u8,
    timer: std.time.Timer,

    pub fn start(label: []const u8) ?ScopedTimer {
        return .{
            .label = label,
            .timer = std.time.Timer.start() catch return null,
        };
    }

    pub fn stop(self: ScopedTimer) void {
        log_mutex.lock();
        defer log_mutex.unlock();
        const elapsed = self.timer.read();
        std.debug.print("[timer] {s}: {d} ns\n", .{ self.label, elapsed });
    }
};
