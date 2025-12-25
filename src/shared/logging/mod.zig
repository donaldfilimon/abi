const std = @import("std");

pub const Level = enum {
    trace,
    debug,
    info,
    warn,
    err,
};

pub fn log(level: Level, comptime fmt: []const u8, args: anytype) void {
    std.debug.print("[{s}] ", .{@tagName(level)});
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
    start_ns: i128,

    pub fn start(label: []const u8) ScopedTimer {
        return .{
            .label = label,
            .start_ns = std.time.nanoTimestamp(),
        };
    }

    pub fn stop(self: ScopedTimer) void {
        const elapsed = std.time.nanoTimestamp() - self.start_ns;
        std.debug.print("[timer] {s}: {d} ns\n", .{ self.label, elapsed });
    }
};
