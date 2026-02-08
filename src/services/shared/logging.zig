//! Simple logging helpers with scoped timing.
const std = @import("std");
const time = @import("../../services/shared/time.zig");
const sync = @import("../../services/shared/sync.zig");

pub const Level = enum {
    trace,
    debug,
    info,
    warn,
    err,
};

var log_mutex: sync.Mutex = .{};

/// Low-level log backend — writes directly to stderr via std.debug.print.
/// This is intentional: std.log would create a circular dependency since
/// this module IS the log implementation layer.
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
    timer: time.Timer,

    pub fn start(label: []const u8) ?ScopedTimer {
        return .{
            .label = label,
            .timer = time.Timer.start() catch return null,
        };
    }

    pub fn stop(self: ScopedTimer) void {
        log_mutex.lock();
        defer log_mutex.unlock();
        const elapsed = self.timer.read();
        std.debug.print("[timer] {s}: {d} ns\n", .{ self.label, elapsed });
    }
};

// ============================================================================
// Tests
// ============================================================================

test "log level ordering" {
    // Verify log levels are defined in increasing severity order
    try std.testing.expectEqual(@intFromEnum(Level.trace), 0);
    try std.testing.expectEqual(@intFromEnum(Level.debug), 1);
    try std.testing.expectEqual(@intFromEnum(Level.info), 2);
    try std.testing.expectEqual(@intFromEnum(Level.warn), 3);
    try std.testing.expectEqual(@intFromEnum(Level.err), 4);
}

test "scoped timer start returns timer" {
    const timer = ScopedTimer.start("test");
    try std.testing.expect(timer != null);
    if (timer) |t| {
        try std.testing.expectEqualStrings("test", t.label);
    }
}
