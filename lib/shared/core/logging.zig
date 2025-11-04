const std = @import("std");
const errors = @import("errors.zig");

/// Log levels for the lightweight core logger.
pub const LogLevel = enum(u8) {
    debug = 0,
    info = 1,
    warn = 2,
    err = 3,
    fatal = 4,

    pub fn toString(self: LogLevel) []const u8 {
        return switch (self) {
            .debug => "DEBUG",
            .info => "INFO",
            .warn => "WARN",
            .err => "ERROR",
            .fatal => "FATAL",
        };
    }
};

/// Lightweight logging system used before the structured logger is configured.
pub const log = struct {
    var allocator: ?std.mem.Allocator = null;
    var current_level: LogLevel = .info;
    var enabled: bool = false;

    /// Initialize logging system.
    pub fn init(alloc: std.mem.Allocator) errors.AbiError!void {
        allocator = alloc;
        enabled = true;
    }

    /// Deinitialize logging system.
    pub fn deinit() void {
        enabled = false;
        allocator = null;
    }

    /// Set log level.
    pub fn setLevel(level: LogLevel) void {
        current_level = level;
    }

    /// Log a debug message.
    pub fn debug(comptime format: []const u8, args: anytype) void {
        logMessage(.debug, format, args);
    }

    /// Log an info message.
    pub fn info(comptime format: []const u8, args: anytype) void {
        logMessage(.info, format, args);
    }

    /// Log a warning message.
    pub fn warn(comptime format: []const u8, args: anytype) void {
        logMessage(.warn, format, args);
    }

    /// Log an error message.
    pub fn err(comptime format: []const u8, args: anytype) void {
        logMessage(.err, format, args);
    }

    /// Log a fatal message.
    pub fn fatal(comptime format: []const u8, args: anytype) void {
        logMessage(.fatal, format, args);
    }

    fn logMessage(level: LogLevel, comptime format: []const u8, args: anytype) void {
        if (!enabled or @intFromEnum(level) < @intFromEnum(current_level)) return;

        const timestamp = std.time.timestamp();
        std.debug.print("[{}] {s}: " ++ format ++ "\n", .{ timestamp, level.toString() } ++ args);
    }
};
