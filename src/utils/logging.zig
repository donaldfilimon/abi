//! Logging utilities for WDBX-AI utils
//!
//! Provides logging functionality for the utils module.

const std = @import("std");
const core = @import("../core/mod.zig");

// Re-export core logging functionality
pub const LogLevel = core.log.LogLevel;
pub const Logger = struct {
    level: LogLevel,
    
    pub fn init(level: LogLevel) Logger {
        return Logger{ .level = level };
    }
    
    pub fn log(self: Logger, level: LogLevel, comptime fmt: []const u8, args: anytype) void {
        if (@intFromEnum(level) >= @intFromEnum(self.level)) {
            core.log.logMessage(level, fmt, args);
        }
    }
    
    pub fn debug(self: Logger, comptime fmt: []const u8, args: anytype) void {
        self.log(.debug, fmt, args);
    }
    
    pub fn info(self: Logger, comptime fmt: []const u8, args: anytype) void {
        self.log(.info, fmt, args);
    }
    
    pub fn warn(self: Logger, comptime fmt: []const u8, args: anytype) void {
        self.log(.warn, fmt, args);
    }
    
    pub fn err(self: Logger, comptime fmt: []const u8, args: anytype) void {
        self.log(.err, fmt, args);
    }
};

test "utils logging" {
    const testing = std.testing;
    
    const logger = Logger.init(.info);
    logger.info("Test message", .{});
}