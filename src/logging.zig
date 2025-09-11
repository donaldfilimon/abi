//! Structured Logging Module for ABI Framework
//!
//! This module provides structured logging capabilities with:
//! - Multiple log levels (TRACE, DEBUG, INFO, WARN, ERROR, FATAL)
//! - Structured log entries with key-value pairs
//! - Multiple output formats (JSON, text, colored text)
//! - Configurable log sinks (console, file)
//! - Performance-conscious design with minimal allocations
//! - Thread-safe operations

const std = @import("std");
const platform = @import("platform.zig");

/// Log level enumeration
pub const LogLevel = enum(u8) {
    trace = 0,
    debug = 1,
    info = 2,
    warn = 3,
    err = 4,
    fatal = 5,

    pub fn toString(self: LogLevel) []const u8 {
        return switch (self) {
            .trace => "TRACE",
            .debug => "DEBUG",
            .info => "INFO",
            .warn => "WARN",
            .err => "ERROR",
            .fatal => "FATAL",
        };
    }

    pub fn color(self: LogLevel) []const u8 {
        return switch (self) {
            .trace => platform.Colors.cyan,
            .debug => platform.Colors.blue,
            .info => platform.Colors.green,
            .warn => platform.Colors.yellow,
            .err => platform.Colors.red,
            .fatal => platform.Colors.magenta,
        };
    }
};

/// Output format enumeration
pub const OutputFormat = enum {
    json,
    text,
    colored,
};

/// Logger configuration
pub const LoggerConfig = struct {
    level: LogLevel = .info,
    format: OutputFormat = .colored,
    enable_timestamps: bool = true,
    enable_source_info: bool = true,
    buffer_size: usize = 4096,
};

/// Structured Logger
pub const Logger = struct {
    config: LoggerConfig,
    mutex: std.Thread.Mutex,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: LoggerConfig) !*Logger {
        const logger = try allocator.create(Logger);
        logger.* = .{
            .config = config,
            .mutex = std.Thread.Mutex{},
            .allocator = allocator,
        };
        return logger;
    }

    pub fn deinit(self: *Logger) void {
        self.allocator.destroy(self);
    }

    /// Log a message with structured fields
    pub fn log(
        self: *Logger,
        level: LogLevel,
        comptime message: []const u8,
        fields: anytype,
        source_file: []const u8,
        source_line: u32,
        function_name: []const u8,
    ) !void {
        // Check if we should log this level
        if (@intFromEnum(level) < @intFromEnum(self.config.level)) {
            return;
        }

        self.mutex.lock();
        defer self.mutex.unlock();

        // Format the message
        var buffer = std.ArrayListUnmanaged(u8){};
        defer buffer.deinit(self.allocator);

        switch (self.config.format) {
            .json => try self.writeJsonFormat(&buffer, self.allocator, level, message, fields, source_file, source_line, function_name),
            .text => try self.writeTextFormat(&buffer, self.allocator, level, message, fields, source_file, source_line, function_name, false),
            .colored => try self.writeTextFormat(&buffer, self.allocator, level, message, fields, source_file, source_line, function_name, true),
        }

        // Write to stderr using debug print
        std.debug.print("{s}\n", .{buffer.items});
    }

    /// Convenience methods for different log levels
    pub fn trace(self: *Logger, comptime message: []const u8, fields: anytype, src: std.builtin.SourceLocation) !void {
        try self.log(.trace, message, fields, src.file, src.line, src.fn_name);
    }

    pub fn debug(self: *Logger, comptime message: []const u8, fields: anytype, src: std.builtin.SourceLocation) !void {
        try self.log(.debug, message, fields, src.file, src.line, src.fn_name);
    }

    pub fn info(self: *Logger, comptime message: []const u8, fields: anytype, src: std.builtin.SourceLocation) !void {
        try self.log(.info, message, fields, src.file, src.line, src.fn_name);
    }

    pub fn warn(self: *Logger, comptime message: []const u8, fields: anytype, src: std.builtin.SourceLocation) !void {
        try self.log(.warn, message, fields, src.file, src.line, src.fn_name);
    }

    pub fn err(self: *Logger, comptime message: []const u8, fields: anytype, src: std.builtin.SourceLocation) !void {
        try self.log(.err, message, fields, src.file, src.line, src.fn_name);
    }

    pub fn fatal(self: *Logger, comptime message: []const u8, fields: anytype, src: std.builtin.SourceLocation) !void {
        try self.log(.fatal, message, fields, src.file, src.line, src.fn_name);
    }

    fn writeJsonFormat(
        self: *Logger,
        buffer: *std.ArrayListUnmanaged(u8),
        allocator: std.mem.Allocator,
        level: LogLevel,
        message: []const u8,
        fields: anytype,
        source_file: []const u8,
        source_line: u32,
        function_name: []const u8,
    ) !void {
        _ = self; // unused

        try buffer.append(allocator, '{');
        const level_str = try std.fmt.allocPrint(allocator, "\"level\":\"{s}\",", .{level.toString()});
        defer allocator.free(level_str);
        try buffer.appendSlice(allocator, level_str);

        const msg_str = try std.fmt.allocPrint(allocator, "\"message\":\"{s}\",", .{message});
        defer allocator.free(msg_str);
        try buffer.appendSlice(allocator, msg_str);

        const timestamp_str = try std.fmt.allocPrint(allocator, "\"timestamp\":{},", .{std.time.nanoTimestamp()});
        defer allocator.free(timestamp_str);
        try buffer.appendSlice(allocator, timestamp_str);

        const thread_str = try std.fmt.allocPrint(allocator, "\"thread_id\":{},", .{std.Thread.getCurrentId()});
        defer allocator.free(thread_str);
        try buffer.appendSlice(allocator, thread_str);

        // Add fields
        try buffer.appendSlice(allocator, "\"fields\":{");
        inline for (std.meta.fields(@TypeOf(fields)), 0..) |field, i| {
            if (i > 0) try buffer.append(allocator, ',');
            const value = @field(fields, field.name);
            const field_str = try std.fmt.allocPrint(allocator, "\"{s}\":\"{}\"", .{ field.name, value });
            defer allocator.free(field_str);
            try buffer.appendSlice(allocator, field_str);
        }
        try buffer.appendSlice(allocator, "},");

        // Source info
        const source_str = try std.fmt.allocPrint(allocator, "\"source_file\":\"{s}\",\"source_line\":{},\"function_name\":\"{s}\"", .{ source_file, source_line, function_name });
        defer allocator.free(source_str);
        try buffer.appendSlice(allocator, source_str);
        try buffer.append(allocator, '}');
    }

    fn writeTextFormat(
        self: *Logger,
        buffer: *std.ArrayListUnmanaged(u8),
        allocator: std.mem.Allocator,
        level: LogLevel,
        message: []const u8,
        fields: anytype,
        source_file: []const u8,
        source_line: u32,
        function_name: []const u8,
        use_colors: bool,
    ) !void {
        _ = self; // unused
        const platform_info = platform.PlatformInfo.detect();

        // Level with color
        if (platform_info.supports_ansi_colors and use_colors) {
            const level_str = try std.fmt.allocPrint(allocator, "{s}{s}{s} ", .{ level.color(), level.toString(), platform.Colors.reset });
            defer allocator.free(level_str);
            try buffer.appendSlice(allocator, level_str);
        } else {
            const level_str = try std.fmt.allocPrint(allocator, "{s} ", .{level.toString()});
            defer allocator.free(level_str);
            try buffer.appendSlice(allocator, level_str);
        }

        // Message
        try buffer.appendSlice(allocator, message);

        // Fields
        inline for (std.meta.fields(@TypeOf(fields)), 0..) |field, i| {
            if (i == 0) try buffer.appendSlice(allocator, " |");
            const value = @field(fields, field.name);
            const field_str = try std.fmt.allocPrint(allocator, " {s}={}", .{ field.name, value });
            defer allocator.free(field_str);
            try buffer.appendSlice(allocator, field_str);
        }

        // Source info (in debug/trace mode)
        if (level == .debug or level == .trace) {
            const source_str = try std.fmt.allocPrint(allocator, " ({s}:{d} in {s})", .{ source_file, source_line, function_name });
            defer allocator.free(source_str);
            try buffer.appendSlice(allocator, source_str);
        }
    }
};

/// Global logger instance
var global_logger: ?*Logger = null;

/// Initialize global logger
pub fn initGlobalLogger(allocator: std.mem.Allocator, config: LoggerConfig) !void {
    if (global_logger != null) {
        deinitGlobalLogger();
    }
    global_logger = try Logger.init(allocator, config);
}

/// Deinitialize global logger
pub fn deinitGlobalLogger() void {
    if (global_logger) |logger| {
        global_logger = null;
        logger.deinit();
    }
}

/// Get global logger instance
pub fn getGlobalLogger() ?*Logger {
    return global_logger;
}

/// Global logging functions
pub fn log(
    level: LogLevel,
    comptime message: []const u8,
    fields: anytype,
    src: std.builtin.SourceLocation,
) !void {
    if (global_logger) |logger| {
        try logger.log(level, message, fields, src.file, src.line, src.fn_name);
    }
}

pub fn trace(comptime message: []const u8, fields: anytype, src: std.builtin.SourceLocation) !void {
    try log(.trace, message, fields, src);
}

pub fn debug(comptime message: []const u8, fields: anytype, src: std.builtin.SourceLocation) !void {
    try log(.debug, message, fields, src);
}

pub fn info(comptime message: []const u8, fields: anytype, src: std.builtin.SourceLocation) !void {
    try log(.info, message, fields, src);
}

pub fn warn(comptime message: []const u8, fields: anytype, src: std.builtin.SourceLocation) !void {
    try log(.warn, message, fields, src);
}

pub fn err(comptime message: []const u8, fields: anytype, src: std.builtin.SourceLocation) !void {
    try log(.err, message, fields, src);
}

pub fn fatal(comptime message: []const u8, fields: anytype, src: std.builtin.SourceLocation) !void {
    try log(.fatal, message, fields, src);
}

test "LogLevel functionality" {
    const testing = std.testing;

    try testing.expectEqualStrings("TRACE", LogLevel.trace.toString());
    try testing.expectEqualStrings("DEBUG", LogLevel.debug.toString());
    try testing.expectEqualStrings("INFO", LogLevel.info.toString());
    try testing.expectEqualStrings("WARN", LogLevel.warn.toString());
    try testing.expectEqualStrings("ERROR", LogLevel.err.toString());
    try testing.expectEqualStrings("FATAL", LogLevel.fatal.toString());

    // Test level ordering
    try testing.expect(@intFromEnum(LogLevel.trace) < @intFromEnum(LogLevel.debug));
    try testing.expect(@intFromEnum(LogLevel.debug) < @intFromEnum(LogLevel.info));
    try testing.expect(@intFromEnum(LogLevel.info) < @intFromEnum(LogLevel.warn));
    try testing.expect(@intFromEnum(LogLevel.warn) < @intFromEnum(LogLevel.err));
    try testing.expect(@intFromEnum(LogLevel.err) < @intFromEnum(LogLevel.fatal));
}

test "Logger basic functionality" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const config = LoggerConfig{
        .level = .debug,
        .format = .text,
    };

    const logger = try Logger.init(allocator, config);
    defer logger.deinit();

    // Test configuration
    try testing.expectEqual(LogLevel.debug, logger.config.level);
    try testing.expectEqual(OutputFormat.text, logger.config.format);
}

test "Global logger functionality" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const config = LoggerConfig{
        .level = .info,
        .format = .json,
    };

    try initGlobalLogger(allocator, config);
    defer deinitGlobalLogger();

    const global = getGlobalLogger();
    try testing.expect(global != null);
    try testing.expectEqual(LogLevel.info, global.?.config.level);
    try testing.expectEqual(OutputFormat.json, global.?.config.format);
}
