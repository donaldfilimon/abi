const std = @import("std");
const time = @import("time.zig");

pub const Level = enum(u8) {
    debug,
    info,
    warn,
    err,
};

pub const LogEntry = struct {
    level: Level,
    module: []const u8,
    message: []const u8,
    timestamp: i64,
};

pub const Logger = struct {
    allocator: std.mem.Allocator,
    level: Level,
    entries: std.ArrayListUnmanaged(LogEntry),

    pub fn init(allocator: std.mem.Allocator, min_level: Level) Logger {
        return .{
            .allocator = allocator,
            .level = min_level,
            .entries = std.ArrayListUnmanaged(LogEntry).empty,
        };
    }

    pub fn deinit(self: *Logger) void {
        for (self.entries.items) |*entry| {
            self.allocator.free(entry.module);
            self.allocator.free(entry.message);
        }
        self.entries.deinit(self.allocator);
    }

    pub fn log(self: *Logger, level: Level, module: []const u8, comptime fmt: []const u8, args: anytype) void {
        if (@backingInt(level) < @backingInt(self.level)) return;

        const module_copy = self.allocator.dupe(u8, module) catch return;
        errdefer self.allocator.free(module_copy);

        const message_buf = std.fmt.allocPrint(self.allocator, fmt, args) catch return;
        errdefer self.allocator.free(message_buf);

        const entry = LogEntry{
            .level = level,
            .module = module_copy,
            .message = message_buf,
            .timestamp = time.unixMs(),
        };

        self.entries.append(self.allocator, entry) catch |append_err| {
            self.allocator.free(module_copy);
            self.allocator.free(message_buf);
            std.log.err("Logger: failed to append entry: {}", .{append_err});
            return;
        };

        const level_str = switch (level) {
            .debug => "DEBUG",
            .info => "INFO",
            .warn => "WARN",
            .err => "ERROR",
        };

        std.log.warn("[{s}] [{s}] {s}", .{ level_str, module, message_buf });
    }

    pub fn debug(self: *Logger, module: []const u8, comptime fmt: []const u8, args: anytype) void {
        self.log(.debug, module, fmt, args);
    }

    pub fn info(self: *Logger, module: []const u8, comptime fmt: []const u8, args: anytype) void {
        self.log(.info, module, fmt, args);
    }

    pub fn warn(self: *Logger, module: []const u8, comptime fmt: []const u8, args: anytype) void {
        self.log(.warn, module, fmt, args);
    }

    pub fn err(self: *Logger, module: []const u8, comptime fmt: []const u8, args: anytype) void {
        self.log(.err, module, fmt, args);
    }

    pub fn getEntries(self: *const Logger) []const LogEntry {
        return self.entries.items;
    }

    pub fn clear(self: *Logger) void {
        for (self.entries.items) |*entry| {
            self.allocator.free(entry.module);
            self.allocator.free(entry.message);
        }
        self.entries.clearRetainingCapacity();
    }
};

test {
    std.testing.refAllDecls(@This());
}

test "Logger init and deinit" {
    var logger = Logger.init(std.testing.allocator, .debug);
    defer logger.deinit();

    try std.testing.expectEqual(.debug, logger.level);
    try std.testing.expectEqual(@as(usize, 0), logger.entries.items.len);
}

test "Logger log entries" {
    var logger = Logger.init(std.testing.allocator, .debug);
    defer logger.deinit();

    logger.info("test", "hello {s}", .{"world"});
    try std.testing.expectEqual(@as(usize, 1), logger.entries.items.len);
    try std.testing.expectEqual(.info, logger.entries.items[0].level);
}

test "Logger level filtering" {
    var logger = Logger.init(std.testing.allocator, .warn);
    defer logger.deinit();

    logger.debug("test", "should not appear", .{});
    logger.info("test", "should not appear", .{});
    try std.testing.expectEqual(@as(usize, 0), logger.entries.items.len);

    logger.warn("test", "should appear", .{});
    try std.testing.expectEqual(@as(usize, 1), logger.entries.items.len);
}

test "Logger clear" {
    var logger = Logger.init(std.testing.allocator, .debug);
    defer logger.deinit();

    logger.info("test", "entry 1", .{});
    logger.info("test", "entry 2", .{});
    try std.testing.expectEqual(@as(usize, 2), logger.entries.items.len);

    logger.clear();
    try std.testing.expectEqual(@as(usize, 0), logger.entries.items.len);
}
