//! Common Patterns Module
//!
//! Provides reusable patterns for initialization, cleanup, and error handling
//! following Zig 0.16 best practices.

const std = @import("std");

/// Standard allocator type alias for consistency
pub const Allocator = std.mem.Allocator;

/// Writer interface for testable I/O
pub const Writer = std.io.AnyWriter;

/// Common initialization pattern for framework components
pub fn InitPattern(comptime T: type) type {
    return struct {
        const Self = @This();
        
        /// Standard initialization function signature
        pub fn init(allocator: Allocator) !T {
            return T{
                .allocator = allocator,
            };
        }
        
        /// Standard initialization with options
        pub fn initWithOptions(allocator: Allocator, options: anytype) !T {
            _ = options;
            return T{
                .allocator = allocator,
            };
        }
    };
}

/// Common cleanup pattern for framework components
pub fn CleanupPattern(comptime T: type) type {
    return struct {
        /// Standard cleanup function signature
        pub fn deinit(self: *T) void {
            if (@hasField(T, "allocator")) {
                // Cleanup any allocated resources
                if (@hasField(T, "data") and @TypeOf(self.data) == []u8) {
                    self.allocator.free(self.data);
                }
            }
        }
    };
}

/// Error handling pattern with context
pub const ErrorContext = struct {
    message: []const u8,
    location: ?std.builtin.SourceLocation = null,
    cause: ?anyerror = null,
    
    pub fn init(message: []const u8) ErrorContext {
        return .{ .message = message };
    }
    
    pub fn withLocation(self: ErrorContext, location: std.builtin.SourceLocation) ErrorContext {
        return .{
            .message = self.message,
            .location = location,
            .cause = self.cause,
        };
    }
    
    pub fn withCause(self: ErrorContext, cause: anyerror) ErrorContext {
        return .{
            .message = self.message,
            .location = self.location,
            .cause = cause,
        };
    }
    
    pub fn format(self: ErrorContext, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        try writer.print("Error: {s}", .{self.message});
        if (self.location) |loc| {
            try writer.print(" at {s}:{d}:{d}", .{ loc.file, loc.line, loc.column });
        }
        if (self.cause) |cause| {
            try writer.print(" (caused by: {s})", .{@errorName(cause)});
        }
    }
};

/// Logging pattern that respects I/O boundaries
pub const Logger = struct {
    writer: Writer,
    level: Level,
    
    pub const Level = enum(u8) {
        debug = 0,
        info = 1,
        warn = 2,
        err = 3,
    };
    
    pub fn init(writer: Writer, level: Level) Logger {
        return .{ .writer = writer, .level = level };
    }
    
    pub fn debug(self: Logger, comptime fmt: []const u8, args: anytype) !void {
        if (@intFromEnum(self.level) <= @intFromEnum(Level.debug)) {
            try self.writer.print("[DEBUG] " ++ fmt ++ "\n", args);
        }
    }
    
    pub fn info(self: Logger, comptime fmt: []const u8, args: anytype) !void {
        if (@intFromEnum(self.level) <= @intFromEnum(Level.info)) {
            try self.writer.print("[INFO] " ++ fmt ++ "\n", args);
        }
    }
    
    pub fn warn(self: Logger, comptime fmt: []const u8, args: anytype) !void {
        if (@intFromEnum(self.level) <= @intFromEnum(Level.warn)) {
            try self.writer.print("[WARN] " ++ fmt ++ "\n", args);
        }
    }
    
    pub fn err(self: Logger, comptime fmt: []const u8, args: anytype) !void {
        if (@intFromEnum(self.level) <= @intFromEnum(Level.err)) {
            try self.writer.print("[ERROR] " ++ fmt ++ "\n", args);
        }
    }
};

/// Resource management pattern with RAII
pub fn ResourceManager(comptime T: type) type {
    return struct {
        const Self = @This();
        
        resource: T,
        allocator: Allocator,
        cleanup_fn: ?*const fn (*T) void = null,
        
        pub fn init(allocator: Allocator, resource: T) Self {
            return .{
                .resource = resource,
                .allocator = allocator,
            };
        }
        
        pub fn initWithCleanup(allocator: Allocator, resource: T, cleanup_fn: *const fn (*T) void) Self {
            return .{
                .resource = resource,
                .allocator = allocator,
                .cleanup_fn = cleanup_fn,
            };
        }
        
        pub fn deinit(self: *Self) void {
            if (self.cleanup_fn) |cleanup| {
                cleanup(&self.resource);
            }
        }
        
        pub fn get(self: *Self) *T {
            return &self.resource;
        }
    };
}

/// Configuration pattern for framework components
pub fn ConfigPattern(comptime T: type) type {
    return struct {
        pub fn default() T {
            return std.mem.zeroes(T);
        }
        
        pub fn validate(config: T) !void {
            _ = config;
            // Override in specific implementations
        }
        
        pub fn merge(base: T, override: T) T {
            _ = base;
            return override;
        }
    };
}

test "InitPattern creates proper initialization function" {
    const TestStruct = struct {
        allocator: Allocator,
        data: []u8 = &[_]u8{},
    };
    
    const Pattern = InitPattern(TestStruct);
    var instance = try Pattern.init(std.testing.allocator);
    try std.testing.expect(instance.allocator.ptr == std.testing.allocator.ptr);
}

test "Logger respects level filtering" {
    var buffer = std.ArrayList(u8).init(std.testing.allocator);
    defer buffer.deinit();
    
    const writer = buffer.writer().any();
    var logger = Logger.init(writer, .warn);
    
    try logger.debug("debug message", .{});
    try logger.info("info message", .{});
    try logger.warn("warn message", .{});
    
    const output = buffer.items;
    try std.testing.expect(std.mem.indexOf(u8, output, "debug") == null);
    try std.testing.expect(std.mem.indexOf(u8, output, "info") == null);
    try std.testing.expect(std.mem.indexOf(u8, output, "warn") != null);
}

test "ErrorContext provides rich error information" {
    var buffer = std.ArrayList(u8).init(std.testing.allocator);
    defer buffer.deinit();
    
    const writer = buffer.writer();
    const ctx = ErrorContext.init("Test error")
        .withLocation(@src())
        .withCause(error.TestError);
    
    try writer.print("{}", .{ctx});
    
    const output = buffer.items;
    try std.testing.expect(std.mem.indexOf(u8, output, "Test error") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "common.zig") != null);
}