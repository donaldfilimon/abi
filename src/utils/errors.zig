//! Error Handling Utilities
//!
//! This module provides consistent error handling patterns for the WDBX database.

const std = @import("std");

/// Comprehensive error set for WDBX
pub const ErrorSet = error{
    // Database errors
    InvalidFileFormat,
    CorruptedData,
    InvalidDimensions,
    IndexOutOfBounds,
    DatabaseNotInitialized,
    DimensionMismatch,
    
    // Storage errors
    InsufficientMemory,
    FileSystemError,
    ReadOnly,
    FileNotOpen,
    HeaderNotInitialized,
    
    // Index errors
    InvalidIndex,
    IndexCorrupted,
    MetricMismatch,
    
    // Network errors
    ConnectionFailed,
    ConnectionResetByPeer,
    BrokenPipe,
    Timeout,
    InvalidProtocol,
    
    // API errors
    InvalidRequest,
    Unauthorized,
    RateLimitExceeded,
    InvalidToken,
    
    // General errors
    NotImplemented,
    InvalidOperation,
    InvalidArgument,
    OutOfMemory,
    Unexpected,
};

/// Error context for detailed error reporting
pub const ErrorContext = struct {
    error_type: ErrorSet,
    message: []const u8,
    file: []const u8,
    line: u32,
    timestamp: i64,
    
    pub fn format(self: ErrorContext, writer: anytype) !void {
        try writer.print("[{d}] {s} at {s}:{d}: {s}\n", .{
            self.timestamp,
            @errorName(self.error_type),
            self.file,
            self.line,
            self.message,
        });
    }
};

/// Error handler for consistent error reporting
pub const ErrorHandler = struct {
    allocator: std.mem.Allocator,
    contexts: std.ArrayList(ErrorContext),
    max_contexts: usize,
    
    const Self = @This();
    
    pub fn init(allocator: std.mem.Allocator, max_contexts: usize) Self {
        return .{
            .allocator = allocator,
            .contexts = std.ArrayList(ErrorContext).init(allocator),
            .max_contexts = max_contexts,
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.contexts.deinit();
    }
    
    pub fn recordError(
        self: *Self,
        err: ErrorSet,
        message: []const u8,
        file: []const u8,
        line: u32,
    ) !void {
        const context = ErrorContext{
            .error_type = err,
            .message = message,
            .file = file,
            .line = line,
            .timestamp = std.time.milliTimestamp(),
        };
        
        try self.contexts.append(context);
        
        // Limit stored contexts
        if (self.contexts.items.len > self.max_contexts) {
            _ = self.contexts.orderedRemove(0);
        }
    }
    
    pub fn getLastError(self: *Self) ?ErrorContext {
        if (self.contexts.items.len == 0) return null;
        return self.contexts.items[self.contexts.items.len - 1];
    }
    
    pub fn clearErrors(self: *Self) void {
        self.contexts.clearRetainingCapacity();
    }
    
    pub fn printErrors(self: *Self, writer: anytype) !void {
        for (self.contexts.items) |context| {
            try context.format(writer);
        }
    }
};

/// Result type for error handling
pub fn Result(comptime T: type) type {
    return union(enum) {
        ok: T,
        err: ErrorContext,
        
        pub fn isOk(self: @This()) bool {
            return self == .ok;
        }
        
        pub fn isErr(self: @This()) bool {
            return self == .err;
        }
        
        pub fn unwrap(self: @This()) T {
            return switch (self) {
                .ok => |value| value,
                .err => |context| {
                    std.debug.print("Error: ", .{});
                    context.format(std.io.getStdErr().writer()) catch {};
                    @panic("Called unwrap on error result");
                },
            };
        }
        
        pub fn unwrapOr(self: @This(), default: T) T {
            return switch (self) {
                .ok => |value| value,
                .err => default,
            };
        }
    };
}

/// Macro-like function for recording errors with context
pub fn recordError(
    handler: *ErrorHandler,
    err: ErrorSet,
    comptime message: []const u8,
) !void {
    const src = @src();
    try handler.recordError(err, message, src.file, src.line);
}

/// Retry mechanism for transient errors
pub fn retry(
    comptime T: type,
    comptime func: fn () anyerror!T,
    max_attempts: u32,
    delay_ms: u64,
) !T {
    var attempt: u32 = 0;
    while (attempt < max_attempts) : (attempt += 1) {
        if (func()) |result| {
            return result;
        } else |err| {
            if (attempt + 1 < max_attempts) {
                // Check if error is retryable
                switch (err) {
                    error.ConnectionFailed,
                    error.ConnectionResetByPeer,
                    error.Timeout,
                    error.BrokenPipe,
                    => {
                        std.time.sleep(delay_ms * std.time.ns_per_ms);
                        continue;
                    },
                    else => return err,
                }
            }
            return err;
        }
    }
    unreachable;
}

// Tests
test "ErrorHandler basic operations" {
    const testing = std.testing;
    
    var handler = ErrorHandler.init(testing.allocator, 10);
    defer handler.deinit();
    
    try handler.recordError(error.InvalidDimensions, "Test error", "test.zig", 100);
    
    const last_error = handler.getLastError();
    try testing.expect(last_error != null);
    try testing.expectEqual(error.InvalidDimensions, last_error.?.error_type);
}

test "Result type" {
    const testing = std.testing;
    
    const ok_result = Result(i32){ .ok = 42 };
    try testing.expect(ok_result.isOk());
    try testing.expectEqual(@as(i32, 42), ok_result.unwrap());
    
    const err_result = Result(i32){ .err = .{
        .error_type = error.InvalidArgument,
        .message = "Test error",
        .file = "test.zig",
        .line = 100,
        .timestamp = 0,
    }};
    try testing.expect(err_result.isErr());
    try testing.expectEqual(@as(i32, 0), err_result.unwrapOr(0));
}

test "Retry mechanism" {
    const testing = std.testing;
    
    var counter: u32 = 0;
    const flaky_func = struct {
        fn call(c: *u32) !i32 {
            c.* += 1;
            if (c.* < 3) {
                return error.ConnectionFailed;
            }
            return 42;
        }
    };
    
    const result = try retry(i32, struct {
        fn wrapper() !i32 {
            return flaky_func.call(&counter);
        }
    }.wrapper, 5, 10);
    
    try testing.expectEqual(@as(i32, 42), result);
    try testing.expectEqual(@as(u32, 3), counter);
}