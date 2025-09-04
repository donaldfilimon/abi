//! Unified error handling for WDBX-AI
//!
//! This module provides consistent error types and handling patterns
//! used throughout the codebase.

const std = @import("std");
const core = @import("core/mod.zig");

/// Main error set for WDBX-AI
pub const WdbxError = error{
    // Database errors
    DatabaseNotFound,
    DatabaseCorrupted,
    DatabaseLocked,
    TransactionFailed,
    RecordNotFound,
    DuplicateRecord,
    InvalidQuery,
    
    // Vector errors
    DimensionMismatch,
    InvalidVector,
    VectorTooLarge,
    
    // Index errors
    IndexNotFound,
    IndexCorrupted,
    IndexBuildFailed,
    
    // I/O errors
    FileNotFound,
    PermissionDenied,
    DiskFull,
    IoError,
    
    // Network errors
    ConnectionFailed,
    ConnectionTimeout,
    NetworkError,
    
    // Configuration errors
    InvalidConfig,
    MissingConfig,
    
    // Resource errors
    OutOfMemory,
    TooManyOpenFiles,
    ResourceExhausted,
    
    // System errors
    NotImplemented,
    InternalError,
    SystemError,
    
    // AI/ML errors
    ModelNotFound,
    ModelLoadFailed,
    InferenceFailed,
    TrainingFailed,
};

/// Extended error information with context
pub const ErrorInfo = struct {
    error_type: WdbxError,
    message: []const u8,
    context: ?[]const u8 = null,
    source_location: ?std.builtin.SourceLocation = null,
    timestamp: i64,
    
    pub fn init(error_type: WdbxError, message: []const u8) ErrorInfo {
        return .{
            .error_type = error_type,
            .message = message,
            .timestamp = std.time.timestamp(),
        };
    }
    
    pub fn withContext(self: ErrorInfo, context: []const u8) ErrorInfo {
        var new = self;
        new.context = context;
        return new;
    }
    
    pub fn withSource(self: ErrorInfo, src: std.builtin.SourceLocation) ErrorInfo {
        var new = self;
        new.source_location = src;
        return new;
    }
    
    pub fn format(self: ErrorInfo, writer: anytype) !void {
        try writer.print("[{s}] {s}", .{ @errorName(self.error_type), self.message });
        
        if (self.context) |ctx| {
            try writer.print(" - Context: {s}", .{ctx});
        }
        
        if (self.source_location) |loc| {
            try writer.print(" at {s}:{d}:{d}", .{ loc.file, loc.line, loc.column });
        }
    }
};

/// Result type for operations that can fail with detailed error info
pub fn ResultWithInfo(comptime T: type) type {
    return union(enum) {
        ok: T,
        err: ErrorInfo,
        
        pub fn isOk(self: @This()) bool {
            return self == .ok;
        }
        
        pub fn isErr(self: @This()) bool {
            return self == .err;
        }
        
        pub fn unwrap(self: @This()) T {
            return switch (self) {
                .ok => |value| value,
                .err => |info| {
                    std.log.err("Unwrap called on error: {}", .{info});
                    unreachable;
                },
            };
        }
        
        pub fn unwrapOr(self: @This(), default: T) T {
            return switch (self) {
                .ok => |value| value,
                .err => default,
            };
        }
        
        pub fn mapErr(self: @This(), comptime f: fn (ErrorInfo) ErrorInfo) @This() {
            return switch (self) {
                .ok => self,
                .err => |info| .{ .err = f(info) },
            };
        }
    };
}

/// Error handler for graceful error recovery
pub const ErrorHandler = struct {
    allocator: std.mem.Allocator,
    handlers: std.AutoHashMap(WdbxError, *const fn (ErrorInfo) void),
    default_handler: ?*const fn (ErrorInfo) void,
    
    pub fn init(allocator: std.mem.Allocator) ErrorHandler {
        return .{
            .allocator = allocator,
            .handlers = std.AutoHashMap(WdbxError, *const fn (ErrorInfo) void).init(allocator),
            .default_handler = null,
        };
    }
    
    pub fn deinit(self: *ErrorHandler) void {
        self.handlers.deinit();
    }
    
    pub fn register(self: *ErrorHandler, error_type: WdbxError, handler: *const fn (ErrorInfo) void) !void {
        try self.handlers.put(error_type, handler);
    }
    
    pub fn setDefault(self: *ErrorHandler, handler: *const fn (ErrorInfo) void) void {
        self.default_handler = handler;
    }
    
    pub fn handle(self: *ErrorHandler, info: ErrorInfo) void {
        if (self.handlers.get(info.error_type)) |handler| {
            handler(info);
        } else if (self.default_handler) |handler| {
            handler(info);
        } else {
            // Default behavior: log the error
            std.log.err("{}", .{info});
        }
    }
};

/// Panic handler with better formatting
pub fn panicHandler(msg: []const u8, error_return_trace: ?*std.builtin.StackTrace, ret_addr: ?usize) noreturn {
    _ = error_return_trace;
    
    const stderr = std.io.getStdErr().writer();
    
    stderr.print("\n", .{}) catch {};
    stderr.print("╔══════════════════════════════════════════════════════════════╗\n", .{}) catch {};
    stderr.print("║                      WDBX-AI PANIC                           ║\n", .{}) catch {};
    stderr.print("╚══════════════════════════════════════════════════════════════╝\n", .{}) catch {};
    stderr.print("\n", .{}) catch {};
    stderr.print("Message: {s}\n", .{msg}) catch {};
    stderr.print("\n", .{}) catch {};
    
    // Print stack trace
    stderr.print("Stack trace:\n", .{}) catch {};
    var it = std.debug.StackIterator.init(ret_addr orelse @returnAddress(), null);
    var i: usize = 0;
    while (it.next()) |addr| : (i += 1) {
        if (i > 0) { // Skip the panic handler itself
            stderr.print("  [{d}] 0x{x}\n", .{ i - 1, addr }) catch {};
        }
    }
    
    std.process.exit(1);
}

/// Error recovery strategies
pub const RecoveryStrategy = enum {
    retry,
    retry_with_backoff,
    fallback,
    ignore,
    propagate,
    
    pub fn execute(
        self: RecoveryStrategy,
        comptime T: type,
        operation: anytype,
        options: RecoveryOptions,
    ) !T {
        return switch (self) {
            .retry => try executeWithRetry(T, operation, options.max_retries orelse 3, 0),
            .retry_with_backoff => try executeWithBackoff(T, operation, options),
            .fallback => operation() catch |_| options.fallback_value.?,
            .ignore => operation() catch |_| return error.Ignored,
            .propagate => try operation(),
        };
    }
};

pub const RecoveryOptions = struct {
    max_retries: ?u32 = 3,
    initial_delay_ms: ?u64 = 100,
    max_delay_ms: ?u64 = 10000,
    backoff_factor: ?f32 = 2.0,
    fallback_value: ?anytype = null,
};

fn executeWithRetry(comptime T: type, operation: anytype, max_retries: u32, attempt: u32) !T {
    return operation() catch |err| {
        if (attempt >= max_retries) return err;
        return try executeWithRetry(T, operation, max_retries, attempt + 1);
    };
}

fn executeWithBackoff(comptime T: type, operation: anytype, options: RecoveryOptions) !T {
    var delay_ms = options.initial_delay_ms orelse 100;
    var attempt: u32 = 0;
    
    while (attempt < (options.max_retries orelse 3)) : (attempt += 1) {
        if (operation()) |result| {
            return result;
        } else |_| {
            if (attempt < (options.max_retries orelse 3) - 1) {
                std.time.sleep(delay_ms * std.time.ns_per_ms);
                delay_ms = @min(
                    @as(u64, @intFromFloat(@as(f32, @floatFromInt(delay_ms)) * (options.backoff_factor orelse 2.0))),
                    options.max_delay_ms orelse 10000
                );
            }
        }
    }
    
    return try operation(); // Final attempt that will propagate the error
}

/// Error aggregator for collecting multiple errors
pub const ErrorAggregator = struct {
    allocator: std.mem.Allocator,
    errors: std.ArrayList(ErrorInfo),
    
    pub fn init(allocator: std.mem.Allocator) ErrorAggregator {
        return .{
            .allocator = allocator,
            .errors = std.ArrayList(ErrorInfo).init(allocator),
        };
    }
    
    pub fn deinit(self: *ErrorAggregator) void {
        self.errors.deinit();
    }
    
    pub fn add(self: *ErrorAggregator, info: ErrorInfo) !void {
        try self.errors.append(info);
    }
    
    pub fn addError(self: *ErrorAggregator, err: WdbxError, message: []const u8) !void {
        try self.add(ErrorInfo.init(err, message));
    }
    
    pub fn hasErrors(self: *ErrorAggregator) bool {
        return self.errors.items.len > 0;
    }
    
    pub fn count(self: *ErrorAggregator) usize {
        return self.errors.items.len;
    }
    
    pub fn format(self: *ErrorAggregator, writer: anytype) !void {
        if (self.errors.items.len == 0) {
            try writer.writeAll("No errors");
            return;
        }
        
        try writer.print("Error Summary ({d} errors):\n", .{self.errors.items.len});
        for (self.errors.items, 0..) |err, i| {
            try writer.print("  [{d}] ", .{i + 1});
            try err.format(writer);
            try writer.writeByte('\n');
        }
    }
    
    pub fn clear(self: *ErrorAggregator) void {
        self.errors.clearRetainingCapacity();
    }
};

/// Assertion with custom error type
pub fn assertWithError(condition: bool, err: WdbxError, comptime fmt: []const u8, args: anytype) !void {
    if (!condition) {
        const message = try std.fmt.allocPrint(std.heap.page_allocator, fmt, args);
        defer std.heap.page_allocator.free(message);
        
        const info = ErrorInfo.init(err, message).withSource(@src());
        std.log.err("{}", .{info});
        return err;
    }
}

/// Ensure a value meets a condition with custom error
pub fn ensureWithError(
    comptime T: type,
    value: T,
    condition: fn (T) bool,
    err: WdbxError,
    comptime fmt: []const u8,
    args: anytype,
) !T {
    if (!condition(value)) {
        const message = try std.fmt.allocPrint(std.heap.page_allocator, fmt, args);
        defer std.heap.page_allocator.free(message);
        
        const info = ErrorInfo.init(err, message).withSource(@src());
        std.log.err("{}", .{info});
        return err;
    }
    return value;
}

test "error handling" {
    const testing = std.testing;
    
    // Test ErrorInfo
    const info = ErrorInfo.init(WdbxError.DatabaseNotFound, "Test database not found")
        .withContext("during test execution")
        .withSource(@src());
    
    var buf: [256]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);
    try info.format(stream.writer());
    
    try testing.expect(std.mem.indexOf(u8, stream.getWritten(), "DatabaseNotFound") != null);
    try testing.expect(std.mem.indexOf(u8, stream.getWritten(), "Test database not found") != null);
}

test "error recovery" {
    const testing = std.testing;
    
    var attempt_count: u32 = 0;
    const operation = struct {
        fn fail(count: *u32) !i32 {
            count.* += 1;
            if (count.* < 3) {
                return error.TemporaryFailure;
            }
            return 42;
        }
    }.fail;
    
    const result = try RecoveryStrategy.retry.execute(
        i32,
        struct {
            fn call() !i32 {
                return operation(&attempt_count);
            }
        }.call,
        .{ .max_retries = 5 },
    );
    
    try testing.expectEqual(@as(i32, 42), result);
    try testing.expectEqual(@as(u32, 3), attempt_count);
}

test "error aggregator" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    var aggregator = ErrorAggregator.init(allocator);
    defer aggregator.deinit();
    
    try aggregator.addError(WdbxError.FileNotFound, "config.toml not found");
    try aggregator.addError(WdbxError.InvalidConfig, "missing required field");
    
    try testing.expectEqual(@as(usize, 2), aggregator.count());
    try testing.expect(aggregator.hasErrors());
}