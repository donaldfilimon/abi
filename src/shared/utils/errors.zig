//! Shared error handling utilities and patterns.
//!
//! Provides common error handling patterns, logging helpers, and error
//! propagation utilities used across the ABI framework.

const std = @import("std");

/// Common error categories used across modules
pub const ErrorCategory = enum {
    io,
    memory,
    validation,
    network,
    computation,
    configuration,
};

/// Common errors across the framework
pub const CommonError = std.fs.File.OpenError ||
    std.fs.File.ReadError ||
    std.fs.File.WriteError ||
    std.mem.Allocator.Error ||
    error{ Timeout, PermissionDenied, InvalidInput, InvalidFormat, NotFound };

/// Standardized error context for logging and debugging
pub const ErrorContext = struct {
    category: ErrorCategory,
    operation: []const u8,
    details: ?[]const u8 = null,
    source_location: std.builtin.SourceLocation,

    pub fn init(
        category: ErrorCategory,
        operation: []const u8,
        details: ?[]const u8,
    ) ErrorContext {
        return .{
            .category = category,
            .operation = operation,
            .details = details,
            .source_location = @src(),
        };
    }

    pub fn log(self: ErrorContext, level: std.log.Level, err: CommonError) void {
        const details_msg = if (self.details) |d| d else "no details";
        switch (level) {
            .err => std.log.err("[{t}] {s} failed: {} - {s} (at {s}:{d})", .{
                self.category,
                self.operation,
                err,
                details_msg,
                self.source_location.file,
                self.source_location.line,
            }),
            .warn => std.log.warn("[{t}] {s} warning: {} - {s}", .{
                self.category,
                self.operation,
                err,
                details_msg,
            }),
            .info => std.log.info("[{t}] {s} info: {} - {s}", .{
                self.category,
                self.operation,
                err,
                details_msg,
            }),
            else => {},
        }
    }
};

/// Result type for operations that may succeed or fail with context
pub fn Result(comptime T: type, comptime E: type) type {
    return union(enum) {
        success: T,
        failure: struct {
            err: E,
            context: ErrorContext,
        },

        pub fn initSuccess(value: T) @This() {
            return .{ .success = value };
        }

        pub fn initFailure(err: E, context: ErrorContext) @This() {
            return .{ .failure = .{ .err = err, .context = context } };
        }

        pub fn isSuccess(self: @This()) bool {
            return self == .success;
        }

        pub fn unwrap(self: @This()) !T {
            switch (self) {
                .success => |value| return value,
                .failure => |failure| {
                    failure.context.log(.err, failure.err);
                    return failure.err;
                },
            }
        }

        pub fn unwrapOr(self: @This(), default_value: T) T {
            return switch (self) {
                .success => |value| value,
                .failure => |failure| blk: {
                    failure.context.log(.warn, failure.err);
                    break :blk default_value;
                },
            };
        }
    };
}

/// Resource management helper with automatic cleanup
pub fn ResourceManager(comptime T: type) type {
    return struct {
        resource: ?T = null,
        cleanup_fn: ?*const fn (*T) void = null,

        pub fn init() @This() {
            return .{};
        }

        pub fn set(self: *@This(), resource: T, cleanup: *const fn (*T) void) void {
            if (self.resource != null) {
                if (self.cleanup_fn) |cleanup_fn| {
                    cleanup_fn(self.resource.?);
                }
            }
            self.resource = resource;
            self.cleanup_fn = cleanup;
        }

        pub fn get(self: *@This()) ?T {
            return self.resource;
        }

        pub fn take(self: *@This()) ?T {
            const res = self.resource;
            self.resource = null;
            return res;
        }

        pub fn deinit(self: *@This()) void {
            if (self.resource) |resource| {
                if (self.cleanup_fn) |cleanup_fn| {
                    cleanup_fn(resource);
                }
            }
            self.resource = null;
            self.cleanup_fn = null;
        }
    };
}

/// Common error handling patterns
pub const ErrorPatterns = struct {
    /// Handle allocation errors with context
    pub fn handleAllocError(
        requested_size: usize,
        context: ErrorContext,
    ) std.mem.Allocator.Error {
        context.log(.err, error.OutOfMemory);
        std.log.err("Allocation failed: requested {} bytes from allocator", .{requested_size});
        return error.OutOfMemory;
    }

    /// IO error set for retry logic
    pub const IoErrorSet = std.fs.File.OpenError ||
        std.fs.File.ReadError ||
        std.fs.File.WriteError ||
        error{ Timeout, ConnectionRefused, BrokenPipe, NetworkUnreachable };

    /// Handle IO errors with retry logic
    pub fn handleIoError(
        err: IoErrorSet,
        operation: []const u8,
        max_retries: u8,
    ) IoErrorSet {
        var retries: u8 = 0;
        while (retries < max_retries) : (retries += 1) {
            std.log.warn("IO operation '{s}' failed (attempt {}/{}): {}", .{ operation, retries + 1, max_retries, err });
            std.time.sleep(100 * std.time.ns_per_ms * retries); // Exponential backoff

            // In a real implementation, you'd retry the operation here
            // For now, just log and return the error
            break;
        }
        return err;
    }

    /// Validate input with comprehensive error reporting
    pub fn validateInput(
        input: anytype,
        field_name: []const u8,
        constraints: struct {
            min_len: ?usize = null,
            max_len: ?usize = null,
            not_empty: bool = false,
            not_null: bool = false,
        },
    ) !void {
        const T = @TypeOf(input);

        if (constraints.not_null) {
            if (@typeInfo(T) == .Optional) {
                if (input == null) {
                    std.log.err("Field '{s}' cannot be null", .{field_name});
                    return error.InvalidInput;
                }
            }
        }

        if (@typeInfo(T) == .Pointer and @typeInfo(T).Pointer.size == .Slice) {
            const len = input.len;
            if (constraints.not_empty and len == 0) {
                std.log.err("Field '{s}' cannot be empty", .{field_name});
                return error.InvalidInput;
            }
            if (constraints.min_len) |min| {
                if (len < min) {
                    std.log.err("Field '{s}' length {} is less than minimum {}", .{ field_name, len, min });
                    return error.InvalidInput;
                }
            }
            if (constraints.max_len) |max| {
                if (len > max) {
                    std.log.err("Field '{s}' length {} exceeds maximum {}", .{ field_name, len, max });
                    return error.InvalidInput;
                }
            }
        }
    }
};

test "ErrorContext logging" {
    const context = ErrorContext.init(.io, "file_read", "test file");
    // This would log in real usage
    _ = context;
}

test "Result type operations" {
    const result = Result(u32, CommonError).initSuccess(42);
    try std.testing.expect(result.isSuccess());
    try std.testing.expectEqual(@as(u32, 42), try result.unwrap());
}

test "ResourceManager lifecycle" {
    var manager = ResourceManager([]u8).init();
    defer manager.deinit();

    const test_data = try std.testing.allocator.dupe(u8, "test");
    manager.set(test_data, struct {
        fn cleanup(data: *[]u8) void {
            std.testing.allocator.free(data.*);
        }
    }.cleanup);

    try std.testing.expect(manager.get() != null);
    try std.testing.expectEqualStrings("test", manager.get().?);
}

test "ErrorPatterns validation" {
    // Test string validation
    try ErrorPatterns.validateInput("valid", "test_field", .{ .min_len = 3 });
    try std.testing.expectError(error.InvalidInput, ErrorPatterns.validateInput("", "test_field", .{ .not_empty = true }));
    try std.testing.expectError(error.InvalidInput, ErrorPatterns.validateInput("x", "test_field", .{ .min_len = 3 }));
}
