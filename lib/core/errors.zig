//! Core Errors Module
//!
//! Error definitions and handling utilities for the framework

const std = @import("std");
const types = @import("types.zig");

/// Framework error set
pub const Error = error{
    /// Invalid configuration
    InvalidConfig,
    /// Invalid parameter
    InvalidParameter,
    /// Resource not found
    NotFound,
    /// Resource already exists
    AlreadyExists,
    /// Operation timeout
    Timeout,
    /// Rate limit exceeded
    RateLimited,
    /// Insufficient permissions
    PermissionDenied,
    /// Service unavailable
    Unavailable,
    /// Internal error
    InternalError,
    /// Network error
    NetworkError,
    /// Serialization error
    SerializationError,
    /// Deserialization error
    DeserializationError,
    /// Validation error
    ValidationError,
    /// Empty value where non-empty expected
    Empty,
    /// Out of memory
    OutOfMemory,
    /// Feature not implemented
    NotImplemented,
    /// Feature disabled
    FeatureDisabled,
};

/// Error context for providing additional information
pub const ErrorContext = struct {
    /// The error code
    code: types.ErrorCode,
    /// Human-readable message
    message: []const u8,
    /// Optional source location
    source: ?[]const u8 = null,
    /// Optional additional data
    data: ?[]const u8 = null,

    /// Creates a new error context
    pub fn init(code: types.ErrorCode, message: []const u8) ErrorContext {
        return ErrorContext{
            .code = code,
            .message = message,
        };
    }

    /// Creates an error context with source location
    pub fn initWithSource(code: types.ErrorCode, message: []const u8, source: []const u8) ErrorContext {
        return ErrorContext{
            .code = code,
            .message = message,
            .source = source,
        };
    }

    /// Formats the error context as a string
    pub fn format(self: ErrorContext, allocator: std.mem.Allocator) ![]u8 {
        if (self.source) |src| {
            return std.fmt.allocPrint(allocator, "[{s}] {s}: {s}", .{ @tagName(self.code), src, self.message });
        } else {
            return std.fmt.allocPrint(allocator, "[{s}] {s}", .{ @tagName(self.code), self.message });
        }
    }
};

/// Error handler trait
pub fn ErrorHandler(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Handles an error and returns a result
        handle: fn (self: *Self, err: Error, context: ?ErrorContext) T,

        /// Default error handler that returns the error
        pub fn default() Self {
            return Self{
                .handle = struct {
                    fn handle(self: *Self, err: Error, context: ?ErrorContext) T {
                        _ = self;
                        _ = context;
                        return err;
                    }
                }.handle,
            };
        }

        /// Logging error handler that logs and returns error
        pub fn logging(logger: *std.log.Logger) Self {
            return Self{
                .handle = struct {
                    fn handle(self: *Self, err: Error, context: ?ErrorContext) T {
                        _ = self;
                        if (context) |ctx| {
                            logger.err("Error: {s} - {s}", .{ @errorName(err), ctx.message });
                        } else {
                            logger.err("Error: {s}", .{@errorName(err)});
                        }
                        return err;
                    }
                }.handle,
            };
        }
    };
}

/// Utility functions for error handling
pub const utils = struct {
    /// Ensures a value is not empty
    pub fn ensureNonEmpty(_: []const u8, value: []const u8) !void {
        if (value.len == 0) {
            return Error.Empty;
        }
    }

    /// Ensures a value is not null
    pub fn ensureNotNull(comptime T: type, name: []const u8, value: ?T) !void {
        _ = name; // Parameter name for documentation
        if (value == null) {
            return Error.InvalidParameter;
        }
    }

    /// Converts a Zig error to an error code
    pub fn errorToCode(err: Error) types.ErrorCode {
        return switch (err) {
            Error.InvalidConfig => .config_error,
            Error.InvalidParameter => .invalid_request,
            Error.NotFound => .not_found,
            Error.AlreadyExists => .invalid_request,
            Error.Timeout => .timeout,
            Error.RateLimited => .rate_limited,
            Error.PermissionDenied => .forbidden,
            Error.Unavailable => .unavailable,
            Error.NetworkError => .unavailable,
            Error.SerializationError => .invalid_request,
            Error.DeserializationError => .invalid_request,
            Error.ValidationError => .invalid_request,
            Error.Empty => .invalid_request,
            Error.OutOfMemory => .internal_error,
            Error.NotImplemented => .internal_error,
            Error.FeatureDisabled => .forbidden,
            Error.InternalError => .internal_error,
        };
    }

    /// Creates an error context from a Zig error
    pub fn createContext(err: Error, message: []const u8) ErrorContext {
        return ErrorContext.init(errorToCode(err), message);
    }
};

test "errors - error context" {
    const context = ErrorContext.init(.invalid_request, "test error");
    const formatted = try context.format(std.testing.allocator);
    defer std.testing.allocator.free(formatted);

    try std.testing.expectEqualStrings("[invalid_request] test error", formatted);
}

test "errors - error context with source" {
    const context = ErrorContext.initWithSource(.not_found, "resource not found", "database");
    const formatted = try context.format(std.testing.allocator);
    defer std.testing.allocator.free(formatted);

    try std.testing.expectEqualStrings("[not_found] database: resource not found", formatted);
}

test "errors - utils" {
    try utils.ensureNonEmpty("name", "test");
    try std.testing.expectError(Error.Empty, utils.ensureNonEmpty("name", ""));

    const value: ?u32 = 42;
    try utils.ensureNotNull("value", value);

    const null_value: ?u32 = null;
    try std.testing.expectError(Error.InvalidParameter, utils.ensureNotNull("value", null_value));
}
