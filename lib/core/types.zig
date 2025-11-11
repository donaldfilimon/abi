//! Core Types Module
//!
//! Fundamental types and error definitions used throughout the framework

/// Standard error codes used across the framework
pub const ErrorCode = enum(u16) {
    /// Operation completed successfully
    ok = 0,
    /// Invalid request or parameters
    invalid_request = 400,
    /// Service unavailable or busy
    unavailable = 503,
    /// Internal server error
    internal_error = 500,
    /// Resource not found
    not_found = 404,
    /// Authentication required
    unauthorized = 401,
    /// Insufficient permissions
    forbidden = 403,
    /// Request timeout
    timeout = 408,
    /// Rate limit exceeded
    rate_limited = 429,
    /// Configuration error
    config_error = 422,
};

/// Standard result structure for API responses
pub const Result = struct {
    /// Error code indicating success or failure type
    code: ErrorCode = .ok,
    /// Human-readable message describing the result
    message: []const u8 = "",
    /// Optional data payload
    data: ?[]const u8 = null,
};

/// Generic result type for operations that can fail
pub fn GenericResult(comptime T: type) type {
    return struct {
        const Self = @This();

        /// The result value (only valid if success is true)
        value: T,
        /// Whether the operation was successful
        success: bool,
        /// Error message if operation failed
        error_message: []const u8 = "",

        /// Creates a successful result
        pub fn success(value: T) Self {
            return Self{
                .value = value,
                .success = true,
            };
        }

        /// Creates a failed result
        pub fn failure(error_message: []const u8) Self {
            return Self{
                .value = undefined,
                .success = false,
                .error_message = error_message,
            };
        }
    };
}

/// Version information structure
pub const Version = struct {
    major: u32,
    minor: u32,
    patch: u32,

    /// Creates a version from string
    pub fn fromString(version_string: []const u8) !Version {
        var parts = std.mem.split(u8, version_string, ".");
        const major_str = parts.next() orelse return error.InvalidVersion;
        const minor_str = parts.next() orelse return error.InvalidVersion;
        const patch_str = parts.next() orelse return error.InvalidVersion;

        return Version{
            .major = try std.fmt.parseInt(u32, major_str, 10),
            .minor = try std.fmt.parseInt(u32, minor_str, 10),
            .patch = try std.fmt.parseInt(u32, patch_str, 10),
        };
    }

    /// Formats version as string
    pub fn toString(self: Version, allocator: std.mem.Allocator) ![]u8 {
        return std.fmt.allocPrint(allocator, "{d}.{d}.{d}", .{ self.major, self.minor, self.patch });
    }

    /// Compares versions (returns -1, 0, or 1)
    pub fn compare(self: Version, other: Version) i8 {
        if (self.major != other.major) return if (self.major > other.major) 1 else -1;
        if (self.minor != other.minor) return if (self.minor > other.minor) 1 else -1;
        if (self.patch != other.patch) return if (self.patch > other.patch) 1 else -1;
        return 0;
    }
};

const std = @import("std");

test "types - result" {
    const result = GenericResult(u32).success(42);
    try std.testing.expect(result.success);
    try std.testing.expectEqual(@as(u32, 42), result.value);

    const failure = GenericResult(u32).failure("test error");
    try std.testing.expect(!failure.success);
    try std.testing.expectEqualStrings("test error", failure.error_message);
}

test "types - version" {
    const version = try Version.fromString("1.2.3");
    try std.testing.expectEqual(@as(u32, 1), version.major);
    try std.testing.expectEqual(@as(u32, 2), version.minor);
    try std.testing.expectEqual(@as(u32, 3), version.patch);

    const version_str = try version.toString(std.testing.allocator);
    defer std.testing.allocator.free(version_str);
    try std.testing.expectEqualStrings("1.2.3", version_str);

    const newer = try Version.fromString("2.0.0");
    try std.testing.expectEqual(@as(i8, -1), version.compare(newer));
}
