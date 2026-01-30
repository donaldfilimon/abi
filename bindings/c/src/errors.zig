//! C-compatible error code mappings for ABI framework.
//! Maps Zig errors to integer error codes for FFI.

const std = @import("std");

pub const Error = c_int;

// Error codes matching abi_errors.h
pub const OK: Error = 0;
pub const INIT_FAILED: Error = -1;
pub const ALREADY_INITIALIZED: Error = -2;
pub const NOT_INITIALIZED: Error = -3;
pub const OUT_OF_MEMORY: Error = -4;
pub const INVALID_ARGUMENT: Error = -5;
pub const FEATURE_DISABLED: Error = -6;
pub const TIMEOUT: Error = -7;
pub const IO: Error = -8;
pub const GPU_UNAVAILABLE: Error = -9;
pub const DATABASE_ERROR: Error = -10;
pub const NETWORK_ERROR: Error = -11;
pub const AI_ERROR: Error = -12;
pub const UNKNOWN: Error = -99;

/// Convert a Zig error to a C error code.
pub fn fromZigError(err: anyerror) Error {
    return switch (err) {
        error.OutOfMemory => OUT_OF_MEMORY,
        error.InvalidArgument => INVALID_ARGUMENT,
        error.FeatureDisabled => FEATURE_DISABLED,
        error.GpuDisabled => FEATURE_DISABLED,
        error.AiDisabled => FEATURE_DISABLED,
        error.DatabaseDisabled => FEATURE_DISABLED,
        error.NetworkDisabled => FEATURE_DISABLED,
        error.Timeout => TIMEOUT,
        error.GpuUnavailable => GPU_UNAVAILABLE,
        error.GpuInitFailed => GPU_UNAVAILABLE,
        error.DatabaseError => DATABASE_ERROR,
        error.NetworkError => NETWORK_ERROR,
        else => UNKNOWN,
    };
}

/// Get human-readable error message.
pub fn errorString(code: Error) [*:0]const u8 {
    return switch (code) {
        OK => "Success",
        INIT_FAILED => "Initialization failed",
        ALREADY_INITIALIZED => "Already initialized",
        NOT_INITIALIZED => "Not initialized",
        OUT_OF_MEMORY => "Out of memory",
        INVALID_ARGUMENT => "Invalid argument",
        FEATURE_DISABLED => "Feature disabled",
        TIMEOUT => "Operation timed out",
        IO => "I/O error",
        GPU_UNAVAILABLE => "GPU unavailable",
        DATABASE_ERROR => "Database error",
        NETWORK_ERROR => "Network error",
        AI_ERROR => "AI error",
        else => "Unknown error",
    };
}

// C export
pub export fn abi_error_string(code: Error) [*:0]const u8 {
    return errorString(code);
}

test "error code mapping" {
    try std.testing.expectEqual(OUT_OF_MEMORY, fromZigError(error.OutOfMemory));
    try std.testing.expectEqual(FEATURE_DISABLED, fromZigError(error.FeatureDisabled));
    try std.testing.expectEqual(UNKNOWN, fromZigError(error.FileNotFound));
}

test "error string" {
    try std.testing.expectEqualStrings("Success", std.mem.span(errorString(OK)));
    try std.testing.expectEqualStrings("Out of memory", std.mem.span(errorString(OUT_OF_MEMORY)));
}
