const std = @import("std");

pub const AbiError = error{
    InvalidConfig,
    InvalidPath,
    FileNotFound,
    PermissionDenied,
    OutOfMemory,
    InternalError,
    NotInitialized,
    AlreadyInitialized,
    InvalidState,
    Timeout,
    ConnectionFailed,
    ParseError,
    ValidationError,
    UnsupportedOperation,
    NotFound,
};

pub const ErrorContext = struct {
    error_type: AbiError,
    message: []const u8,
    source: []const u8,
    line: u32,
};

pub fn formatError(err: AbiError) []const u8 {
    return switch (err) {
        error.InvalidConfig => "invalid configuration",
        error.InvalidPath => "invalid path",
        error.FileNotFound => "file not found",
        error.PermissionDenied => "permission denied",
        error.OutOfMemory => "out of memory",
        error.InternalError => "internal error",
        error.NotInitialized => "not initialized",
        error.AlreadyInitialized => "already initialized",
        error.InvalidState => "invalid state",
        error.Timeout => "operation timed out",
        error.ConnectionFailed => "connection failed",
        error.ParseError => "parse error",
        error.ValidationError => "validation error",
        error.UnsupportedOperation => "unsupported operation",
        error.NotFound => "not found",
    };
}

pub fn describeError(err: anyerror) []const u8 {
    inline for (@typeInfo(AbiError).error_set.?) |e| {
        if (std.mem.eql(u8, @errorName(err), e.name)) {
            return formatError(@as(AbiError, @field(AbiError, e.name)));
        }
    }
    return "unknown error";
}

test {
    std.testing.refAllDecls(@This());
}

test "formatError" {
    try std.testing.expectEqualStrings("invalid configuration", formatError(error.InvalidConfig));
    try std.testing.expectEqualStrings("file not found", formatError(error.FileNotFound));
    try std.testing.expectEqualStrings("out of memory", formatError(error.OutOfMemory));
    try std.testing.expectEqualStrings("not found", formatError(error.NotFound));
}

test "AbiError error set" {
    const err: AbiError = error.InvalidConfig;
    try std.testing.expectEqual(error.InvalidConfig, err);
}
