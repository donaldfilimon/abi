const std = @import("std");

/// Exit codes exposed to the host environment. Mirrors AGENTS.md specification.
pub const ExitCode = enum(u8) {
    success = 0,
    usage = 1,
    config = 2,
    runtime = 3,
    io = 4,
    backend_missing = 5,
};

/// Canonical error set used by CLI command handlers so we can map into exit codes.
pub const CommandError = error{
    InvalidArgument,
    MissingArgument,
    RateLimited,
    ConfigFailure,
    RuntimeFailure,
    IoFailure,
    BackendUnavailable,
};

/// Map a `CommandError` into an exit code.
pub fn exitCodeFor(err: CommandError) ExitCode {
    return switch (err) {
        error.InvalidArgument, error.MissingArgument => .usage,
        error.ConfigFailure => .config,
        error.RateLimited, error.RuntimeFailure => .runtime,
        error.IoFailure => .io,
        error.BackendUnavailable => .backend_missing,
    };
}

/// Convert generic allocation failures into CLI-specific errors.
pub fn mapAllocatorError(err: std.mem.Allocator.Error) CommandError {
    _ = err;
    return error.RuntimeFailure;
}
