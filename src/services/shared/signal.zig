//! POSIX signal handling for graceful shutdown.
//!
//! Installs handlers for SIGINT and SIGTERM that set a shared atomic flag.
//! The CLI main loop checks this flag to initiate graceful shutdown.

const std = @import("std");
const builtin = @import("builtin");

var shutdown_requested = std.atomic.Value(bool).init(false);

/// Returns true if a shutdown signal has been received.
pub fn isShutdownRequested() bool {
    return shutdown_requested.load(.acquire);
}

/// Request shutdown programmatically (for testing or internal use).
pub fn requestShutdown() void {
    shutdown_requested.store(true, .release);
}

/// Reset shutdown flag (for testing).
pub fn reset() void {
    shutdown_requested.store(false, .release);
}

/// Install signal handlers for SIGINT and SIGTERM.
/// On non-POSIX platforms this is a no-op.
pub fn install() void {
    if (comptime builtin.os.tag == .windows) return;

    const handler: std.posix.Sigaction = .{
        .handler = .{ .handler = handleSignal },
        .mask = std.posix.empty_sigset,
        .flags = .{},
    };

    std.posix.sigaction(std.posix.SIG.INT, &handler, null) catch |err| {
        std.log.warn("Failed to register SIGINT handler: {t}", .{err});
    };
    std.posix.sigaction(std.posix.SIG.TERM, &handler, null) catch |err| {
        std.log.warn("Failed to register SIGTERM handler: {t}", .{err});
    };
}

fn handleSignal(sig: c_int) callconv(.c) void {
    _ = sig;
    shutdown_requested.store(true, .release);
}
