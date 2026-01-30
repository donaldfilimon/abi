//! Central I/O backend for Zig 0.16
//! ------------------------------------------------------------
//! This module provides a single `std.Io` instance that the rest of the
//! code‑base can share.  Zig 0.16 introduced the `std.Io` abstraction
//! which replaces the deprecated `std.Io.Dir.cwd()` and other direct
//! filesystem calls.  All file‑system and network operations should
//! import this module and use `io_backend.io` rather than creating their
//! own I/O context.
//!
//! Usage pattern (in `main.zig` or wherever the program entry lives):
//! ```zig
//! const IoBackend = @import("io").IoBackend;
//! var backend = try IoBackend.init(allocator);
//! defer backend.deinit();
//! // Pass `backend.io` down to the framework or any subsystem that
//! // needs file or network I/O.
//! ```
//!
//! The `IoBackend` is deliberately lightweight: it only stores the
//! allocator and the `std.Io` value.  No additional state is required.
//! When the process exits the backend is simply de‑initialised.

const std = @import("std");

pub const IoBackend = struct {
    /// Allocator used for any temporary allocations required by the
    /// `std.Io` implementation.
    allocator: std.mem.Allocator,

    /// The `std.Io` instance that is passed to lower‑level modules.
    io: std.Io,

    /// Initialise a new I/O backend.
    /// For library usage we use an empty environment; a CLI can pass a
    /// populated environment if needed.
    pub fn init(allocator: std.mem.Allocator) !IoBackend {
        // Initialise the threaded I/O backend with an empty environment.
        // This works for both library and CLI contexts.  If a CLI needs
        // environment variables you can replace `.empty` with
        // `std.process.Environ.init(allocator)` and pass that instead.
        var backend = try std.Io.Threaded.init(allocator, .{
            .environ = std.process.Environ.empty,
        });
        // No explicit deinit for Threaded is required in Zig 0.16,
        // but we keep the function signature for future‑proofing.
        return .{
            .allocator = allocator,
            .io = backend.io(),
        };
    }

    /// De‑initialise the backend.
    /// Currently a no‑op because the underlying Threaded backend does not
    /// expose a deinit method, but keeping the function makes future
    /// migrations easier.
    pub fn deinit(self: *IoBackend) void {
        // Placeholder for any future cleanup required by the Io subsystem.
        _ = self;
    }
};
