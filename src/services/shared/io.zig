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
//! const abi = @import("abi");
//! const IoBackend = abi.services.shared.io.IoBackend;
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

    /// Owns the underlying Zig 0.16 I/O backend.
    /// Keeping this alive ensures `io` remains valid for the backend lifetime.
    backend: std.Io.Threaded,

    /// The `std.Io` instance that is passed to lower‑level modules.
    io: std.Io,

    fn initThreaded(allocator: std.mem.Allocator, options: anytype) !std.Io.Threaded {
        // Zig 0.16 dev snapshots have shifted whether `std.Io.Threaded.init` returns
        // `Threaded` or `!Threaded`. Support both shapes for robustness.
        const Result = @TypeOf(std.Io.Threaded.init(allocator, options));
        if (@typeInfo(Result) == .error_union) {
            return try std.Io.Threaded.init(allocator, options);
        }
        return std.Io.Threaded.init(allocator, options);
    }

    /// Initialise a new I/O backend.
    /// For library usage we use an empty environment; a CLI can pass a
    /// populated environment if needed.
    pub fn init(allocator: std.mem.Allocator) !IoBackend {
        // Initialise the threaded I/O backend with an empty environment.
        // This works for both library and CLI contexts. If a CLI needs
        // environment variables you can pass `init.environ` from
        // `std.process.Init.Minimal` instead of `.empty`.
        var backend = try initThreaded(allocator, .{ .environ = std.process.Environ.empty });
        errdefer backend.deinit();

        return .{
            .allocator = allocator,
            .backend = backend,
            .io = backend.io(),
        };
    }

    /// De‑initialise the backend.
    pub fn deinit(self: *IoBackend) void {
        self.backend.deinit();
        self.* = undefined;
    }
};

test {
    std.testing.refAllDecls(@This());
}
