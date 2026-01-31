//! CLI I/O backend helper (Zig 0.16).

const std = @import("std");

/// Initialize a threaded I/O backend with an empty environment.
pub fn initIoBackend(allocator: std.mem.Allocator) std.Io.Threaded {
    return initIoBackendWithEnv(allocator, std.process.Environ.empty);
}

/// Initialize a threaded I/O backend with the provided environment.
pub fn initIoBackendWithEnv(
    allocator: std.mem.Allocator,
    environ: std.process.Environ,
) std.Io.Threaded {
    return std.Io.Threaded.init(allocator, .{
        .environ = environ,
    });
}
