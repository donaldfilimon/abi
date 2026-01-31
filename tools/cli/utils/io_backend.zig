//! CLI I/O backend helper (Zig 0.16).

const std = @import("std");

/// Initialize a threaded I/O backend with the process environment.
pub fn initIoBackend(allocator: std.mem.Allocator) std.Io.Threaded {
    return std.Io.Threaded.init(allocator, .{
        .environ = std.process.Environ.empty,
    });
}
