//! CLI I/O backend helper (Zig 0.16).

const std = @import("std");

fn initEnviron(allocator: std.mem.Allocator) std.process.Environ {
    if (@hasDecl(std.process.Environ, "init")) {
        const Result = @TypeOf(std.process.Environ.init(allocator));
        if (@typeInfo(Result) == .error_union) {
            return std.process.Environ.init(allocator) catch std.process.Environ.empty;
        }
        return std.process.Environ.init(allocator);
    }

    return std.process.Environ.empty;
}

/// Initialize a threaded I/O backend with the process environment.
pub fn initIoBackend(allocator: std.mem.Allocator) std.Io.Threaded {
    return std.Io.Threaded.init(allocator, .{
        .environ = initEnviron(allocator),
    });
}
