const std = @import("std");
const builtin = @import("builtin");

/// Ensures the repository is compiled with Zig 0.15.x toolchains.
///
/// Emits a compile error if the detected compiler version is not 0.15.x.
pub fn ensureZig015x() void {
    comptime {
        if (builtin.zig_version.major != 0 or builtin.zig_version.minor != 15) {
            @compileError("This codebase requires Zig 0.15.x; detected " ++
                std.fmt.comptimePrint("{d}.{d}.{d}", .{
                    builtin.zig_version.major,
                    builtin.zig_version.minor,
                    builtin.zig_version.patch,
                }));
        }
    }
}
