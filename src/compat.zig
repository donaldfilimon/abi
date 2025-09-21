const std = @import("std");
const builtin = @import("builtin");

comptime {
    if (builtin.zig_version.major != 0 or builtin.zig_version.minor != 15) {
        @compileError("This project requires Zig 0.15.x; current compiler is " ++
            std.fmt.comptimePrint("{d}.{d}.{d}", .{
                builtin.zig_version.major,
                builtin.zig_version.minor,
                builtin.zig_version.patch,
            }));
    }
}
