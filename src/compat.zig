const std = @import("std");
const builtin = @import("builtin");

comptime {
    const is_supported = (builtin.zig_version.major == 0 and builtin.zig_version.minor >= 16) or
        (builtin.zig_version.major > 0);

    if (!is_supported) {
        @compileError("This code requires Zig 0.16.0-dev or newer; current compiler is " ++
            std.fmt.comptimePrint("{d}.{d}.{d}", .{
                builtin.zig_version.major,
                builtin.zig_version.minor,
                builtin.zig_version.patch,
            }));
    }
}
