const std = @import("std");
const builtin = @import("builtin");

comptime {
    if (builtin.zig_version.major == 0 and
        (builtin.zig_version.minor < 15 or
            (builtin.zig_version.minor == 15 and builtin.zig_version.patch < 2)))
    {
        @compileError("This code requires Zig 0.15.2 or newer");
    }
}
