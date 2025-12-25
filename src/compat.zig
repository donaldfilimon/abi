const std = @import("std");
const builtin = @import("builtin");

comptime {
    if (builtin.zig_version.major == 0 and
        (builtin.zig_version.minor < 16 or
            (builtin.zig_version.minor == 16 and builtin.zig_version.patch < 0)))
    {
        @compileError("This code requires Zig 0.16.0 or newer");
    }
}
