const std = @import("std");
const builtin = @import("builtin");

comptime {
    if (builtin.zig_version.major == 0 and builtin.zig_version.minor < 16) {
        @compileError("This code requires Zig 0.16.0 or newer");
    }
}

test "zig version check" {
    if (builtin.zig_version.major == 0 and builtin.zig_version.minor < 16) {
        @compileError("This code requires Zig 0.16.0 or newer");
    }
}
