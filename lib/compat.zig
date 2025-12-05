const std = @import("std");
const builtin = @import("builtin");

// Version compatibility check - temporarily allow 0.15.2 for testing
// TODO: Re-enable strict 0.16+ check after migration is complete
comptime {
    if (builtin.zig_version.major == 0 and builtin.zig_version.minor < 15) {
        @compileError("This code requires Zig 0.15.0 or newer");
    }

    // Warning for versions below 0.16
    if (builtin.zig_version.major == 0 and builtin.zig_version.minor < 16) {
        @compileLog("Warning: Using Zig 0.15.x. This codebase is designed for Zig 0.16.0+");
    }
}

test "zig version check" {
    if (builtin.zig_version.major == 0 and builtin.zig_version.minor < 15) {
        @compileError("This code requires Zig 0.15.0 or newer");
    }
}
