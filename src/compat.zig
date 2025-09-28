const std = @import("std");
const builtin = @import("builtin");

const minimum_version = std.SemanticVersion.parse("0.16.0-dev.254+6dd0270a1") catch @panic("invalid version");

comptime {
    if (builtin.zig_version.order(minimum_version) == .lt) {
        @compileError("This code requires Zig 0.16.0-dev.254+6dd0270a1 or newer; current compiler is " ++
            std.fmt.comptimePrint("{d}.{d}.{d}", .{
                builtin.zig_version.major,
                builtin.zig_version.minor,
                builtin.zig_version.patch,
            }));
    }
}
