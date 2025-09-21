const std = @import("std");
const builtin = @import("builtin");

pub const supportedMajor: u8 = 0;
pub const supportedMinor: u8 = 15;

comptime {
    const version = builtin.zig_version;
    if (version.major != supportedMajor or version.minor != supportedMinor) {
        @compileError(std.fmt.comptimePrint(
            "abi requires Zig {d}.{d}.x but detected {d}.{d}.{d}. Update your toolchain to Zig 0.15.x.",
            .{ supportedMajor, supportedMinor, version.major, version.minor, version.patch },
        ));
    }
}
