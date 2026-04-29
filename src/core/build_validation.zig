const std = @import("std");

pub fn validateToolchain(b: *std.Build) !void {
    const min_zig = "0.17.0";
    const current_zig = @import("builtin").zig_version_string;
    if (std.mem.order(u8, current_zig, min_zig) == .lt) {
        @compileError("ABI requires Zig " ++ min_zig ++ " or newer. Current: " ++ current_zig);
    }
}
