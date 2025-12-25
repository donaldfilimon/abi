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

pub fn sleep(nanoseconds: u64) void {
    if (builtin.os.tag == .windows) {
        const ms = nanoseconds / std.time.ns_per_ms;
        _ = std.os.windows.kernel32.SleepEx(@intCast(ms), 0);
    } else {
        const s = nanoseconds / std.time.ns_per_s;
        const ns = nanoseconds % std.time.ns_per_s;
        _ = std.posix.nanosleep(&.{ .sec = @intCast(s), .nsec = @intCast(ns) }, null);
    }
}
