const std = @import("std");
const builtin = @import("builtin");

var requested = std.atomic.Value(bool).init(false);

pub fn request() void {
    requested.store(true, .release);
}

pub fn isRequested() bool {
    return requested.load(.acquire);
}

pub fn installSignalHandlers() void {
    // `switch` on a comptime-known tag analyzes only the matching prong, so the
    // POSIX sigaction path is never type-checked on Windows (which lacks it).
    switch (builtin.os.tag) {
        // Windows has no POSIX sigaction. Ctrl-C handling via
        // SetConsoleCtrlHandler is a documented gap (std does not expose it in
        // this toolchain); the stdio EOF path still drives a clean shutdown.
        .windows => {},
        else => installPosixSignalHandlers(),
    }
}

fn installPosixSignalHandlers() void {
    const posix = std.posix;
    const handler = posix.Sigaction{
        .handler = .{ .handler = signalHandler },
        .mask = posix.sigemptyset(),
        .flags = 0,
    };
    posix.sigaction(posix.SIG.INT, &handler, null);
    posix.sigaction(posix.SIG.TERM, &handler, null);
}

fn signalHandler(sig: @TypeOf(std.posix.SIG.INT)) callconv(.c) void {
    _ = sig;
    request();
}

test {
    std.testing.refAllDecls(@This());
}
