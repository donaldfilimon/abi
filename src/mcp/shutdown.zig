const std = @import("std");

var requested = std.atomic.Value(bool).init(false);

pub fn request() void {
    requested.store(true, .release);
}

pub fn isRequested() bool {
    return requested.load(.acquire);
}

pub fn installSignalHandlers() void {
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
