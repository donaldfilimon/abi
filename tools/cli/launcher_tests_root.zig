const std = @import("std");

test {
    _ = @import("terminal/launcher/launcher_catalog.zig");
    _ = @import("terminal/launcher/palette.zig");
}

test {
    std.testing.refAllDecls(@This());
}
