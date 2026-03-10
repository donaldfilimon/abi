const std = @import("std");

test {
    _ = @import("terminal/launcher/launcher_catalog");
    _ = @import("terminal/launcher/palette");
}

test {
    std.testing.refAllDecls(@This());
}
