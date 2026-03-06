const std = @import("std");

test {
    _ = @import("wdbx/wdbx.zig");
    _ = @import("features/database/wdbx.zig");
}

test {
    std.testing.refAllDecls(@This());
}
