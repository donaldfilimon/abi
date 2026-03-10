const std = @import("std");

test {
    _ = @import("mod");
    _ = @import("utils/mod.zig");
    _ = @import("framework/mod.zig");
    _ = @import("commands/mod.zig");
    _ = @import("tests/nested_integration_test");
}

test {
    std.testing.refAllDecls(@This());
}
