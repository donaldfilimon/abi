const std = @import("std");
const core = @import("../mod.zig");

test "core.version returns placeholder" {
    try std.testing.expectEqualStrings("0.1.0", core.version());
}
