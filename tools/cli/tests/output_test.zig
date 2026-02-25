const std = @import("std");
const utils = @import("../utils/mod.zig");

test "output module inline tests" {
    // Pull in all inline tests from the output module.
    _ = utils.output;
}

test "color re-export alias works" {
    // The `color` alias should refer to the same Color struct.
    utils.output.enableColor();
    try std.testing.expectEqualStrings(utils.output.Color.red(), utils.output.color.red());
}

test {
    std.testing.refAllDecls(@This());
}
