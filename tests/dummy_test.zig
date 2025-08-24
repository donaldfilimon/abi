const std = @import("std");
const testing = std.testing;

test "simple test" {
    try testing.expect(1 + 1 == 2);
}
