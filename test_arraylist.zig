const std = @import("std");

test "arraylist api" {
    const testing = std.testing;

    // Test ArrayList init method
    var list = std.ArrayList(u32){};
    defer list.deinit(testing.allocator);

    try list.append(testing.allocator, 42);
    try testing.expect(list.items.len == 1);
    try testing.expect(list.items[0] == 42);
}
