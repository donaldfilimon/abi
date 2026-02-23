const std = @import("std");
const utils = @import("../utils/mod.zig");

test "utils.output print capture" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    
    // We want to test capturing output.
}
