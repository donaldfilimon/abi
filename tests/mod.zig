const std = @import("std");
const abi = @import("abi");

test "abi version returns non-empty string" {
    try std.testing.expect(abi.version().len > 0);
}

test "phase5 integration tests" {
    @import("phase5_integration.zig");
    std.testing.refAllDecls(@This());
}
