const coordinator = @import("coordinator.zig");
const std = @import("std");

pub const DispatchError = coordinator.DispatchError;

test {
    std.testing.refAllDecls(@This());
}
