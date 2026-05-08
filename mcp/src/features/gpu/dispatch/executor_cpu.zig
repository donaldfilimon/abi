const coordinator = @import("coordinator/mod.zig");
const std = @import("std");

pub const DispatchError = coordinator.DispatchError;

test {
    std.testing.refAllDecls(@This());
}
