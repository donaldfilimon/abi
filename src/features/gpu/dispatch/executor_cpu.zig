const coordinator = @import("coordinator");
const std = @import("std");

pub const DispatchError = coordinator.DispatchError;

test {
    std.testing.refAllDecls(@This());
}
