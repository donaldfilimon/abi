//! Simple monitoring stub used by tutorials to show how custom modules plug
//! into the ABI example harness.

const std = @import("std");

pub const Monitoring = struct {
    pub fn printHello() void {
        std.debug.print("Monitoring module placeholder.\n", .{});
    }
};
