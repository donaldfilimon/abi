//! Shared focused-test helpers.

const std = @import("std");

pub fn refAllDecls(comptime namespace: type) void {
    std.testing.refAllDecls(namespace);
}

test {
    refAllDecls(@This());
}
