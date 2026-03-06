//! Documents stub — disabled at compile time.

const std = @import("std");

pub const html = struct {};
pub const pdf = struct {};

test {
    std.testing.refAllDecls(@This());
}
