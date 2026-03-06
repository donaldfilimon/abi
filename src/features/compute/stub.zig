//! Compute stub — disabled at compile time.

const std = @import("std");

pub const mesh = struct {};

test {
    std.testing.refAllDecls(@This());
}
