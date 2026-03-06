//! Desktop stub — disabled at compile time.

const std = @import("std");

pub const macos_menu = struct {};

test {
    std.testing.refAllDecls(@This());
}
