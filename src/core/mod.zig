const std = @import("std");

pub const types = @import("types.zig");
pub const utils = @import("utils.zig");
pub const collections = @import("collections.zig");

pub fn version() []const u8 {
    return "0.1.0";
}

test {
    std.testing.refAllDecls(@This());
}
