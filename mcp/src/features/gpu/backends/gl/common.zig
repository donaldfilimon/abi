const std = @import("std");

pub const Api = enum {
    opengl,
    opengles,
};

test {
    std.testing.refAllDecls(@This());
}
