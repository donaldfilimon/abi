//! Parity required-declaration specs split out from the monolithic parity file.
const std = @import("std");

pub const gpu = @import("gpu.zig");
pub const required = @import("required.zig");

test {
    std.testing.refAllDecls(@This());
}
