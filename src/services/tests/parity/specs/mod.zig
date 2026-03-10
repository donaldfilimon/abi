//! Parity required-declaration specs split out from the monolithic parity file.
const std = @import("std");

pub const gpu = @import("gpu");
pub const required = @import("required");

test {
    std.testing.refAllDecls(@This());
}
