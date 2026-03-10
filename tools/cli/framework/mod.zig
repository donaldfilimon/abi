//! Common descriptor-driven CLI framework.

pub const types = @import("types");
pub const context = @import("context");
pub const completion = @import("completion");
pub const help = @import("help");
pub const router = @import("router");
pub const errors = @import("errors");

const std = @import("std");
test {
    std.testing.refAllDecls(@This());
}
