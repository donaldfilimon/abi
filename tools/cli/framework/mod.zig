//! Common descriptor-driven CLI framework.

pub const types = @import("types.zig");
pub const context = @import("context.zig");
pub const completion = @import("completion.zig");
pub const help = @import("help.zig");
pub const router = @import("router.zig");
pub const errors = @import("errors.zig");

const std = @import("std");
test {
    std.testing.refAllDecls(@This());
}
