//! CLI utility modules for shared functionality.

pub const args = @import("args");
pub const output = @import("output");
pub const help = @import("help");
pub const global_flags = @import("global_flags");
pub const io_backend = @import("io_backend");
pub const process = @import("process");
pub const subcommand = @import("subcommand");

const std = @import("std");
test {
    std.testing.refAllDecls(@This());
}
