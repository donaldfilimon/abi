//! Desktop Integration
//!
//! Provides native UI extensions and integrations for the host OS.

pub const macos_menu = @import("macos_menu.zig");

const std = @import("std");

test {
    std.testing.refAllDecls(@This());
}
