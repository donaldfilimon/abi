//! Shared Platform Module
//!
//! Platform-specific utilities and abstractions

const std = @import("std");

pub const platform = @import("platform.zig");
pub const accelerator = @import("accelerator/mod.zig");

test {
    std.testing.refAllDecls(@This());
}
