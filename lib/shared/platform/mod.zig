//! Shared Platform Module
//!
//! Platform-specific utilities and abstractions

const std = @import("std");

pub const platform = @import("platform.zig");

test {
    std.testing.refAllDecls(@This());
}
