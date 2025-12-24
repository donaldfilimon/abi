//! GPU Backends Module
//!
//! Cross-platform GPU backend implementations

const std = @import("std");

pub const backends = @import("backends.zig");

test {
    std.testing.refAllDecls(@This());
}
