//! Shared Logging Module
//!
//! Centralized logging and telemetry functionality

const std = @import("std");

pub const logging = @import("logging.zig");

test {
    std.testing.refAllDecls(@This());
}
