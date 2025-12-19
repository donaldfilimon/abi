//! Shared Utilities Module
//!
//! Cross-cutting utility functions and helpers

const std = @import("std");

// Modular utility components
pub const http = @import("http/mod.zig");
pub const json = @import("json/mod.zig");
pub const string = @import("string/mod.zig");
pub const math = @import("math/mod.zig");

// Additional utilities
pub const encoding = @import("encoding/mod.zig");
pub const fs = @import("fs/mod.zig");
pub const net = @import("net/mod.zig");
pub const crypto = @import("crypto/mod.zig");
pub const security = @import("security.zig");

// Main utilities interface (legacy compatibility)
pub const utils = @import("utils.zig");

test {
    std.testing.refAllDecls(@This());
}
