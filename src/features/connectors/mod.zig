//! Connectors Feature Module
//!
//! External service integrations and plugin system

const std = @import("std");

// AI service connectors
pub const openai = @import("openai.zig");
pub const ollama = @import("ollama.zig");

// Plugin system
pub const plugin = @import("plugin.zig");

// Legacy compatibility removed - circular import fixed

test {
    std.testing.refAllDecls(@This());
}
