//! Shared Core Module
//!
//! Core system functionality and framework components

const std = @import("std");

// Core system components
pub const core = @import("core.zig");
pub const config = @import("config.zig");
pub const profile = @import("profile.zig");

// System utilities
pub const lifecycle = @import("lifecycle.zig");
pub const errors = @import("errors.zig");

// Legacy compatibility (will be removed in future version)
// Use shared/logging instead for new code
pub const logging = @import("logging.zig");
// Use features/ai/persona_manifest instead for new code
pub const persona_manifest = @import("persona_manifest.zig");
pub const profiles = @import("profiles.zig");

// Common containers for framework-level usage.
pub const ArrayList = std.ArrayList;
pub const StringHashMap = std.StringHashMap;

// Minimal error set used by runtime component registry.
pub const Error = error{
    AlreadyExists,
    NotFound,
};

test {
    std.testing.refAllDecls(@This());
}
