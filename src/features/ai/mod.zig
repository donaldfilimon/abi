//! AI Feature Root
//!
//! Exposes AI sub-features: core, profiles, llm, agents, reasoning, and training.

const std = @import("std");
const build_options = @import("build_options");

// Public types and sub-features
pub const types = @import("types.zig");
pub const config = @import("config.zig");
pub const registry = @import("registry.zig");
pub const profiles = @import("profiles/mod.zig");

// Sub-feature modules (conditional implementation vs stub)
pub const core = if (build_options.feat_ai) @import("types.zig") else @import("core/stub.zig");
pub const agents = if (build_options.feat_ai) @import("agents/mod.zig") else @import("agents/stub.zig");
pub const llm = if (build_options.feat_ai) @import("llm/mod.zig") else @import("llm/stub.zig");
pub const training = if (build_options.feat_ai) @import("training/mod.zig") else @import("training/stub.zig");
pub const reasoning = if (build_options.feat_ai) @import("reasoning/mod.zig") else @import("reasoning/stub.zig");
pub const explore = if (build_options.feat_ai) @import("explore/mod.zig") else @import("explore/stub.zig");

// Compatibility layer for Phase 4 transition
pub const personas = profiles;

/// Initialize the AI feature set.
pub fn init(allocator: std.mem.Allocator, cfg: config.AiConfig) !void {
    if (!build_options.feat_ai) return;
    _ = allocator;
    _ = cfg;
    // Initialization logic for modular AI features
}

test {
    std.testing.refAllDecls(@This());
}
