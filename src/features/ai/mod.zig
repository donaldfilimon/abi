//! AI Feature Root
//!
//! Exposes AI sub-features: core, profiles, llm, agents, reasoning, and training.

const std = @import("std");
const build_options = @import("build_options");

// Public types and sub-features
pub const types = @import("types");
pub const config = @import("config");
pub const registry = @import("registry");
pub const profiles = @import("profiles");

// Sub-feature modules (conditional implementation vs stub)
pub const core = if (build_options.feat_ai) @import("types") else @import("core/stub");
pub const agents = if (build_options.feat_ai) @import("agents") else @import("agents/stub");
pub const llm = if (build_options.feat_ai) @import("llm") else @import("llm/stub");
pub const training = if (build_options.feat_ai) @import("training") else @import("training/stub");
pub const reasoning = if (build_options.feat_ai) @import("reasoning") else @import("reasoning/stub");
pub const explore = if (build_options.feat_ai) @import("explore") else @import("explore/stub");

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
