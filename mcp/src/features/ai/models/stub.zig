//! Stub-compatible model metadata exports.
//!
//! The models registry is dependency-light and remains available when the
//! broader AI feature is disabled, so the stub forwards to the canonical module
//! instead of duplicating its declarations.

const canonical = @import("mod.zig");

pub const registry = canonical.registry;
pub const ModelRegistry = canonical.ModelRegistry;
pub const ModelInfo = canonical.ModelInfo;

test {
    @import("std").testing.refAllDecls(@This());
}
