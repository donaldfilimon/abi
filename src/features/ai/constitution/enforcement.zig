//! Constitutional Enforcement — Integration hooks for ABI's safety systems.
//!
//! Provides enforcement mechanisms that integrate with:
//! - Pre-generation: System prompt preamble injection
//! - Training: Constitutional loss term for RLHF reward model
//! - Post-generation: Response validation against principles
//! - Reflection: Constitutional alignment scoring for Abbey
//!
//! Safety heuristics use pattern-based detection with context-aware scoring.
//! Patterns found inside code blocks (``` fenced) are weighted lower to
//! reduce false positives when discussing code legitimately.
//!
//! Implementation is decomposed into per-principle validator modules under
//! `enforcement/`. This file is a thin re-export facade.

const std = @import("std");

// -- Sub-modules (one per constitutional principle) --
pub const common = @import("enforcement/common.zig");
pub const safety = @import("enforcement/safety.zig");
pub const honesty = @import("enforcement/honesty.zig");
pub const privacy = @import("enforcement/privacy.zig");
pub const fairness = @import("enforcement/fairness.zig");
pub const autonomy = @import("enforcement/autonomy.zig");
pub const transparency = @import("enforcement/transparency.zig");
pub const engine = @import("enforcement/engine.zig");

// -- Re-exported types (preserve public surface) --
pub const ConstitutionalScore = common.ConstitutionalScore;
pub const Violation = common.Violation;
pub const SafetyScore = common.SafetyScore;
pub const SafetyViolation = common.SafetyViolation;
pub const BiasScore = fairness.BiasScore;
pub const MAX_BIAS_ATTRIBUTES = fairness.MAX_BIAS_ATTRIBUTES;
pub const DEFAULT_BIAS_THRESHOLD = fairness.DEFAULT_BIAS_THRESHOLD;

// -- Re-exported functions (preserve public surface) --
pub const getSystemPreamble = transparency.getSystemPreamble;
pub const computeConstitutionalLoss = transparency.computeConstitutionalLoss;
pub const alignmentScore = transparency.alignmentScore;
pub const computeBias = fairness.computeBias;

// Re-export core evaluation functions from the engine
pub const evaluateResponse = engine.evaluateResponse;
pub const evaluateSafety = engine.evaluateSafety;
pub const checkForbiddenPattern = engine.checkForbiddenPattern;

test {
    std.testing.refAllDecls(@This());
}
