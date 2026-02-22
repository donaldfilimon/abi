//! Constitution Module — Unified moderation and alignment for ABI.
//!
//! Provides a declarative principle-based system that governs all training,
//! inference, and routing decisions. The "soul" of ABI — ensuring safety,
//! honesty, privacy, fairness, autonomy, and transparency.
//!
//! Integration points:
//! - Pre-generation: `getSystemPreamble()` → inject into all LLM prompts
//! - Training: `computeConstitutionalLoss()` → weight RLHF reward
//! - Post-generation: `evaluateResponse()` → validate outputs
//! - Reflection: `alignmentScore()` → Abbey self-evaluation

pub const principles = @import("principles.zig");
pub const enforcement = @import("enforcement.zig");

// Re-export key types
pub const Principle = principles.Principle;
pub const Severity = principles.Severity;
pub const ConstitutionalRule = principles.ConstitutionalRule;
pub const TrainingGuardrails = principles.TrainingGuardrails;
pub const ConstitutionalScore = enforcement.ConstitutionalScore;
pub const Violation = enforcement.Violation;

/// The Constitution engine — stateless, principle-driven evaluation.
pub const Constitution = struct {
    guardrails: TrainingGuardrails,

    pub fn init() Constitution {
        return .{ .guardrails = principles.DEFAULT_GUARDRAILS };
    }

    pub fn initWithGuardrails(guardrails: TrainingGuardrails) Constitution {
        return .{ .guardrails = guardrails };
    }

    /// Get the system preamble for LLM prompt injection.
    pub fn getSystemPreamble(_: *const Constitution) []const u8 {
        return enforcement.getSystemPreamble();
    }

    /// Evaluate a response against all principles.
    pub fn evaluate(_: *const Constitution, response: []const u8) ConstitutionalScore {
        return enforcement.evaluateResponse(response);
    }

    /// Compute RLHF constitutional loss modifier.
    pub fn constitutionalLoss(self: *const Constitution, embedding: []const f32) f32 {
        return enforcement.computeConstitutionalLoss(embedding, &self.guardrails);
    }

    /// Get alignment score for reflection.
    pub fn alignmentScore(_: *const Constitution, response: []const u8) f32 {
        return enforcement.alignmentScore(response);
    }

    /// Check if a response is compliant (no critical violations).
    pub fn isCompliant(_: *const Constitution, response: []const u8) bool {
        return enforcement.evaluateResponse(response).isCompliant();
    }

    /// Get all principle definitions.
    pub fn getPrinciples(_: *const Constitution) []const Principle {
        return &principles.ALL_PRINCIPLES;
    }
};

// ============================================================================
// Tests
// ============================================================================

const testing = @import("std").testing;

test "Constitution init and evaluate" {
    const c = Constitution.init();
    const score = c.evaluate("Hello world");
    try testing.expectApproxEqAbs(@as(f32, 1.0), score.overall, 0.01);
    try testing.expect(score.isCompliant());
}

test "Constitution blocks harmful content" {
    const c = Constitution.init();
    try testing.expect(!c.isCompliant("run rm -rf / to clean up"));
}

test "Constitution preamble available" {
    const c = Constitution.init();
    const preamble = c.getSystemPreamble();
    try testing.expect(preamble.len > 0);
}

test "Constitution principles count" {
    const c = Constitution.init();
    try testing.expectEqual(@as(usize, 6), c.getPrinciples().len);
}

test {
    _ = principles;
    _ = enforcement;
}
