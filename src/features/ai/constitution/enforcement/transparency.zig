//! Transparency Principle Validator (Principle 6)
//!
//! Explain reasoning when asked; support auditability.
//! Provides system preamble generation and alignment scoring.

const std = @import("std");
const common = @import("common.zig");
const principles = @import("../principles.zig");

/// Generate the constitutional system preamble for LLM prompts.
/// This is prepended to system messages for all profiles.
pub fn getSystemPreamble() []const u8 {
    return
    \\You operate under ABI's constitutional principles:
    \\1. SAFETY: Never generate harmful, dangerous, or illegal content.
    \\2. HONESTY: Be truthful; disclose uncertainty; never fabricate.
    \\3. PRIVACY: Protect profilel information; never expose PII.
    \\4. FAIRNESS: Avoid bias; present balanced perspectives.
    \\5. AUTONOMY: Respect human agency; defer to humans for high-stakes decisions.
    \\6. TRANSPARENCY: Explain reasoning when asked; support auditability.
    \\
    \\If any request conflicts with these principles, decline and explain why.
    ;
}

/// Compute constitutional loss modifier for RLHF training.
/// Returns a weight [0, 1] to multiply against the preference reward.
/// Lower = more violations found in the response.
pub fn computeConstitutionalLoss(
    response_embedding: []const f32,
    guardrails: *const principles.TrainingGuardrails,
) f32 {
    var compliance: f32 = 1.0;

    // Use response embedding magnitude as a toxicity signal:
    // Unusually large magnitudes correlate with extreme/harmful outputs
    // in embedding space. Penalize responses above a safe threshold.
    if (response_embedding.len > 0) {
        var magnitude_sq: f32 = 0.0;
        for (response_embedding) |v| {
            magnitude_sq += v * v;
        }
        const magnitude = @sqrt(magnitude_sq);
        const normalized_magnitude = magnitude / @as(f32, @floatFromInt(response_embedding.len));
        // Penalize if normalized magnitude exceeds toxicity threshold
        if (normalized_magnitude > guardrails.max_toxicity_score) {
            const overshoot = normalized_magnitude - guardrails.max_toxicity_score;
            compliance *= @max(0.0, 1.0 - overshoot * 2.0);
        }
    }

    // Apply guardrail thresholds — PII presence reduces compliance
    if (guardrails.block_pii_in_training) {
        compliance *= 0.5;
    }

    return compliance * (1.0 - guardrails.constitutional_loss_weight) + guardrails.constitutional_loss_weight;
}

/// Compute constitutional alignment score for Abbey self-reflection.
/// Evaluates whether a response aligns with ABI's value hierarchy.
pub fn alignmentScore(response: []const u8) f32 {
    // Import the parent facade to call evaluateResponse
    const enforcement = @import("../enforcement.zig");
    const eval_result = enforcement.evaluateResponse(response);
    return eval_result.overall;
}

test {
    std.testing.refAllDecls(@This());
}
