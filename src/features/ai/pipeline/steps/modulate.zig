//! Modulate Step — EMA preference adjustment.
//!
//! Applies learned user preferences to routing weights set by the route step.
//! In the standalone pipeline, this applies a fixed Abbey preference bias.
//! When wired to the real `AdaptiveModulator`, it would call `modulate()` with
//! the session's `UserProfile` for per-user EMA-based weight adjustment.
//!
//! Must run after the route step (requires `routing_weights` to be set).

const std = @import("std");
const types = @import("../types.zig");
const ctx_mod = @import("../context.zig");
const PipelineContext = ctx_mod.PipelineContext;

/// Default preference bias toward Abbey (the conversational profile).
const default_preference_bias: f32 = 0.05;

pub fn execute(pctx: *PipelineContext, _: types.ModulateConfig) !void {
    var weights = pctx.routing_weights orelse return;

    // Apply preference bias — in production, sourced from AdaptiveModulator's
    // per-session UserProfile EMA scores. Standalone: slight Abbey favor.
    weights.abbey_weight = std.math.clamp(weights.abbey_weight + default_preference_bias, 0.0, 1.0);

    // Renormalize to sum = 1.0
    const total = weights.abbey_weight + weights.aviva_weight + weights.abi_weight;
    if (total > 0) {
        weights.abbey_weight /= total;
        weights.aviva_weight /= total;
        weights.abi_weight /= total;
    }

    pctx.routing_weights = weights;

    // Re-derive primary profile after weight adjustment
    if (weights.abbey_weight >= weights.aviva_weight and weights.abbey_weight >= weights.abi_weight) {
        pctx.primary_profile = .abbey;
    } else if (weights.aviva_weight >= weights.abbey_weight and weights.aviva_weight >= weights.abi_weight) {
        pctx.primary_profile = .aviva;
    } else {
        pctx.primary_profile = .abi;
    }
}
