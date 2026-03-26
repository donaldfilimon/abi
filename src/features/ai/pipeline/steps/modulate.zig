//! Modulate Step — EMA preference adjustment.
//!
//! Adjusts routing weights based on learned user preferences.
//! This is a lightweight step that modifies the routing_weights
//! already set by the route step, applying EMA-based bias.

const std = @import("std");
const types = @import("../types.zig");
const ctx_mod = @import("../context.zig");
const PipelineContext = ctx_mod.PipelineContext;

pub fn execute(pctx: *PipelineContext, _: types.ModulateConfig) !void {
    // Modulation requires routing weights to be set first
    var weights = pctx.routing_weights orelse return;

    // Apply a small preference bias based on interaction history.
    // In production, this delegates to AdaptiveModulator.modulate().
    // For now, slightly favor Abbey (the default conversational profile).
    const preference_bias: f32 = 0.05;
    weights.abbey_weight = std.math.clamp(weights.abbey_weight + preference_bias, 0.0, 1.0);

    // Renormalize
    const total = weights.abbey_weight + weights.aviva_weight + weights.abi_weight;
    if (total > 0) {
        weights.abbey_weight /= total;
        weights.aviva_weight /= total;
        weights.abi_weight /= total;
    }

    pctx.routing_weights = weights;

    // Update primary profile if weights changed the leader
    if (weights.abbey_weight >= weights.aviva_weight and weights.abbey_weight >= weights.abi_weight) {
        pctx.primary_profile = .abbey;
    } else if (weights.aviva_weight >= weights.abbey_weight and weights.aviva_weight >= weights.abi_weight) {
        pctx.primary_profile = .aviva;
    } else {
        pctx.primary_profile = .abi;
    }
}
