//! Turns candidate sets into useful memory.

const std = @import("std");

pub const WeightProfile = struct {
    semantic_similarity: f32 = 1.0,
    recency: f32 = 1.0,
    trust_score: f32 = 1.0,
    user_pinning: f32 = 1.0,
    project_locality: f32 = 1.0,
    persona_preference: f32 = 1.0,
    contradiction_penalty: f32 = 1.0,
    past_usefulness: f32 = 1.0,
};

pub const Scorer = struct {
    // TODO: Output is a ranked candidate list plus score trace
};
