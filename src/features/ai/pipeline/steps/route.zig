//! Route Step — Profile routing decision.
//!
//! Determines which profile (Abbey/Aviva/Abi) should handle the request
//! using heuristic keyword matching. Each profile has a set of trigger
//! keywords; matches increase that profile's weight. Weights are normalized
//! to sum to 1.0, and the highest-weight profile becomes `primary_profile`.
//!
//! Routing strategies (from `RouteConfig.strategy`):
//! - `.heuristic` / `.adaptive` — keyword scoring (current implementation)
//! - `.abi_backed` — reserved for Abi-driven routing via the real router

const std = @import("std");
const types = @import("../types.zig");
const ctx_mod = @import("../context.zig");
const PipelineContext = ctx_mod.PipelineContext;

/// Abbey: empathetic, conversational, emotionally aware.
const abbey_keywords = [_][]const u8{ "feel", "think", "opinion", "help me", "explain", "understand", "why" };
/// Aviva: technical, precise, code-focused.
const aviva_keywords = [_][]const u8{ "code", "function", "implement", "debug", "error", "compile", "api", "syntax" };
/// Abi: compliance, policy, safety-oriented.
const abi_keywords = [_][]const u8{ "policy", "privacy", "comply", "regulate", "moderate", "safe", "filter" };

pub fn execute(pctx: *PipelineContext, _: types.RouteConfig) !void {
    const text = pctx.rendered_prompt orelse pctx.input;
    const lower = try toLower(pctx.allocator, text);
    defer pctx.allocator.free(lower);

    var scores = [3]f32{ 0.33, 0.33, 0.34 }; // abbey, aviva, abi

    scores[0] += scoreKeywords(lower, &abbey_keywords);
    scores[1] += scoreKeywords(lower, &aviva_keywords);
    scores[2] += scoreKeywords(lower, &abi_keywords);

    // Normalize to sum = 1.0
    const total = scores[0] + scores[1] + scores[2];
    if (total > 0) {
        scores[0] /= total;
        scores[1] /= total;
        scores[2] /= total;
    }

    pctx.routing_weights = .{
        .abbey_weight = scores[0],
        .aviva_weight = scores[1],
        .abi_weight = scores[2],
    };

    pctx.primary_profile = getPrimaryProfile(scores);
}

/// Returns the profile tag for the highest-scoring profile.
fn getPrimaryProfile(scores: [3]f32) types.ProfileTag.ProfileType {
    if (scores[0] >= scores[1] and scores[0] >= scores[2]) return .abbey;
    if (scores[1] >= scores[0] and scores[1] >= scores[2]) return .aviva;
    return .abi;
}

/// Counts keyword matches and returns a weighted score.
fn scoreKeywords(text: []const u8, keywords: []const []const u8) f32 {
    var score: f32 = 0;
    for (keywords) |kw| {
        if (std.mem.indexOf(u8, text, kw) != null) score += 0.15;
    }
    return score;
}

fn toLower(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    const result = try allocator.alloc(u8, text.len);
    for (text, 0..) |c, i| {
        result[i] = if (c >= 'A' and c <= 'Z') c + 32 else c;
    }
    return result;
}
