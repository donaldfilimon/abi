//! Route Step — Profile routing decision.
//!
//! Uses heuristic keyword matching to determine which profile
//! (Abbey/Aviva/Abi) should handle the request. Sets routing_weights
//! and primary_profile on the PipelineContext.

const std = @import("std");
const types = @import("../types.zig");
const ctx_mod = @import("../context.zig");
const PipelineContext = ctx_mod.PipelineContext;

/// Heuristic keyword sets for profile routing.
const abbey_keywords = [_][]const u8{ "feel", "think", "opinion", "help me", "explain", "understand", "why" };
const aviva_keywords = [_][]const u8{ "code", "function", "implement", "debug", "error", "compile", "api", "syntax" };
const abi_keywords = [_][]const u8{ "policy", "privacy", "comply", "regulate", "moderate", "safe", "filter" };

pub fn execute(pctx: *PipelineContext, _: types.RouteConfig) !void {
    const text = pctx.rendered_prompt orelse pctx.input;
    const lower = try toLower(pctx.allocator, text);
    defer pctx.allocator.free(lower);

    var abbey_score: f32 = 0.33;
    var aviva_score: f32 = 0.33;
    var abi_score: f32 = 0.34;

    // Score based on keyword matches
    for (abbey_keywords) |kw| {
        if (std.mem.indexOf(u8, lower, kw) != null) abbey_score += 0.15;
    }
    for (aviva_keywords) |kw| {
        if (std.mem.indexOf(u8, lower, kw) != null) aviva_score += 0.15;
    }
    for (abi_keywords) |kw| {
        if (std.mem.indexOf(u8, lower, kw) != null) abi_score += 0.15;
    }

    // Normalize to sum = 1.0
    const total = abbey_score + aviva_score + abi_score;
    if (total > 0) {
        abbey_score /= total;
        aviva_score /= total;
        abi_score /= total;
    }

    pctx.routing_weights = .{
        .abbey_weight = abbey_score,
        .aviva_weight = aviva_score,
        .abi_weight = abi_score,
    };

    // Determine primary profile
    if (abbey_score >= aviva_score and abbey_score >= abi_score) {
        pctx.primary_profile = .abbey;
    } else if (aviva_score >= abbey_score and aviva_score >= abi_score) {
        pctx.primary_profile = .aviva;
    } else {
        pctx.primary_profile = .abi;
    }
}

fn toLower(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    const result = try allocator.alloc(u8, text.len);
    for (text, 0..) |c, i| {
        result[i] = if (c >= 'A' and c <= 'Z') c + 32 else c;
    }
    return result;
}
