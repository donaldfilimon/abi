//! Validate Step — Constitution check.
//!
//! Evaluates the generated response against the 6-principle constitution.
//! When `feat_reasoning` is enabled, delegates to `Constitution.evaluate()`
//! for full principle-based evaluation. Otherwise falls back to keyword scan.
//! Sets `validation_passed` on the PipelineContext. When validation fails
//! and fallback is enabled, replaces the response with a safe default.

const std = @import("std");
const build_options = @import("build_options");
const types = @import("../types.zig");
const ctx_mod = @import("../context.zig");
const PipelineContext = ctx_mod.PipelineContext;

const constitution_mod = if (build_options.feat_reasoning)
    @import("../../constitution/mod.zig")
else
    @import("../../constitution/stub.zig");

/// Fallback keyword patterns used when Constitution is unavailable.
const unsafe_patterns = [_][]const u8{
    "harm",
    "dangerous",
    "illegal",
    "exploit",
};

pub fn execute(pctx: *PipelineContext, cfg: types.ValidateConfig) !void {
    const response = pctx.generated_response orelse return;

    var is_safe: bool = undefined;
    var score_overall: f32 = 1.0;

    if (cfg.target == .constitution and build_options.feat_reasoning) {
        // Full constitution evaluation via 6-principle engine
        const constitution = constitution_mod.Constitution.init();
        const score = constitution.evaluate(response);
        is_safe = score.isCompliant();
        score_overall = score.overall;

        // Store score in pipeline metadata
        try pctx.setMetadata("validation_score", try std.fmt.allocPrint(
            pctx.allocator,
            "{d:.2}",
            .{score_overall},
        ));
        try pctx.setMetadata("validation_violations", try std.fmt.allocPrint(
            pctx.allocator,
            "{d}",
            .{@as(u32, score.violation_count)},
        ));
    } else {
        // Fallback: simple keyword scan
        const lower = try toLower(pctx.allocator, response);
        defer pctx.allocator.free(lower);

        is_safe = true;
        for (unsafe_patterns) |pattern| {
            if (std.mem.indexOf(u8, lower, pattern) != null) {
                is_safe = false;
                break;
            }
        }
    }

    pctx.validation_passed = is_safe;

    if (!is_safe and cfg.fallback_on_failure) {
        const fallback = try pctx.allocator.dupe(
            u8,
            "I'm unable to provide that response as it may not align with safety guidelines.",
        );
        if (pctx.generated_response) |old| pctx.allocator.free(old);
        pctx.generated_response = fallback;
    }
}

fn toLower(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    const result = try allocator.alloc(u8, text.len);
    for (text, 0..) |c, i| {
        result[i] = if (c >= 'A' and c <= 'Z') c + 32 else c;
    }
    return result;
}
