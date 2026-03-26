//! Validate Step — Constitution check.
//!
//! Evaluates the generated response against the 6-principle constitution.
//! Sets validation_passed on the PipelineContext. When validation fails
//! and fallback is enabled, replaces the response with a safe default.

const std = @import("std");
const types = @import("../types.zig");
const ctx_mod = @import("../context.zig");
const PipelineContext = ctx_mod.PipelineContext;

/// Basic content safety checks (placeholder for full constitution integration).
const unsafe_patterns = [_][]const u8{
    "harm",
    "dangerous",
    "illegal",
    "exploit",
};

pub fn execute(pctx: *PipelineContext, cfg: types.ValidateConfig) !void {
    const response = pctx.generated_response orelse return;

    // Simple safety scan — in production, delegates to Constitution.validate()
    var is_safe = true;
    const lower = try toLower(pctx.allocator, response);
    defer pctx.allocator.free(lower);

    for (unsafe_patterns) |pattern| {
        if (std.mem.indexOf(u8, lower, pattern) != null) {
            is_safe = false;
            break;
        }
    }

    pctx.validation_passed = is_safe;

    if (!is_safe and cfg.target == .constitution) {
        // Replace with safe fallback if configured
        if (cfg.fallback_on_failure) {
            const fallback = try pctx.allocator.dupe(
                u8,
                "I'm unable to provide that response as it may not align with safety guidelines.",
            );
            if (pctx.generated_response) |old| pctx.allocator.free(old);
            pctx.generated_response = fallback;
        }
    }
}

fn toLower(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    const result = try allocator.alloc(u8, text.len);
    for (text, 0..) |c, i| {
        result[i] = if (c >= 'A' and c <= 'Z') c + 32 else c;
    }
    return result;
}
