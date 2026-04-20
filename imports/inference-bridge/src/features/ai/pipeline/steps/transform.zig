//! Transform Step — User-provided transformation.
//!
//! Applies a user-supplied function to the current generated response
//! (or rendered prompt if no response yet).

const std = @import("std");
const types = @import("../types.zig");
const ctx_mod = @import("../context.zig");
const PipelineContext = ctx_mod.PipelineContext;

pub fn execute(pctx: *PipelineContext, cfg: types.TransformConfig) !void {
    const text = pctx.generated_response orelse pctx.rendered_prompt orelse return;

    const transformed = try cfg.transform_fn(text, pctx.allocator);

    // Replace the response with the transformed text
    if (pctx.generated_response) |old| pctx.allocator.free(old);
    pctx.generated_response = transformed;
}

fn uppercaseTransform(text: []const u8, allocator: std.mem.Allocator) ![]const u8 {
    const result = try allocator.alloc(u8, text.len);
    for (text, 0..) |c, i| {
        result[i] = if (c >= 'a' and c <= 'z') c - 32 else c;
    }
    return result;
}

test "transform applies function to generated response" {
    const allocator = std.testing.allocator;
    var pctx = try PipelineContext.init(allocator, "hi", "session-1", 1);
    defer pctx.deinit();

    try pctx.setResponse("hello world");

    try execute(&pctx, .{ .transform_fn = &uppercaseTransform });

    try std.testing.expectEqualStrings("HELLO WORLD", pctx.generated_response.?);
}

test "transform falls back to rendered prompt" {
    const allocator = std.testing.allocator;
    var pctx = try PipelineContext.init(allocator, "hi", "session-2", 2);
    defer pctx.deinit();

    try pctx.setPrompt("rendered prompt");

    try execute(&pctx, .{ .transform_fn = &uppercaseTransform });

    try std.testing.expectEqualStrings("RENDERED PROMPT", pctx.generated_response.?);
}

test "transform skips when no response or prompt" {
    const allocator = std.testing.allocator;
    var pctx = try PipelineContext.init(allocator, "hi", "session-3", 3);
    defer pctx.deinit();

    // No response or rendered_prompt — should return without error
    try execute(&pctx, .{ .transform_fn = &uppercaseTransform });

    try std.testing.expect(pctx.generated_response == null);
}
