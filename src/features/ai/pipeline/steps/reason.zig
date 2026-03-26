//! Reason Step — Chain-of-thought reasoning.
//!
//! Adds a reasoning trace to the pipeline context, recording
//! the step-by-step analysis that led to the current state.
//! Sets a confidence score in metadata.

const std = @import("std");
const types = @import("../types.zig");
const ctx_mod = @import("../context.zig");
const PipelineContext = ctx_mod.PipelineContext;

pub fn execute(pctx: *PipelineContext, _: types.ReasonConfig) !void {
    // Build a reasoning summary from current pipeline state
    var reasoning = std.ArrayListUnmanaged(u8){};
    defer reasoning.deinit(pctx.allocator);

    try reasoning.appendSlice(pctx.allocator, "Reasoning trace:\n");

    // Step 1: Input analysis
    try std.fmt.format(reasoning.writer(pctx.allocator), "1. Input length: {d} chars\n", .{pctx.input.len});

    // Step 2: Context availability
    try std.fmt.format(reasoning.writer(pctx.allocator), "2. Context fragments: {d}\n", .{pctx.context_fragments.items.len});

    // Step 3: Routing decision
    if (pctx.primary_profile) |pp| {
        const name = switch (pp) {
            .abbey => "Abbey",
            .aviva => "Aviva",
            .abi => "Abi",
            .blended => "Blended",
        };
        try std.fmt.format(reasoning.writer(pctx.allocator), "3. Routed to: {s}\n", .{name});
    }

    // Step 4: Confidence estimate
    const confidence: f32 = if (pctx.routing_weights) |w| blk: {
        const max_w = @max(w.abbey_weight, @max(w.aviva_weight, w.abi_weight));
        break :blk max_w;
    } else 0.5;

    try std.fmt.format(reasoning.writer(pctx.allocator), "4. Confidence: {d:.2}\n", .{confidence});

    // Store reasoning as metadata
    try pctx.setMetadata("reasoning", reasoning.items);

    // Store confidence as metadata
    var conf_buf: [8]u8 = undefined;
    const conf_str = std.fmt.bufPrint(&conf_buf, "{d:.2}", .{confidence}) catch "0.50";
    try pctx.setMetadata("confidence", conf_str);
}
