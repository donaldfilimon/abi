//! Store Step — Persist pipeline state to WDBX block chain.
//!
//! Creates a comprehensive ConversationBlock from the accumulated
//! PipelineContext state with all routing metadata and embeddings.

const std = @import("std");
const types = @import("../types.zig");
const ctx_mod = @import("../context.zig");
const persistence = @import("../persistence.zig");
const PipelineContext = ctx_mod.PipelineContext;
const PipelineError = types.PipelineError;

pub fn execute(pctx: *PipelineContext, _: types.StoreConfig) !void {
    const chain = pctx.chain orelse return;

    // Build a full block config from current pipeline state.
    // Embeddings live on this stack frame; addBlock dupes them before return.
    var query_embedding = PipelineContext.hashEmbedding(pctx.input);
    var resp_embedding_buf: [4]f32 = undefined;
    const has_response = pctx.generated_response != null;
    if (pctx.generated_response) |r| {
        resp_embedding_buf = PipelineContext.hashEmbedding(r);
    }

    const profile_tag: types.ProfileTag = if (pctx.primary_profile) |pp|
        .{ .primary_profile = pp }
    else
        .{ .primary_profile = .abbey };

    const routing_weights = pctx.routing_weights orelse types.RoutingWeights{
        .abbey_weight = 1.0,
        .aviva_weight = 0.0,
        .abi_weight = 0.0,
    };

    // Get previous hash for chaining
    var previous_hash: [32]u8 = .{0} ** 32;
    if (chain.current_head) |head_id| {
        if (chain.blocks.get(head_id)) |head_block| {
            previous_hash = head_block.hash;
        }
    }

    const config = types.BlockConfig{
        .query_embedding = &query_embedding,
        .response_embedding = if (has_response) &resp_embedding_buf else null,
        .profile_tag = profile_tag,
        .routing_weights = routing_weights,
        .intent = if (!pctx.validation_passed) .safety_critical else .general,
        .pipeline_step = .store,
        .pipeline_id = pctx.pipeline_id,
        .step_index = pctx.current_step,
        .previous_hash = previous_hash,
    };

    const block_id = try chain.addBlock(config);
    try pctx.recordBlock(block_id);
}

test "store with nil chain returns immediately" {
    const allocator = std.testing.allocator;
    var pctx = try PipelineContext.init(allocator, "hello", "session-1", 1);
    defer pctx.deinit();

    // chain is null by default — should short-circuit
    try execute(&pctx, .{});

    try std.testing.expectEqual(@as(usize, 0), pctx.block_ids.items.len);
}
