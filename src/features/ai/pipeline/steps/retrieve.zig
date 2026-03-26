//! Retrieve Step — Pull context from WDBX block chain.
//!
//! Reads the k most recent conversation blocks, applies recency decay
//! scoring, and populates PipelineContext.context_fragments.

const std = @import("std");
const foundation = @import("../../../../foundation/mod.zig");
const types = @import("../types.zig");
const ctx_mod = @import("../context.zig");
const PipelineContext = ctx_mod.PipelineContext;
const PipelineError = types.PipelineError;

pub fn execute(pctx: *PipelineContext, cfg: types.RetrieveConfig) !void {
    const chain = pctx.chain orelse return;

    // Traverse backward through the chain to get recent block IDs
    const block_ids = chain.traverseBackward(cfg.k) catch {
        return;
    };
    defer pctx.allocator.free(block_ids);

    for (block_ids) |block_id| {
        const block = chain.getBlock(block_id) orelse continue;

        // Build a text representation of the block's content
        var buf: [256]u8 = undefined;
        const profile_name = switch (block.profile_tag.primary_profile) {
            .abbey => "Abbey",
            .aviva => "Aviva",
            .abi => "Abi",
            .blended => "Blended",
        };

        const text = std.fmt.bufPrint(&buf, "[{s}] (decay={d:.2})", .{
            profile_name,
            if (cfg.apply_recency_decay)
                block.getRecencyDecay(foundation.time.nowSeconds())
            else
                @as(f32, 1.0),
        }) catch continue;

        pctx.addFragment(text) catch continue;
    }
}
