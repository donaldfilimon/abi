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
