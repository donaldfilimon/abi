//! Generate Step — LLM inference.
//!
//! Produces a response using the rendered prompt and routing decision.
//! When no LLM backend is available, generates a structured echo response
//! that includes the profile and prompt information (demo mode).

const std = @import("std");
const types = @import("../types.zig");
const ctx_mod = @import("../context.zig");
const PipelineContext = ctx_mod.PipelineContext;

pub fn execute(pctx: *PipelineContext, _: types.GenerateConfig) !void {
    const prompt = pctx.rendered_prompt orelse pctx.input;

    // Determine profile name for response framing
    const profile_name = if (pctx.primary_profile) |pp| switch (pp) {
        .abbey => "Abbey",
        .aviva => "Aviva",
        .abi => "Abi",
        .blended => "Blended",
    } else "Abbey";

    // Demo mode: generate a structured echo response.
    // In production, this would delegate to ClientWrapper / inference engine.
    const response = try std.fmt.allocPrint(
        pctx.allocator,
        "[{s}] I've considered your request. Based on the context provided, " ++
            "here is my response to: \"{s}\"",
        .{ profile_name, truncate(prompt, 100) },
    );

    if (pctx.generated_response) |old| pctx.allocator.free(old);
    pctx.generated_response = response;
}

fn truncate(text: []const u8, max_len: usize) []const u8 {
    if (text.len <= max_len) return text;
    return text[0..max_len];
}
