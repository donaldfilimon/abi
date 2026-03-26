//! Generate Step — LLM inference.
//!
//! Produces a response using the rendered prompt and routing decision.
//! When `GenerateConfig.llm_client` is set, delegates to the Abbey
//! `ClientWrapper.complete()` for real LLM inference. Otherwise generates
//! a structured echo response (demo mode).

const std = @import("std");
const build_options = @import("build_options");
const types = @import("../types.zig");
const ctx_mod = @import("../context.zig");
const PipelineContext = ctx_mod.PipelineContext;

const client_mod = if (build_options.feat_reasoning)
    @import("../../abbey/client.zig")
else
    struct {};

pub fn execute(pctx: *PipelineContext, cfg: types.GenerateConfig) !void {
    const prompt = pctx.rendered_prompt orelse pctx.input;
    const profile_name = getProfileName(pctx);

    // Try real LLM client when provided and feature is enabled
    if (build_options.feat_reasoning) {
        if (cfg.llm_client) |opaque_client| {
            const wrapper: *client_mod.ClientWrapper = @ptrCast(@alignCast(opaque_client));
            if (wrapper.isAvailable()) {
                const messages = [_]client_mod.ChatMessage{.{
                    .role = "user",
                    .content = prompt,
                }};
                const request = client_mod.CompletionRequest{
                    .messages = &messages,
                    .temperature = cfg.temperature orelse 0.7,
                    .max_tokens = cfg.max_tokens orelse 2048,
                };
                if (wrapper.complete(request)) |resp| {
                    const duped = try pctx.allocator.dupe(u8, resp.content);
                    if (pctx.generated_response) |old| pctx.allocator.free(old);
                    pctx.generated_response = duped;
                    return;
                } else |_| {
                    // Fall through to demo mode on client error
                }
            }
        }
    }

    // Demo mode: structured echo response
    const response = try std.fmt.allocPrint(
        pctx.allocator,
        "[{s}] I've considered your request. Based on the context provided, " ++
            "here is my response to: \"{s}\"",
        .{ profile_name, truncate(prompt, 100) },
    );

    if (pctx.generated_response) |old| pctx.allocator.free(old);
    pctx.generated_response = response;
}

fn getProfileName(pctx: *const PipelineContext) []const u8 {
    return if (pctx.primary_profile) |pp| switch (pp) {
        .abbey => "Abbey",
        .aviva => "Aviva",
        .abi => "Abi",
        .blended => "Blended",
    } else "Abbey";
}

fn truncate(text: []const u8, max_len: usize) []const u8 {
    if (text.len <= max_len) return text;
    return text[0..max_len];
}
