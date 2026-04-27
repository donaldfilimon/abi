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
                if (wrapper.complete(request)) |resp_val| {
                    var resp = resp_val;
                    defer resp.deinit(pctx.allocator);
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

test "generate demo mode produces echo response" {
    const allocator = std.testing.allocator;
    var pctx = try PipelineContext.init(allocator, "What is Zig?", "session-1", 1);
    defer pctx.deinit();

    // No llm_client set — falls through to demo mode
    try execute(&pctx, .{});

    const response = pctx.generated_response.?;
    // Default profile is Abbey when primary_profile is null
    try std.testing.expect(std.mem.startsWith(u8, response, "[Abbey]"));
    try std.testing.expect(std.mem.indexOf(u8, response, "What is Zig?") != null);
}

test "generate uses rendered prompt when available" {
    const allocator = std.testing.allocator;
    var pctx = try PipelineContext.init(allocator, "raw input", "session-2", 2);
    defer pctx.deinit();

    try pctx.setPrompt("Rendered: please answer");

    try execute(&pctx, .{});

    const response = pctx.generated_response.?;
    try std.testing.expect(std.mem.indexOf(u8, response, "Rendered: please answer") != null);
}

test "generate uses correct profile name" {
    const allocator = std.testing.allocator;
    var pctx = try PipelineContext.init(allocator, "hello", "session-3", 3);
    defer pctx.deinit();

    pctx.primary_profile = .aviva;

    try execute(&pctx, .{});

    const response = pctx.generated_response.?;
    try std.testing.expect(std.mem.startsWith(u8, response, "[Aviva]"));
}

test "generate truncates long prompts in demo response" {
    const allocator = std.testing.allocator;
    // Create a prompt longer than 100 chars
    const long_input = "a" ** 150;
    var pctx = try PipelineContext.init(allocator, long_input, "session-4", 4);
    defer pctx.deinit();

    try execute(&pctx, .{});

    const response = pctx.generated_response.?;
    // The truncated portion should be at most 100 chars of 'a'
    try std.testing.expect(response.len > 0);
    // Full 150-char input should NOT appear in the response
    try std.testing.expect(std.mem.indexOf(u8, response, long_input) == null);
}

test "truncate helper" {
    try std.testing.expectEqualStrings("hello", truncate("hello", 10));
    try std.testing.expectEqualStrings("hel", truncate("hello", 3));
    try std.testing.expectEqualStrings("", truncate("", 5));
}
