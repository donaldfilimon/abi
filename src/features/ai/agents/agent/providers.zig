const std = @import("std");
const provider_router_mod = @import("../../llm/providers/router.zig");

pub fn generateProviderRouterResponse(
    agent: anytype,
    input: []const u8,
    allocator: std.mem.Allocator,
) ![]u8 {
    var result = try provider_router_mod.generate(allocator, .{
        .model = agent.config.model,
        .prompt = input,
        .backend = agent.config.provider_backend,
        .fallback = agent.config.provider_fallback,
        .strict_backend = agent.config.provider_strict_backend,
        .plugin_id = agent.config.provider_plugin_id,
        .max_tokens = agent.config.max_tokens,
        .temperature = agent.config.temperature,
        .top_p = agent.config.top_p,
    });
    defer result.deinit(allocator);

    return try allocator.dupe(u8, result.content);
}
