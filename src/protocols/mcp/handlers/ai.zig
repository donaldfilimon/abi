//! MCP AI Tool Handlers
//!
//! Handlers for `abi_chat` tool, providing access to the multi-profile
//! AI routing and inference pipeline.

const std = @import("std");
const build_options = @import("build_options");
const root = @import("../../../root.zig");

pub fn handleAbiChat(
    allocator: std.mem.Allocator,
    params: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    if (!build_options.feat_ai) {
        try out.appendSlice(allocator, "AI features are disabled in this build.");
        return;
    }
    if (!build_options.feat_inference) {
        try out.appendSlice(allocator, "Inference engine is disabled in this build.");
        return;
    }

    const p = params orelse return error.InvalidParams;
    const message_val = p.get("message") orelse return error.InvalidParams;
    if (message_val != .string) return error.InvalidParams;
    const message = message_val.string;

    // Route through multi-profile pipeline
    var registry = root.ai.profile.ProfileRegistry.init(allocator, .{});
    defer registry.deinit();

    var router = root.ai.profile.MultiProfileRouter.init(allocator, &registry, .{});
    defer router.deinit();

    const decision = router.route(message);

    // Run inference via engine
    var engine = try root.inference.Engine.init(allocator, .{
        .backend = .connector,
        .model_id = "ollama/llama3",
        .kv_cache_pages = 100,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 8,
    });
    defer engine.deinit();

    var result = try engine.generate(.{
        .id = 1,
        .prompt = message,
        .max_tokens = 512,
        .temperature = 0.7,
        .top_p = 0.9,
        .top_k = 40,
        .profile_id = @intFromEnum(decision.primary),
    });
    defer result.deinit(allocator);

    try out.appendSlice(allocator, result.text);
}

test {
    std.testing.refAllDecls(@This());
}
