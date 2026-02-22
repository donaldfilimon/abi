const std = @import("std");
const connectors = @import("../../../../services/connectors/mod.zig");
const types = @import("types.zig");
const plugins = @import("plugins/mod.zig");

pub fn isAvailable(
    allocator: std.mem.Allocator,
    provider: types.ProviderId,
    plugin_id: ?[]const u8,
) bool {
    return switch (provider) {
        .local_gguf => true,
        .llama_cpp => connectors.llama_cpp.isAvailable(),
        .mlx => connectors.mlx.isAvailable(),
        .ollama => connectors.ollama.isAvailable(),
        .lm_studio => connectors.lm_studio.isAvailable(),
        .vllm => connectors.vllm.isAvailable(),
        .anthropic => connectors.anthropic.isAvailable(),
        .openai => connectors.openai.isAvailable(),
        .codex => connectors.codex.isAvailable(),
        .opencode => connectors.opencode.isAvailable(),
        .claude => connectors.claude.isAvailable(),
        .gemini => connectors.gemini.isAvailable(),
        .ollama_passthrough => connectors.ollama_passthrough.isAvailable(),
        .plugin_http => blk: {
            if (plugin_id == null) break :blk plugins.loader.hasAnyEnabled(.http);
            const match = plugins.loader.findEnabledByKind(allocator, .http, plugin_id) catch null;
            if (match) |entry| {
                var owned = entry;
                defer owned.deinit(allocator);
                break :blk true;
            }
            break :blk false;
        },
        .plugin_native => blk: {
            if (plugin_id == null) break :blk plugins.loader.hasAnyEnabled(.native);
            const match = plugins.loader.findEnabledByKind(allocator, .native, plugin_id) catch null;
            if (match) |entry| {
                var owned = entry;
                defer owned.deinit(allocator);
                break :blk true;
            }
            break :blk false;
        },
    };
}

test "health checks support new providers" {
    const allocator = std.testing.allocator;
    _ = isAvailable(allocator, .codex, null);
    _ = isAvailable(allocator, .opencode, null);
    _ = isAvailable(allocator, .claude, null);
    _ = isAvailable(allocator, .gemini, null);
    _ = isAvailable(allocator, .ollama_passthrough, null);
}
