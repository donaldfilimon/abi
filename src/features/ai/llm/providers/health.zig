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
