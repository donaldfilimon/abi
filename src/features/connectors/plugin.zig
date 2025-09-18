const std = @import("std");
const plugins = @import("plugins");
const types = plugins.types;
const iface = plugins.interface;
const connectors = @import("mod.zig");

const PluginInfo = types.PluginInfo;
const PluginVersion = types.PluginVersion;
const PluginType = types.PluginType;
const PluginContext = types.PluginContext;

pub const EmbeddingApi = extern struct {
    // Returns 0 on success, negative on error
    // Allocates the output vector using context.allocator. Caller must free via free_vector
    embed_text: *const fn (context: *PluginContext, text_ptr: [*]const u8, text_len: usize, out_ptr: *[*]f32, out_len: *usize) callconv(.c) c_int,
    free_vector: *const fn (context: *PluginContext, ptr: [*]f32, len: usize) callconv(.c) void,
};

var EMBEDDING_API: EmbeddingApi = .{
    .embed_text = api_embed_text,
    .free_vector = api_free_vector,
};

const INFO = PluginInfo{
    .name = "embedding_connectors",
    .version = PluginVersion.init(1, 0, 0),
    .author = "Abi Team",
    .description = "Unified embedding connectors (Ollama/OpenAI) via plugin API",
    .plugin_type = .embedding_generator,
    .abi_version = iface.PLUGIN_ABI_VERSION,
};

fn get_info() callconv(.c) *const PluginInfo {
    return &INFO;
}

fn init(context: *PluginContext) callconv(.c) c_int {
    _ = context;
    // Nothing to initialize; configuration is read per-call from context.config
    return 0;
}

fn deinit(context: *PluginContext) callconv(.c) void {
    _ = context;
}

fn get_api(api_name: [*:0]const u8) callconv(.c) ?*anyopaque {
    const name = std.mem.span(api_name);
    if (std.mem.eql(u8, name, "embedding")) return &EMBEDDING_API;
    return null;
}

pub var PLUGIN_INTERFACE: iface.PluginInterface = .{
    .get_info = get_info,
    .init = init,
    .deinit = deinit,
    .get_api = get_api,
};

fn api_embed_text(ctx: *PluginContext, text_ptr: [*]const u8, text_len: usize, out_ptr: *[*]f32, out_len: *usize) callconv(.c) c_int {
    const allocator = ctx.allocator;
    const text = text_ptr[0..text_len];

    const cfg = ctx.config;
    const provider_name = cfg.getParameter("provider") orelse "ollama";

    var provider_config: connectors.ProviderConfig = undefined;
    if (std.mem.eql(u8, provider_name, "ollama")) {
        const host = cfg.getParameter("host") orelse (connectors.OllamaConfig{}).host;
        const model = cfg.getParameter("model") orelse (connectors.OllamaConfig{}).model;
        provider_config = .{ .ollama = .{ .host = host, .model = model } };
    } else if (std.mem.eql(u8, provider_name, "openai")) {
        const base_url = cfg.getParameter("base_url") orelse "https://api.openai.com/v1";
        const model = cfg.getParameter("model") orelse "text-embedding-3-small";
        const api_key = cfg.getParameter("api_key") orelse return -2; // Missing API key
        provider_config = .{ .openai = .{ .base_url = base_url, .api_key = api_key, .model = model } };
    } else {
        return -3; // Unknown provider
    }

    const vec = connectors.embedText(allocator, provider_config, text) catch return -1;
    out_ptr.* = vec.ptr;
    out_len.* = vec.len;
    return 0;
}

fn api_free_vector(ctx: *PluginContext, ptr: [*]f32, len: usize) callconv(.c) void {
    ctx.allocator.free(ptr[0..len]);
}

pub fn abi_plugin_create() callconv(.c) ?*const iface.PluginInterface {
    return &PLUGIN_INTERFACE;
}

pub fn getInterface() *const iface.PluginInterface {
    return &PLUGIN_INTERFACE;
}
