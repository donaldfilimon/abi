const std = @import("std");
const llm = @import("../mod.zig");
const connectors = @import("../../../../services/connectors/mod.zig");
const types = @import("types.zig");
const errors = @import("errors.zig");
const registry = @import("registry.zig");
const health = @import("health.zig");
const plugins_mod = @import("plugins/mod.zig");

pub fn generate(allocator: std.mem.Allocator, cfg: types.GenerateConfig) !types.GenerateResult {
    if (cfg.model.len == 0) return errors.ProviderError.ModelRequired;
    if (cfg.prompt.len == 0) return errors.ProviderError.PromptRequired;

    const chain = try buildChain(allocator, cfg);
    defer allocator.free(chain);

    var last_err: anyerror = errors.ProviderError.NoProviderAvailable;

    for (chain) |provider| {
        if (!health.isAvailable(allocator, provider, cfg.plugin_id)) {
            last_err = errors.ProviderError.NotAvailable;
            if (cfg.strict_backend and cfg.backend != null and provider == cfg.backend.?) {
                return last_err;
            }
            continue;
        }

        const result = generateWithProvider(allocator, cfg, provider) catch |err| {
            last_err = err;
            if (cfg.strict_backend and cfg.backend != null and provider == cfg.backend.?) {
                return err;
            }
            continue;
        };

        return result;
    }

    if (cfg.strict_backend and cfg.backend != null) return last_err;
    return errors.ProviderError.NoProviderAvailable;
}

fn buildChain(allocator: std.mem.Allocator, cfg: types.GenerateConfig) ![]types.ProviderId {
    var list = std.ArrayListUnmanaged(types.ProviderId).empty;
    errdefer list.deinit(allocator);

    if (cfg.backend) |provider| {
        try appendUnique(allocator, &list, provider);

        if (!cfg.strict_backend) {
            for (cfg.fallback) |fallback_provider| {
                try appendUnique(allocator, &list, fallback_provider);
            }

            if (cfg.fallback.len == 0) {
                const defaults = if (registry.looksLikeModelPath(cfg.model))
                    registry.file_model_chain[0..]
                else
                    registry.model_name_chain[0..];

                for (defaults) |default_provider| {
                    try appendUnique(allocator, &list, default_provider);
                }
            }
        }
    } else {
        const defaults = if (registry.looksLikeModelPath(cfg.model))
            registry.file_model_chain[0..]
        else
            registry.model_name_chain[0..];

        for (defaults) |provider| {
            try appendUnique(allocator, &list, provider);
        }
    }

    return list.toOwnedSlice(allocator);
}

fn appendUnique(
    allocator: std.mem.Allocator,
    list: *std.ArrayListUnmanaged(types.ProviderId),
    provider: types.ProviderId,
) !void {
    for (list.items) |existing| {
        if (existing == provider) return;
    }

    try list.append(allocator, provider);
}

fn generateWithProvider(
    allocator: std.mem.Allocator,
    cfg: types.GenerateConfig,
    provider: types.ProviderId,
) !types.GenerateResult {
    return switch (provider) {
        .local_gguf => generateLocalGguf(allocator, cfg),
        .llama_cpp => generateLlamaCpp(allocator, cfg),
        .mlx => generateMlx(allocator, cfg),
        .ollama => generateOllama(allocator, cfg),
        .lm_studio => generateLmStudio(allocator, cfg),
        .vllm => generateVllm(allocator, cfg),
        .plugin_http => generateHttpPlugin(allocator, cfg),
        .plugin_native => generateNativePlugin(allocator, cfg),
    };
}

fn generateLocalGguf(allocator: std.mem.Allocator, cfg: types.GenerateConfig) !types.GenerateResult {
    var engine = llm.Engine.init(allocator, .{
        .max_new_tokens = cfg.max_tokens,
        .temperature = cfg.temperature,
        .top_p = cfg.top_p,
        .top_k = cfg.top_k,
        .repetition_penalty = cfg.repetition_penalty,
        .allow_ollama_fallback = false,
    });
    defer engine.deinit();

    try engine.loadModel(cfg.model);
    const text = try engine.generate(allocator, cfg.prompt);

    return .{
        .provider = .local_gguf,
        .model_used = try allocator.dupe(u8, cfg.model),
        .content = text,
    };
}

fn generateLlamaCpp(allocator: std.mem.Allocator, cfg: types.GenerateConfig) !types.GenerateResult {
    var client = try connectors.llama_cpp.createClient(allocator);
    defer client.deinit();

    try setConnectorModel(allocator, &client.config.model, &client.config.model_owned, cfg.model);

    const text = try client.generate(cfg.prompt, cfg.max_tokens);

    return .{
        .provider = .llama_cpp,
        .model_used = try allocator.dupe(u8, client.config.model),
        .content = text,
    };
}

fn generateMlx(allocator: std.mem.Allocator, cfg: types.GenerateConfig) !types.GenerateResult {
    var client = try connectors.mlx.createClient(allocator);
    defer client.deinit();

    try setConnectorModel(allocator, &client.config.model, &client.config.model_owned, cfg.model);

    const text = try client.generate(cfg.prompt, cfg.max_tokens);

    return .{
        .provider = .mlx,
        .model_used = try allocator.dupe(u8, client.config.model),
        .content = text,
    };
}

fn generateOllama(allocator: std.mem.Allocator, cfg: types.GenerateConfig) !types.GenerateResult {
    var client = try connectors.ollama.createClient(allocator);
    defer client.deinit();

    try setConnectorModel(allocator, &client.config.model, &client.config.model_owned, cfg.model);

    var response = try client.generate(.{
        .model = client.config.model,
        .prompt = cfg.prompt,
        .stream = false,
        .options = .{
            .temperature = cfg.temperature,
            .num_predict = cfg.max_tokens,
            .top_p = cfg.top_p,
            .top_k = cfg.top_k,
        },
    });
    defer response.deinit(allocator);

    return .{
        .provider = .ollama,
        .model_used = try allocator.dupe(u8, response.model),
        .content = try allocator.dupe(u8, response.response),
    };
}

fn generateLmStudio(allocator: std.mem.Allocator, cfg: types.GenerateConfig) !types.GenerateResult {
    var client = try connectors.lm_studio.createClient(allocator);
    defer client.deinit();

    try setConnectorModel(allocator, &client.config.model, &client.config.model_owned, cfg.model);

    const messages = [_]connectors.lm_studio.Message{
        .{ .role = "user", .content = cfg.prompt },
    };

    var response = try client.chatCompletion(.{
        .model = client.config.model,
        .messages = &messages,
        .temperature = cfg.temperature,
        .max_tokens = cfg.max_tokens,
        .top_p = cfg.top_p,
        .stream = false,
    });
    defer deinitLmStudioResponse(allocator, &response);

    if (response.choices.len == 0) return errors.ProviderError.GenerationFailed;

    return .{
        .provider = .lm_studio,
        .model_used = try allocator.dupe(u8, response.model),
        .content = try allocator.dupe(u8, response.choices[0].message.content),
    };
}

fn generateVllm(allocator: std.mem.Allocator, cfg: types.GenerateConfig) !types.GenerateResult {
    var client = try connectors.vllm.createClient(allocator);
    defer client.deinit();

    try setConnectorModel(allocator, &client.config.model, &client.config.model_owned, cfg.model);

    const messages = [_]connectors.vllm.Message{
        .{ .role = "user", .content = cfg.prompt },
    };

    var response = try client.chatCompletion(.{
        .model = client.config.model,
        .messages = &messages,
        .temperature = cfg.temperature,
        .max_tokens = cfg.max_tokens,
        .top_p = cfg.top_p,
        .stream = false,
    });
    defer deinitVllmResponse(allocator, &response);

    if (response.choices.len == 0) return errors.ProviderError.GenerationFailed;

    return .{
        .provider = .vllm,
        .model_used = try allocator.dupe(u8, response.model),
        .content = try allocator.dupe(u8, response.choices[0].message.content),
    };
}

fn generateHttpPlugin(allocator: std.mem.Allocator, cfg: types.GenerateConfig) !types.GenerateResult {
    const entry = try plugins_mod.loader.findEnabledByKind(allocator, .http, cfg.plugin_id) orelse return errors.ProviderError.PluginNotFound;
    defer {
        var mutable = entry;
        mutable.deinit(allocator);
    }
    return plugins_mod.http_plugin.generate(allocator, entry, cfg);
}

fn generateNativePlugin(allocator: std.mem.Allocator, cfg: types.GenerateConfig) !types.GenerateResult {
    const entry = try plugins_mod.loader.findEnabledByKind(allocator, .native, cfg.plugin_id) orelse return errors.ProviderError.PluginNotFound;
    defer {
        var mutable = entry;
        mutable.deinit(allocator);
    }
    return plugins_mod.native_plugin.generate(allocator, entry, cfg);
}

fn setConnectorModel(
    allocator: std.mem.Allocator,
    model_ptr: *[]const u8,
    model_owned_ptr: *bool,
    model: []const u8,
) !void {
    if (model.len == 0) return;

    if (model_owned_ptr.*) {
        allocator.free(@constCast(model_ptr.*));
    }

    model_ptr.* = try allocator.dupe(u8, model);
    model_owned_ptr.* = true;
}

fn deinitLmStudioResponse(allocator: std.mem.Allocator, response: *connectors.lm_studio.ChatCompletionResponse) void {
    allocator.free(response.id);
    allocator.free(response.model);
    for (response.choices) |*choice| {
        allocator.free(choice.message.role);
        allocator.free(choice.message.content);
        allocator.free(choice.finish_reason);
    }
    allocator.free(response.choices);
    response.* = undefined;
}

fn deinitVllmResponse(allocator: std.mem.Allocator, response: *connectors.vllm.ChatCompletionResponse) void {
    allocator.free(response.id);
    allocator.free(response.model);
    for (response.choices) |*choice| {
        allocator.free(choice.message.role);
        allocator.free(choice.message.content);
        allocator.free(choice.finish_reason);
    }
    allocator.free(response.choices);
    response.* = undefined;
}
