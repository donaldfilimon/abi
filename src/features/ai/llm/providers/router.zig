const std = @import("std");
const llm = @import("../mod.zig");
const connectors = @import("../../../../services/connectors/mod.zig");
const types = @import("types.zig");
const errors = @import("errors.zig");
const registry = @import("registry.zig");
const model_profiles = @import("model_profiles.zig");
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
                // Insert profile-preferred providers first
                const profile_chain = model_profiles.getProviderChain(cfg.model);
                for (profile_chain) |profile_provider| {
                    try appendUnique(allocator, &list, profile_provider);
                }

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
        // Insert profile-preferred providers first (if a profile exists)
        const profile_chain = model_profiles.getProviderChain(cfg.model);
        for (profile_chain) |profile_provider| {
            try appendUnique(allocator, &list, profile_provider);
        }

        // Then append generic fallback chain
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
        .ollama_passthrough => generateOllamaPassthrough(allocator, cfg),
        .lm_studio => generateLmStudio(allocator, cfg),
        .vllm => generateVllm(allocator, cfg),
        .anthropic => generateAnthropic(allocator, cfg),
        .claude => generateClaude(allocator, cfg),
        .openai => generateOpenAI(allocator, cfg),
        .codex => generateCodex(allocator, cfg),
        .opencode => generateOpenCode(allocator, cfg),
        .gemini => generateGemini(allocator, cfg),
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

    // If structured messages are provided, use the chat endpoint
    if (cfg.messages) |chat_messages| {
        const connector_msgs = try convertToConnectorMessages(allocator, connectors.ollama.Message, chat_messages);
        defer allocator.free(connector_msgs);

        var chat_response = try client.chat(.{
            .model = client.config.model,
            .messages = connector_msgs,
            .stream = false,
        });
        defer chat_response.deinit(allocator);

        return .{
            .provider = .ollama,
            .model_used = try allocator.dupe(u8, chat_response.model),
            .content = try allocator.dupe(u8, chat_response.message.content),
        };
    }

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

    // Use structured messages if provided, otherwise wrap prompt
    const owned_msgs = if (cfg.messages) |chat_messages|
        try convertToConnectorMessages(allocator, connectors.lm_studio.Message, chat_messages)
    else
        null;
    defer if (owned_msgs) |m| allocator.free(m);

    const default_msgs = [_]connectors.lm_studio.Message{
        .{ .role = "user", .content = cfg.prompt },
    };
    const messages: []const connectors.lm_studio.Message = owned_msgs orelse &default_msgs;

    var response = try client.chatCompletion(.{
        .model = client.config.model,
        .messages = messages,
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

    // Use structured messages if provided, otherwise wrap prompt
    const owned_msgs = if (cfg.messages) |chat_messages|
        try convertToConnectorMessages(allocator, connectors.vllm.Message, chat_messages)
    else
        null;
    defer if (owned_msgs) |m| allocator.free(m);

    const default_msgs = [_]connectors.vllm.Message{
        .{ .role = "user", .content = cfg.prompt },
    };
    const messages: []const connectors.vllm.Message = owned_msgs orelse &default_msgs;

    var response = try client.chatCompletion(.{
        .model = client.config.model,
        .messages = messages,
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

fn generateOllamaPassthrough(allocator: std.mem.Allocator, cfg: types.GenerateConfig) !types.GenerateResult {
    var client = try connectors.ollama_passthrough.createClient(allocator);
    defer client.deinit();

    try setConnectorModel(allocator, &client.config.model, &client.config.model_owned, cfg.model);

    const owned_msgs = if (cfg.messages) |chat_messages|
        try convertToConnectorMessages(allocator, connectors.ollama_passthrough.Message, chat_messages)
    else
        null;
    defer if (owned_msgs) |m| allocator.free(m);

    const default_msgs = [_]connectors.ollama_passthrough.Message{
        .{ .role = "user", .content = cfg.prompt },
    };
    const messages: []const connectors.ollama_passthrough.Message = owned_msgs orelse &default_msgs;

    var response = try client.chatCompletion(.{
        .model = client.config.model,
        .messages = messages,
        .temperature = cfg.temperature,
        .max_tokens = cfg.max_tokens,
        .top_p = cfg.top_p,
        .stream = false,
    });
    defer deinitVllmResponse(allocator, &response);

    if (response.choices.len == 0) return errors.ProviderError.GenerationFailed;

    return .{
        .provider = .ollama_passthrough,
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

    const new_model = try allocator.dupe(u8, model);

    if (model_owned_ptr.*) {
        allocator.free(@constCast(model_ptr.*));
    }

    model_ptr.* = new_model;
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

fn deinitAnthropicResponse(allocator: std.mem.Allocator, response: *connectors.anthropic.MessagesResponse) void {
    allocator.free(response.id);
    allocator.free(response.type);
    allocator.free(response.role);
    allocator.free(response.model);
    if (response.stop_reason) |sr| allocator.free(sr);
    for (response.content) |*block| {
        allocator.free(block.type);
        allocator.free(block.text);
    }
    allocator.free(response.content);
    response.* = undefined;
}

fn deinitOpenAIResponse(allocator: std.mem.Allocator, response: *connectors.openai.ChatCompletionResponse) void {
    allocator.free(response.id);
    allocator.free(response.object);
    allocator.free(response.model);
    for (response.choices) |*choice| {
        allocator.free(choice.message.role);
        allocator.free(choice.message.content);
        allocator.free(choice.finish_reason);
    }
    allocator.free(response.choices);
    response.* = undefined;
}

fn generateAnthropic(allocator: std.mem.Allocator, cfg: types.GenerateConfig) !types.GenerateResult {
    var client = try connectors.anthropic.createClient(allocator);
    defer client.deinit();

    try setConnectorModel(allocator, &client.config.model, &client.config.model_owned, cfg.model);

    var response: connectors.anthropic.MessagesResponse = undefined;

    if (cfg.messages) |chat_messages| {
        const connector_msgs = try convertToConnectorMessages(allocator, connectors.anthropic.Message, chat_messages);
        defer allocator.free(connector_msgs);

        if (cfg.system_prompt) |system| {
            response = try client.chatWithSystem(system, connector_msgs);
        } else {
            response = try client.chat(connector_msgs);
        }
    } else {
        if (cfg.system_prompt) |system| {
            const msgs = [_]connectors.anthropic.Message{
                .{ .role = "user", .content = cfg.prompt },
            };
            response = try client.chatWithSystem(system, &msgs);
        } else {
            response = try client.chatSimple(cfg.prompt);
        }
    }
    defer deinitAnthropicResponse(allocator, &response);

    const text = try client.getResponseText(response);
    errdefer allocator.free(text);

    return .{
        .provider = .anthropic,
        .model_used = try allocator.dupe(u8, response.model),
        .content = text,
    };
}

fn generateClaude(allocator: std.mem.Allocator, cfg: types.GenerateConfig) !types.GenerateResult {
    var client = try connectors.claude.createClient(allocator);
    defer client.deinit();

    try setConnectorModel(allocator, &client.config.model, &client.config.model_owned, cfg.model);

    var response: connectors.claude.MessagesResponse = undefined;

    if (cfg.messages) |chat_messages| {
        const connector_msgs = try convertToConnectorMessages(allocator, connectors.claude.Message, chat_messages);
        defer allocator.free(connector_msgs);

        if (cfg.system_prompt) |system| {
            response = try client.chatWithSystem(system, connector_msgs);
        } else {
            response = try client.chat(connector_msgs);
        }
    } else {
        if (cfg.system_prompt) |system| {
            const msgs = [_]connectors.claude.Message{
                .{ .role = "user", .content = cfg.prompt },
            };
            response = try client.chatWithSystem(system, &msgs);
        } else {
            response = try client.chatSimple(cfg.prompt);
        }
    }
    defer deinitAnthropicResponse(allocator, &response);

    const text = try client.getResponseText(response);
    errdefer allocator.free(text);

    return .{
        .provider = .claude,
        .model_used = try allocator.dupe(u8, response.model),
        .content = text,
    };
}

fn generateOpenAI(allocator: std.mem.Allocator, cfg: types.GenerateConfig) !types.GenerateResult {
    var client = try connectors.openai.createClient(allocator);
    defer client.deinit();

    try setConnectorModel(allocator, &client.config.model, &client.config.model_owned, cfg.model);

    var response: connectors.openai.ChatCompletionResponse = undefined;

    if (cfg.messages) |chat_messages| {
        // Build message list; prepend system prompt if provided
        var msg_list = std.ArrayListUnmanaged(connectors.openai.Message).empty;
        defer msg_list.deinit(allocator);

        if (cfg.system_prompt) |system| {
            try msg_list.append(allocator, .{ .role = "system", .content = system });
        }
        for (chat_messages) |msg| {
            try msg_list.append(allocator, .{ .role = msg.role, .content = msg.content });
        }

        response = try client.chatCompletion(.{
            .model = client.config.model,
            .messages = msg_list.items,
            .temperature = cfg.temperature,
            .max_tokens = cfg.max_tokens,
            .stream = false,
        });
    } else {
        if (cfg.system_prompt) |system| {
            var msgs = [_]connectors.openai.Message{
                .{ .role = "system", .content = system },
                .{ .role = "user", .content = cfg.prompt },
            };
            response = try client.chatCompletion(.{
                .model = client.config.model,
                .messages = &msgs,
                .temperature = cfg.temperature,
                .max_tokens = cfg.max_tokens,
                .stream = false,
            });
        } else {
            response = try client.chatSimple(cfg.prompt);
        }
    }
    defer deinitOpenAIResponse(allocator, &response);

    if (response.choices.len == 0) return errors.ProviderError.GenerationFailed;

    return .{
        .provider = .openai,
        .model_used = try allocator.dupe(u8, response.model),
        .content = try allocator.dupe(u8, response.choices[0].message.content),
    };
}

fn generateCodex(allocator: std.mem.Allocator, cfg: types.GenerateConfig) !types.GenerateResult {
    var client = try connectors.codex.createClient(allocator);
    defer client.deinit();

    try setConnectorModel(allocator, &client.config.model, &client.config.model_owned, cfg.model);

    var response: connectors.codex.ChatCompletionResponse = undefined;

    if (cfg.messages) |chat_messages| {
        var msg_list = std.ArrayListUnmanaged(connectors.codex.Message).empty;
        defer msg_list.deinit(allocator);

        if (cfg.system_prompt) |system| {
            try msg_list.append(allocator, .{ .role = "system", .content = system });
        }
        for (chat_messages) |msg| {
            try msg_list.append(allocator, .{ .role = msg.role, .content = msg.content });
        }

        response = try client.chatCompletion(.{
            .model = client.config.model,
            .messages = msg_list.items,
            .temperature = cfg.temperature,
            .max_tokens = cfg.max_tokens,
            .stream = false,
        });
    } else {
        if (cfg.system_prompt) |system| {
            var msgs = [_]connectors.codex.Message{
                .{ .role = "system", .content = system },
                .{ .role = "user", .content = cfg.prompt },
            };
            response = try client.chatCompletion(.{
                .model = client.config.model,
                .messages = &msgs,
                .temperature = cfg.temperature,
                .max_tokens = cfg.max_tokens,
                .stream = false,
            });
        } else {
            response = try client.chatSimple(cfg.prompt);
        }
    }
    defer deinitOpenAIResponse(allocator, &response);

    if (response.choices.len == 0) return errors.ProviderError.GenerationFailed;

    return .{
        .provider = .codex,
        .model_used = try allocator.dupe(u8, response.model),
        .content = try allocator.dupe(u8, response.choices[0].message.content),
    };
}

fn generateOpenCode(allocator: std.mem.Allocator, cfg: types.GenerateConfig) !types.GenerateResult {
    var client = try connectors.opencode.createClient(allocator);
    defer client.deinit();

    try setConnectorModel(allocator, &client.config.model, &client.config.model_owned, cfg.model);

    var response: connectors.opencode.ChatCompletionResponse = undefined;

    if (cfg.messages) |chat_messages| {
        var msg_list = std.ArrayListUnmanaged(connectors.opencode.Message).empty;
        defer msg_list.deinit(allocator);

        if (cfg.system_prompt) |system| {
            try msg_list.append(allocator, .{ .role = "system", .content = system });
        }
        for (chat_messages) |msg| {
            try msg_list.append(allocator, .{ .role = msg.role, .content = msg.content });
        }

        response = try client.chatCompletion(.{
            .model = client.config.model,
            .messages = msg_list.items,
            .temperature = cfg.temperature,
            .max_tokens = cfg.max_tokens,
            .stream = false,
        });
    } else {
        if (cfg.system_prompt) |system| {
            var msgs = [_]connectors.opencode.Message{
                .{ .role = "system", .content = system },
                .{ .role = "user", .content = cfg.prompt },
            };
            response = try client.chatCompletion(.{
                .model = client.config.model,
                .messages = &msgs,
                .temperature = cfg.temperature,
                .max_tokens = cfg.max_tokens,
                .stream = false,
            });
        } else {
            response = try client.chatSimple(cfg.prompt);
        }
    }
    defer deinitOpenAIResponse(allocator, &response);

    if (response.choices.len == 0) return errors.ProviderError.GenerationFailed;

    return .{
        .provider = .opencode,
        .model_used = try allocator.dupe(u8, response.model),
        .content = try allocator.dupe(u8, response.choices[0].message.content),
    };
}

fn generateGemini(allocator: std.mem.Allocator, cfg: types.GenerateConfig) !types.GenerateResult {
    var client = try connectors.gemini.createClient(allocator);
    defer client.deinit();

    try setConnectorModel(allocator, &client.config.model, &client.config.model_owned, cfg.model);

    const owned_msgs = if (cfg.messages) |chat_messages|
        try convertToConnectorMessages(allocator, connectors.gemini.Message, chat_messages)
    else
        null;
    defer if (owned_msgs) |m| allocator.free(m);

    const default_msgs = [_]connectors.gemini.Message{
        .{ .role = "user", .content = cfg.prompt },
    };
    const messages: []const connectors.gemini.Message = owned_msgs orelse &default_msgs;

    var response = try client.generate(.{
        .model = client.config.model,
        .messages = messages,
        .temperature = cfg.temperature,
        .max_output_tokens = cfg.max_tokens,
        .top_p = cfg.top_p,
        .system_prompt = cfg.system_prompt,
    });
    defer response.deinit(allocator);

    return .{
        .provider = .gemini,
        .model_used = try allocator.dupe(u8, response.model),
        .content = try allocator.dupe(u8, response.text),
    };
}

/// Convert provider-agnostic ChatMessage slice to a connector-native Message slice.
/// Both types have the same { role, content } layout, so this is a field-by-field copy.
/// Caller owns the returned slice and must free it with allocator.free().
fn convertToConnectorMessages(
    allocator: std.mem.Allocator,
    comptime ConnectorMessage: type,
    chat_messages: []const types.ChatMessage,
) ![]const ConnectorMessage {
    const msgs = try allocator.alloc(ConnectorMessage, chat_messages.len);
    for (chat_messages, 0..) |msg, i| {
        msgs[i] = .{ .role = msg.role, .content = msg.content };
    }
    return msgs;
}
