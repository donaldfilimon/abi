const std = @import("std");
const build_options = @import("build_options");
const scheduler_mod = @import("../scheduler.zig");
const types = @import("types.zig");
const time_mod = @import("../../foundation/time.zig");
const connectors = @import("../../connectors/mod.zig");
const loaders = @import("../../connectors/loaders.zig");
const shared = @import("../../connectors/shared.zig");
const async_http = @import("../../foundation/mod.zig").utils.async_http;
const anthropic = @import("../../connectors/anthropic.zig");

const llm_model = if (build_options.feat_ai and build_options.feat_llm)
    @import("../../features/ai/llm/model/llama.zig")
else
    struct {
        pub const LlamaModel = void;
    };

/// Known provider identifiers for model ID resolution.
pub const known_providers = [_][]const u8{
    "openai",
    "anthropic",
    "ollama",
    "mistral",
    "cohere",
    "gemini",
    "mlx",
    "huggingface",
    "lm_studio",
    "vllm",
    "llama_cpp",
    "codex",
};

/// Parsed model identifier — splits "provider/model" into components.
pub const ModelId = struct {
    /// Provider name (e.g. "openai"), or null if no slash separator found.
    provider: ?[]const u8,
    /// Model name (e.g. "gpt-4"), or the entire input if no slash found.
    model: []const u8,
    /// Whether the resolved provider is in the known_providers list.
    is_known_provider: bool,
};

/// Parse a model ID string in "provider/model" format.
pub fn parseModelId(model_id: []const u8) ModelId {
    const slash_pos = std.mem.indexOfScalar(u8, model_id, '/');
    if (slash_pos) |pos| {
        const provider = model_id[0..pos];
        const model = if (pos + 1 < model_id.len) model_id[pos + 1 ..] else "";
        return .{
            .provider = provider,
            .model = model,
            .is_known_provider = isKnownProvider(provider),
        };
    }
    return .{
        .provider = null,
        .model = model_id,
        .is_known_provider = false,
    };
}

/// Check whether a provider name is in the known_providers list.
pub fn isKnownProvider(name: []const u8) bool {
    for (known_providers) |p| {
        if (std.mem.eql(u8, name, p)) return true;
    }
    return false;
}

/// Try to generate via a real LLM connector.
pub fn generateConnector(self: anytype, request: scheduler_mod.Request) !types.Result {
    const start = time_mod.timestampNs();

    const cache_ok = try self.kv_cache.allocate(request.id, request.max_tokens);
    if (!cache_ok) {
        return types.makeErrorResult(request.id, "Error: insufficient KV cache capacity");
    }
    defer self.kv_cache.free(request.id);

    // Try real connector dispatch.
    const response_text = dispatchToConnector(self.allocator, self.config.model_id, request.prompt) catch |err| {
        std.log.err("connector: failed to dispatch to connector for model '{s}': {any}", .{ self.config.model_id, err });
        return err;
    };
    errdefer self.allocator.free(response_text);

    const end = time_mod.timestampNs();
    const elapsed_ns = end - start;
    const latency = types.elapsedNsToMs(elapsed_ns);
    const token_count: u32 = @intCast(@min(response_text.len / 4, std.math.maxInt(u32)));

    _ = self.total_requests.fetchAdd(1, .acq_rel);
    _ = self.total_tokens.fetchAdd(token_count, .acq_rel);
    _ = self.total_elapsed_ns.fetchAdd(elapsed_ns, .acq_rel);

    return types.Result{
        .id = request.id,
        .text = response_text,
        .text_owned = true,
        .tokens = &.{},
        .tokens_owned = false,
        .finish_reason = .stop,
        .prompt_tokens = @intCast(@min(request.prompt.len, std.math.maxInt(u32))),
        .completion_tokens = token_count,
        .latency_ms = latency,
        .ttft_ms = latency,
        .tokens_per_second = types.tokensPerSecond(token_count, elapsed_ns),
    };
}

/// Dispatch a prompt to a real LLM provider connector based on model_id.
/// Returns the response text (caller-owned) or an error.
///
/// Provider resolution order:
/// 1. Parse model_id as "provider/model" (e.g., "openai/gpt-4")
/// 2. Load provider config from environment variables
/// 3. Create client and make chat completion call
/// 4. Return response text (caller owns the allocation)
///
/// Returns an error if env vars are missing or network call fails.
fn dispatchToConnector(allocator: std.mem.Allocator, model_id: []const u8, prompt: []const u8) ![]u8 {
    const parsed = parseModelId(model_id);
    const provider = parsed.provider orelse {
        // No provider prefix — cannot resolve a connector without "provider/model" format
        return error.UnsupportedProvider;
    };
    const model_name = if (parsed.model.len > 0) parsed.model else null;

    // Try to load and use the OpenAI-compatible connector for supported providers.
    // OpenAI-compatible providers (OpenAI, Mistral, LM Studio, vLLM) all share
    // the same chat completions API shape, so we use the OpenAI client for them.
    if (std.mem.eql(u8, provider, "openai")) {
        return callOpenAICompatible(allocator, loaders.tryLoadOpenAI, model_name, prompt);
    } else if (std.mem.eql(u8, provider, "mistral")) {
        return callOpenAICompatible(allocator, loaders.tryLoadMistral, model_name, prompt);
    } else if (std.mem.eql(u8, provider, "anthropic")) {
        // Anthropic's Messages API is NOT OpenAI-compatible — uses x-api-key header,
        // content blocks response, and /messages endpoint instead of /chat/completions.
        return callAnthropicNative(allocator, model_name, prompt);
    } else if (std.mem.eql(u8, provider, "ollama")) {
        return callOpenAICompatible(allocator, loaders.tryLoadOllama, model_name, prompt);
    } else if (std.mem.eql(u8, provider, "cohere")) {
        return callOpenAICompatible(allocator, loaders.tryLoadCohere, model_name, prompt);
    } else if (std.mem.eql(u8, provider, "gemini")) {
        return callOpenAICompatible(allocator, loaders.tryLoadGemini, model_name, prompt);
    } else if (std.mem.eql(u8, provider, "mlx")) {
        return callOpenAICompatible(allocator, loaders.tryLoadMLX, model_name, prompt);
    } else if (std.mem.eql(u8, provider, "huggingface")) {
        return callOpenAICompatible(allocator, loaders.tryLoadHuggingFace, model_name, prompt);
    } else if (std.mem.eql(u8, provider, "lm_studio") or std.mem.eql(u8, provider, "lmstudio")) {
        return callOpenAICompatible(allocator, loaders.tryLoadLMStudio, model_name, prompt);
    } else if (std.mem.eql(u8, provider, "vllm")) {
        return callOpenAICompatible(allocator, loaders.tryLoadVLLM, model_name, prompt);
    } else if (std.mem.eql(u8, provider, "llama_cpp") or std.mem.eql(u8, provider, "llamacpp")) {
        return callOpenAICompatible(allocator, loaders.tryLoadLlamaCpp, model_name, prompt);
    } else if (std.mem.eql(u8, provider, "codex")) {
        return callOpenAICompatible(allocator, loaders.tryLoadCodex, model_name, prompt);
    } else {
        std.log.warn("Unknown connector provider: {s}", .{provider});
        return error.UnsupportedProvider;
    }
}

/// Generic connector call that loads config from env, creates an HTTP client,
/// makes a chat completion request, and returns the response text.
///
/// Uses comptime type inspection to handle two config shapes:
/// - OpenAI-style: `api_key: []u8` + `base_url: []u8` (openai, mistral, anthropic, cohere, etc.)
/// - OpenAICompat-style: `host: []u8` + `api_key: ?[]u8` (lm_studio, vllm, llama_cpp, ollama, mlx)
///
fn callOpenAICompatible(
    allocator: std.mem.Allocator,
    comptime loader_fn: anytype,
    model_override: ?[]const u8,
    prompt: []const u8,
) ![]u8 {
    // Load config from environment variables
    var config = loader_fn(allocator) catch |err| {
        std.log.warn("connector: loader failed for model override {any}: {s}", .{ model_override, @errorName(err) });
        return switch (err) {
            error.MissingApiKey, error.MissingApiToken => error.MissingApiKey,
            else => error.ApiRequestFailed,
        };
    } orelse return error.MissingApiKey;
    defer config.deinit(allocator);

    const ConfigType = @TypeOf(config);

    // Resolve model name: prefer override, then config default
    const model_name = model_override orelse config.model;

    // Build the chat completions URL based on config shape.
    // OpenAI-style uses "{base_url}/chat/completions" (base_url already includes /v1).
    // Host-style uses "{host}/v1/chat/completions".
    const url = if (comptime @hasField(ConfigType, "base_url"))
        std.fmt.allocPrint(allocator, "{s}/chat/completions", .{config.base_url}) catch return error.OutOfMemory
    else if (comptime @hasField(ConfigType, "host"))
        std.fmt.allocPrint(allocator, "{s}/v1/chat/completions", .{config.host}) catch return error.OutOfMemory
    else
        @compileError("callOpenAICompatible: config must have base_url or host field");
    defer allocator.free(url);

    // Resolve the API key from whichever field the config uses.
    // OpenAI/Mistral/Anthropic/Cohere/Gemini: api_key: []u8 (required)
    // HuggingFace: api_token: []u8 (required, different name)
    // LM Studio/vLLM/llama_cpp/MLX: api_key: ?[]u8 (optional)
    // Ollama: no api_key field at all
    const api_key: ?[]const u8 = if (comptime @hasField(ConfigType, "api_key"))
        // Works for both []u8 (coerces to ?[]const u8) and ?[]u8
        config.api_key
    else if (comptime @hasField(ConfigType, "api_token"))
        config.api_token // HuggingFace uses api_token
    else
        null;

    // Encode the chat request using the shared OpenAI-compatible format.
    const messages = [_]shared.ChatMessage{
        .{ .role = "user", .content = prompt },
    };
    const json_body = shared.openaiCompatEncodeChatRequest(allocator, .{
        .model = model_name,
        .messages = &messages,
    }) catch return error.OutOfMemory;
    defer allocator.free(json_body);

    // Initialize the async HTTP client.
    var http = try async_http.AsyncHttpClient.init(allocator);
    defer http.deinit();

    // Build and send the HTTP request.
    var http_req = async_http.HttpRequest.init(allocator, .post, url) catch
        return error.ApiRequestFailed;
    defer http_req.deinit();

    if (api_key) |key| {
        http_req.setBearerToken(key) catch return error.ApiRequestFailed;
    }
    http_req.setJsonBody(json_body) catch return error.ApiRequestFailed;

    var http_res = http.fetchJsonWithRetry(&http_req, shared.DEFAULT_RETRY_OPTIONS) catch |err| {
        std.log.warn("HTTP request failed: {s}", .{@errorName(err)});
        return error.ApiRequestFailed;
    };
    defer http_res.deinit();

    if (!http_res.isSuccess()) {
        std.log.warn("HTTP {d} from connector", .{http_res.status_code});
        if (http_res.status_code == 429) return error.ApiRequestFailed;
        return error.ApiRequestFailed;
    }

    // Decode the OpenAI-compatible chat response.
    var response = shared.openaiCompatDecodeChatResponse(allocator, http_res.body) catch |err| {
        std.log.warn("Response decode failed: {s}", .{@errorName(err)});
        return error.ApiRequestFailed;
    };
    defer shared.openaiCompatDeinitChatResponse(allocator, &response);

    if (response.choices.len == 0) return error.ApiRequestFailed;

    // Dupe the content so caller owns it — original is freed by deinitChatResponse.
    return allocator.dupe(u8, response.choices[0].message.content) catch return error.OutOfMemory;
}

fn callAnthropicNative(allocator: std.mem.Allocator, model_override: ?[]const u8, prompt: []const u8) ![]u8 {
    var config = (loaders.tryLoadAnthropic(allocator) catch return error.ApiRequestFailed) orelse
        return error.MissingApiKey;

    if (model_override) |override| {
        if (config.model_owned) allocator.free(@constCast(config.model));
        config.model = allocator.dupe(u8, override) catch return error.OutOfMemory;
        config.model_owned = true;
    }

    var client = try anthropic.Client.init(allocator, config);
    defer client.deinit();

    const response = client.chatSimple(prompt) catch |err| {
        std.log.warn("Anthropic API request failed: {s}", .{@errorName(err)});
        return error.ApiRequestFailed;
    };
    // Free response metadata — getResponseText extracts only the text content
    defer {
        allocator.free(response.id);
        allocator.free(response.type);
        allocator.free(response.role);
        allocator.free(response.model);
        if (response.stop_reason) |sr| allocator.free(sr);
        for (response.content) |block| {
            allocator.free(block.type);
            allocator.free(block.text);
        }
        allocator.free(response.content);
    }

    const text = client.getResponseText(response) catch return error.ApiRequestFailed;
    if (text.len == 0) {
        allocator.free(text);
        return error.ApiRequestFailed;
    }

    return text;
}

pub fn generateLocal(self: anytype, request: scheduler_mod.Request) !types.Result {
    if (comptime !(build_options.feat_ai and build_options.feat_llm)) {
        return generateDemo(self, request);
    }

    const model_opaque = self.local_model orelse return generateDemo(self, request);
    const model: *llm_model.LlamaModel = @ptrCast(@alignCast(model_opaque));

    model.setProfile(request.profile_id);

    const start = time_mod.timestampNs();

    const cache_ok = try self.kv_cache.allocate(request.id, request.max_tokens);
    if (!cache_ok) return types.makeErrorResult(request.id, "Error: insufficient KV cache capacity");
    defer self.kv_cache.free(request.id);

    const prompt_tokens = model.encode(request.prompt) catch
        return types.makeErrorResult(request.id, "Error: tokenizer failed");
    defer self.allocator.free(prompt_tokens);

    if (prompt_tokens.len == 0) return types.makeErrorResult(request.id, "Error: empty prompt");

    _ = model.forwardBatch(prompt_tokens, 0) catch
        return types.makeErrorResult(request.id, "Error: prefill failed");

    const tokens = try self.allocator.alloc(u32, request.max_tokens);
    var tokens_owned = true;
    defer if (tokens_owned) self.allocator.free(tokens);

    var local_sampler = self.sampler;
    local_sampler.params.temperature = request.temperature;
    local_sampler.params.top_p = request.top_p;
    local_sampler.params.top_k = request.top_k;

    var last_token: u32 = prompt_tokens[prompt_tokens.len - 1];
    var pos: u32 = @intCast(prompt_tokens.len);
    var actual_count: u32 = 0;
    var finish_reason: types.FinishReason = .length;

    for (0..request.max_tokens) |i| {
        const logits = model.forward(last_token, pos) catch
            return types.makeErrorResult(request.id, "Error: forward pass failed");
        const next_token = local_sampler.sample(logits);
        tokens[i] = next_token;
        actual_count += 1;
        if (next_token <= 2) {
            finish_reason = .stop;
            break;
        }
        last_token = next_token;
        pos += 1;
    }

    const text = model.decode(tokens[0..actual_count]) catch
        try std.fmt.allocPrint(self.allocator, "<{d} tokens generated>", .{actual_count});

    const elapsed_ns = time_mod.timestampNs() - start;
    const latency = types.elapsedNsToMs(elapsed_ns);

    _ = self.total_requests.fetchAdd(1, .acq_rel);
    _ = self.total_tokens.fetchAdd(actual_count, .acq_rel);
    _ = self.total_elapsed_ns.fetchAdd(elapsed_ns, .acq_rel);

    tokens_owned = false;
    return .{
        .id = request.id,
        .text = text,
        .text_owned = true,
        .tokens = tokens,
        .tokens_owned = true,
        .finish_reason = finish_reason,
        .prompt_tokens = @intCast(prompt_tokens.len),
        .completion_tokens = actual_count,
        .latency_ms = latency,
        .ttft_ms = latency,
        .tokens_per_second = types.tokensPerSecond(actual_count, elapsed_ns),
    };
}

pub fn generateDemo(self: anytype, request: scheduler_mod.Request) !types.Result {
    const start = time_mod.timestampNs();

    const cache_ok = try self.kv_cache.allocate(request.id, request.max_tokens);
    if (!cache_ok) {
        return types.makeErrorResult(request.id, "Error: insufficient KV cache capacity");
    }
    defer self.kv_cache.free(request.id);

    const num_tokens = request.max_tokens;
    const tokens = try self.allocator.alloc(u32, num_tokens);
    errdefer self.allocator.free(tokens);
    const vocab_n = @min(self.config.vocab_size, 256);
    var local_sampler = self.sampler;
    local_sampler.params.temperature = request.temperature;
    local_sampler.params.top_p = request.top_p;
    local_sampler.params.top_k = request.top_k;

    var prompt_hash: u64 = 0x517cc1b727220a95;
    for (request.prompt) |byte| {
        prompt_hash ^= @as(u64, byte);
        prompt_hash *%= 0x100000001b3;
    }

    var finish_reason: types.FinishReason = .length;
    var actual_count: u32 = 0;

    for (tokens, 0..) |*t, step_i| {
        var logits_buf: [256]f32 = undefined;
        const logits_slice = logits_buf[0..vocab_n];

        var step_seed = prompt_hash +% @as(u64, @intCast(step_i)) *% 0x9e3779b97f4a7c15;
        for (logits_slice, 0..) |*l, j| {
            const mixed = step_seed ^ (@as(u64, @intCast(j)) *% 0x517cc1b727220a95);
            const bits: u32 = @truncate(mixed >> 16);
            l.* = (@as(f32, @floatFromInt(bits % 1024)) / 256.0) - 2.0;
            step_seed = step_seed *% 0x100000001b3 +% @as(u64, @intCast(j));
        }

        if (step_i > num_tokens / 2) {
            if (vocab_n > 2) {
                logits_slice[1] += @as(f32, @floatFromInt(step_i)) * 0.02;
                logits_slice[2] += @as(f32, @floatFromInt(step_i)) * 0.015;
            }
        }

        t.* = local_sampler.sample(logits_slice);
        actual_count += 1;

        if (t.* <= 2 and step_i >= 4) {
            finish_reason = .stop;
            actual_count = @intCast(step_i + 1);
            break;
        }
    }

    var total_len: usize = 0;
    for (tokens[0..actual_count], 0..) |tok, i| {
        const word = types.demo_vocabulary[tok % types.demo_vocabulary.len];
        total_len += word.len;
        if (i + 1 < actual_count) total_len += 1;
    }

    const text_buf = try self.allocator.alloc(u8, total_len);
    var pos: usize = 0;
    for (tokens[0..actual_count], 0..) |tok, i| {
        const word = types.demo_vocabulary[tok % types.demo_vocabulary.len];
        @memcpy(text_buf[pos..][0..word.len], word);
        pos += word.len;
        if (i + 1 < actual_count) {
            text_buf[pos] = ' ';
            pos += 1;
        }
    }

    const end = time_mod.timestampNs();
    const elapsed_ns = end - start;
    const latency = types.elapsedNsToMs(elapsed_ns);
    const tps = types.tokensPerSecond(actual_count, elapsed_ns);

    _ = self.total_requests.fetchAdd(1, .acq_rel);
    _ = self.total_tokens.fetchAdd(actual_count, .acq_rel);
    _ = self.total_elapsed_ns.fetchAdd(elapsed_ns, .acq_rel);

    return types.Result{
        .id = request.id,
        .text = text_buf,
        .text_owned = true,
        .tokens = tokens,
        .tokens_owned = true,
        .finish_reason = finish_reason,
        .prompt_tokens = @intCast(@min(request.prompt.len, std.math.maxInt(u32))),
        .completion_tokens = actual_count,
        .latency_ms = latency,
        .ttft_ms = latency / @as(f32, @floatFromInt(@max(actual_count, 1))),
        .tokens_per_second = tps,
    };
}

test "parseModelId: provider/model format" {
    const result = parseModelId("openai/gpt-4");
    try std.testing.expectEqualStrings("openai", result.provider.?);
    try std.testing.expectEqualStrings("gpt-4", result.model);
    try std.testing.expect(result.is_known_provider);
}

test "parseModelId: no slash returns null provider" {
    const result = parseModelId("ollama");
    try std.testing.expect(result.provider == null);
    try std.testing.expectEqualStrings("ollama", result.model);
    try std.testing.expect(!result.is_known_provider);
}

test "parseModelId: lm_studio with underscore" {
    const result = parseModelId("lm_studio/phi-3");
    try std.testing.expectEqualStrings("lm_studio", result.provider.?);
    try std.testing.expectEqualStrings("phi-3", result.model);
    try std.testing.expect(result.is_known_provider);
}

test "parseModelId: empty string" {
    const result = parseModelId("");
    try std.testing.expect(result.provider == null);
    try std.testing.expectEqualStrings("", result.model);
    try std.testing.expect(!result.is_known_provider);
}

test "parseModelId: unknown provider" {
    const result = parseModelId("custom-provider/my-model");
    try std.testing.expectEqualStrings("custom-provider", result.provider.?);
    try std.testing.expectEqualStrings("my-model", result.model);
    try std.testing.expect(!result.is_known_provider);
}

test "parseModelId: trailing slash gives empty model" {
    const result = parseModelId("openai/");
    try std.testing.expectEqualStrings("openai", result.provider.?);
    try std.testing.expectEqualStrings("", result.model);
    try std.testing.expect(result.is_known_provider);
}

test "parseModelId: multiple slashes uses first" {
    const result = parseModelId("anthropic/claude-3/sonnet");
    try std.testing.expectEqualStrings("anthropic", result.provider.?);
    try std.testing.expectEqualStrings("claude-3/sonnet", result.model);
    try std.testing.expect(result.is_known_provider);
}

test "isKnownProvider: all known providers" {
    try std.testing.expect(isKnownProvider("openai"));
    try std.testing.expect(isKnownProvider("anthropic"));
    try std.testing.expect(isKnownProvider("ollama"));
    try std.testing.expect(isKnownProvider("mistral"));
    try std.testing.expect(isKnownProvider("cohere"));
    try std.testing.expect(isKnownProvider("gemini"));
    try std.testing.expect(isKnownProvider("mlx"));
    try std.testing.expect(isKnownProvider("huggingface"));
    try std.testing.expect(isKnownProvider("lm_studio"));
    try std.testing.expect(isKnownProvider("vllm"));
    try std.testing.expect(isKnownProvider("llama_cpp"));
    try std.testing.expect(isKnownProvider("codex"));
}

test "isKnownProvider: echo is no longer a provider" {
    try std.testing.expect(!isKnownProvider("echo"));
}

test "isKnownProvider: unknown returns false" {
    try std.testing.expect(!isKnownProvider("foobar"));
    try std.testing.expect(!isKnownProvider(""));
    try std.testing.expect(!isKnownProvider("OpenAI")); // case-sensitive
}

test "dispatchToConnector: unknown provider returns UnsupportedProvider" {
    const result = dispatchToConnector(std.testing.allocator, "unknown-provider/some-model", "hello");
    try std.testing.expectError(error.UnsupportedProvider, result);
}

test "dispatchToConnector: no slash returns UnsupportedProvider" {
    const result = dispatchToConnector(std.testing.allocator, "bare-model-name", "hello");
    try std.testing.expectError(error.UnsupportedProvider, result);
}

test "dispatchToConnector: missing API key returns MissingApiKey" {
    // All real providers will fail with MissingApiKey when no env vars are set.
    // This tests the codex path specifically (OpenAI-compatible, needs OPENAI_API_KEY).
    const result = dispatchToConnector(std.testing.allocator, "codex/code-davinci-002", "hello");
    // Without env vars, loader returns null -> MissingApiKey
    try std.testing.expectError(error.MissingApiKey, result);
}

test "dispatchToConnector: anthropic path returns MissingApiKey without env" {
    // Anthropic native path should fail gracefully without API key env vars
    const result = dispatchToConnector(std.testing.allocator, "anthropic/claude-3-5-sonnet-20241022", "hello");
    try std.testing.expectError(error.MissingApiKey, result);
}

test {
    std.testing.refAllDecls(@This());
}
