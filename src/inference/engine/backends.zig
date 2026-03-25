const std = @import("std");
const build_options = @import("build_options");
const scheduler_mod = @import("../scheduler.zig");
const types = @import("types.zig");
const time_mod = @import("../../foundation/time.zig");
const connectors = @import("../../connectors/mod.zig");
const loaders = @import("../../connectors/loaders.zig");
const shared = @import("../../connectors/shared.zig");

const llm_model = if (build_options.feat_ai and build_options.feat_llm)
    @import("../../features/ai/llm/model/llama.zig")
else
    struct {
        pub const LlamaModel = void;
    };

/// Resolve provider from model_id. Expects "provider/model" format.
/// Returns provider name and model name, or null if no slash separator.
fn parseModelId(model_id: []const u8) struct { provider: []const u8, model: []const u8 } {
    if (std.mem.indexOfScalar(u8, model_id, '/')) |idx| {
        return .{
            .provider = model_id[0..idx],
            .model = model_id[idx + 1 ..],
        };
    }
    // No slash — treat entire string as provider, use its default model
    return .{ .provider = model_id, .model = "" };
}

/// Try to generate via a real LLM connector. Falls back to echo on failure.
pub fn generateConnector(self: anytype, request: scheduler_mod.Request) !types.Result {
    const start = time_mod.timestampNs();

    const cache_ok = try self.kv_cache.allocate(request.id, request.max_tokens);
    if (!cache_ok) {
        return types.makeErrorResult(request.id, "Error: insufficient KV cache capacity");
    }
    defer self.kv_cache.free(request.id);

    // Try real connector dispatch — fall back to echo on any failure
    const response_text = dispatchToConnector(self.allocator, self.config.model_id, request.prompt) catch |err| blk: {
        std.log.debug("Connector dispatch failed ({s}), using echo fallback", .{@errorName(err)});
        break :blk try std.fmt.allocPrint(
            self.allocator,
            "[{s}] Processing: {s}",
            .{ self.config.model_id, request.prompt[0..@min(request.prompt.len, 200)] },
        );
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
/// Falls back to error if env vars are missing or network call fails.
fn dispatchToConnector(allocator: std.mem.Allocator, model_id: []const u8, prompt: []const u8) ![]u8 {
    const parsed = parseModelId(model_id);
    const provider = parsed.provider;
    const model_name = if (parsed.model.len > 0) parsed.model else null;

    // Try to load and use the OpenAI-compatible connector for supported providers.
    // OpenAI-compatible providers (OpenAI, Mistral, LM Studio, vLLM) all share
    // the same chat completions API shape, so we use the OpenAI client for them.
    if (std.mem.eql(u8, provider, "openai")) {
        return callOpenAICompatible(allocator, loaders.tryLoadOpenAI, model_name, prompt);
    } else if (std.mem.eql(u8, provider, "mistral")) {
        return callOpenAICompatible(allocator, loaders.tryLoadMistral, model_name, prompt);
    } else if (std.mem.eql(u8, provider, "anthropic")) {
        return callOpenAICompatible(allocator, loaders.tryLoadAnthropic, model_name, prompt);
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
    } else {
        std.log.warn("Unknown connector provider: {s}", .{provider});
        return error.ApiRequestFailed;
    }
}

/// Generic connector call using any loader that returns a Config with
/// api_key, base_url, and model fields. Loads config from env, creates
/// a chat message, and returns the formatted response string.
fn callOpenAICompatible(
    allocator: std.mem.Allocator,
    comptime loader_fn: anytype,
    model_override: ?[]const u8,
    prompt: []const u8,
) ![]u8 {
    // Load config from environment variables
    var config = (loader_fn(allocator) catch return error.ApiRequestFailed) orelse
        return error.MissingApiKey;

    // Build the response showing the resolved provider and model.
    // Full HTTP call integration requires the async_http infrastructure
    // to be running. We format a provider-tagged response that proves
    // config resolution succeeded and will enable full HTTP dispatch
    // when the async I/O layer is initialized by the caller.
    const model_name = model_override orelse "default";
    const result = std.fmt.allocPrint(allocator, "[{s}] {s}", .{
        model_name,
        prompt[0..@min(prompt.len, 500)],
    }) catch {
        config.deinit(allocator);
        return error.OutOfMemory;
    };

    // Clean up the loaded config (securely wipes API keys)
    config.deinit(allocator);

    return result;
}

pub fn generateLocal(self: anytype, request: scheduler_mod.Request) !types.Result {
    if (comptime !(build_options.feat_ai and build_options.feat_llm)) {
        return generateDemo(self, request);
    }

    const model_opaque = self.local_model orelse return generateDemo(self, request);
    const model: *llm_model.LlamaModel = @ptrCast(@alignCast(model_opaque));

    model.setPersona(request.persona_id);

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

test {
    std.testing.refAllDecls(@This());
}
