const std = @import("std");
const build_options = @import("build_options");
const scheduler_mod = @import("../scheduler.zig");
const types = @import("types.zig");
const time_mod = @import("../../foundation/time.zig");

const llm_model = if (build_options.feat_ai and build_options.feat_llm)
    @import("../../features/ai/llm/model/llama.zig")
else
    struct {
        pub const LlamaModel = void;
    };

pub fn generateConnector(self: anytype, request: scheduler_mod.Request) !types.Result {
    const start = time_mod.timestampNs();

    const cache_ok = try self.kv_cache.allocate(request.id, request.max_tokens);
    if (!cache_ok) {
        return types.makeErrorResult(request.id, "Error: insufficient KV cache capacity");
    }
    defer self.kv_cache.free(request.id);

    // Connector bridge not yet wired — echo the prompt back
    const response_text = try std.fmt.allocPrint(
        self.allocator,
        "[{s}] Processing: {s}",
        .{ self.config.model_id, request.prompt[0..@min(request.prompt.len, 200)] },
    );
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
