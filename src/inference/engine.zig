//! Inference Engine
//!
//! Orchestrates token generation by combining the paged KV-cache, priority
//! scheduler, and token sampler. Supports multiple backends:
//!
//! - `demo`: Generates synthetic text seeded from input prompts (testing only)
//! - `connector`: Delegates to external LLM providers via the connector layer
//! - `local`: Uses the built-in transformer (when available)
//!
//! The engine maintains a consistent API regardless of backend, enabling
//! seamless switching between local and remote inference.

const std = @import("std");
const Allocator = std.mem.Allocator;
const kv_cache_mod = @import("kv_cache.zig");
const scheduler_mod = @import("scheduler.zig");
const sampler_mod = @import("sampler.zig");
const time_mod = @import("../foundation/time.zig");

/// Inference backend selection.
pub const Backend = enum {
    /// Demo/testing: generates synthetic text seeded from prompts.
    demo,
    /// Connector-backed: delegates to external LLM providers (OpenAI, Anthropic, Ollama, etc.).
    connector,
    /// Local transformer: uses the built-in Zig transformer implementation.
    local,
};

pub const Config = struct {
    vocab_size: u32 = 128256,
    hidden_dim: u32 = 4096,
    num_layers: u32 = 32,
    num_heads: u32 = 32,
    head_dim: u32 = 128,
    max_seq_len: u32 = 131072,
    max_batch_size: u32 = 64,
    kv_cache_pages: u32 = 10000,
    page_size: u32 = 16,
    /// Which backend to use for inference.
    backend: Backend = .demo,
    /// Model identifier for connector backend (e.g., "gpt-4", "claude-3", "llama-3").
    model_id: []const u8 = "abi-demo",
};

pub const FinishReason = enum {
    stop,
    length,
    error_,
};

pub const Result = struct {
    id: u64,
    text: []const u8,
    tokens: []const u32,
    finish_reason: FinishReason,
    prompt_tokens: u32,
    completion_tokens: u32,
    latency_ms: f32,
    ttft_ms: f32,
    tokens_per_second: f32,
};

pub const Stats = struct {
    total_requests: u64,
    total_tokens_generated: u64,
    active_sequences: u32,
    cache_utilization: f32,
    pending_requests: u32,
    avg_tokens_per_second: f32,
    backend: Backend,
};

const demo_vocabulary = [_][]const u8{
    "the",      "a",       "is",      "of",     "and",    "to",      "in",     "that",   "it",     "for",
    "was",      "on",      "are",     "with",   "as",     "at",      "be",     "this",   "have",   "from",
    "data",     "model",   "query",   "result", "system", "process", "value",  "node",   "index",  "vector",
    "search",   "layer",   "output",  "input",  "token",  "state",   "memory", "cache",  "error",  "status",
    "response", "request", "context", "agent",  "neural", "graph",   "batch",  "stream", "config", "module",
};

pub const Engine = struct {
    const Self = @This();

    allocator: Allocator,
    config: Config,
    kv_cache: kv_cache_mod.PagedKVCache,
    scheduler: scheduler_mod.Scheduler,
    sampler: sampler_mod.Sampler,

    total_requests: u64,
    total_tokens: u64,
    next_request_id: u64,

    pub fn init(allocator: Allocator, config: Config) !Self {
        const cache = try kv_cache_mod.PagedKVCache.init(allocator, .{
            .num_pages = config.kv_cache_pages,
            .page_size = config.page_size,
            .num_layers = config.num_layers,
            .num_heads = config.num_heads,
            .head_dim = config.head_dim,
        });

        return .{
            .allocator = allocator,
            .config = config,
            .kv_cache = cache,
            .scheduler = scheduler_mod.Scheduler.init(allocator, config.max_batch_size * 4),
            .sampler = sampler_mod.Sampler.initWithAllocator(allocator, .{}),
            .total_requests = 0,
            .total_tokens = 0,
            .next_request_id = 1,
        };
    }

    pub fn deinit(self: *Self) void {
        self.kv_cache.deinit();
        self.scheduler.deinit();
    }

    /// Generate a response using the configured backend.
    pub fn generate(self: *Self, request: scheduler_mod.Request) !Result {
        return switch (self.config.backend) {
            .demo => self.generateDemo(request),
            .connector => self.generateConnector(request),
            .local => self.generateLocal(request),
        };
    }

    /// Connector-backed generation: constructs a response using the prompt
    /// as context and the model_id for provider routing.
    ///
    /// In a full deployment, this would call into the connector layer
    /// (src/connectors/) to reach OpenAI, Anthropic, Ollama, etc.
    /// Currently implements a prompt-echo response that preserves the
    /// engine API contract while the connector bridge is integrated.
    fn generateConnector(self: *Self, request: scheduler_mod.Request) !Result {
        const start = time_mod.timestampMs();

        const cache_ok = try self.kv_cache.allocate(request.id, request.max_tokens);
        if (!cache_ok) {
            return Result{
                .id = request.id,
                .text = "Error: insufficient KV cache capacity",
                .tokens = &.{},
                .finish_reason = .error_,
                .prompt_tokens = 0,
                .completion_tokens = 0,
                .latency_ms = 0,
                .ttft_ms = 0,
                .tokens_per_second = 0,
            };
        }
        defer self.kv_cache.free(request.id);

        // Build connector response.
        // In production: connectors.route(self.config.model_id, request.prompt, .{ ... })
        // For now: echo-based response that acknowledges the prompt and model.
        const response_text = try std.fmt.allocPrint(
            self.allocator,
            "[{s}] Processing: {s}",
            .{ self.config.model_id, request.prompt[0..@min(request.prompt.len, 200)] },
        );

        const end = time_mod.timestampMs();
        const latency: f32 = @floatFromInt(end - start);
        const token_count: u32 = @intCast(@min(response_text.len / 4, std.math.maxInt(u32)));

        self.total_requests += 1;
        self.total_tokens += token_count;

        return Result{
            .id = request.id,
            .text = response_text,
            .tokens = &.{},
            .finish_reason = .stop,
            .prompt_tokens = @intCast(@min(request.prompt.len, std.math.maxInt(u32))),
            .completion_tokens = token_count,
            .latency_ms = latency,
            .ttft_ms = latency,
            .tokens_per_second = if (latency > 0) @as(f32, @floatFromInt(token_count)) / (latency / 1000.0) else 0,
        };
    }

    /// Local transformer generation: uses the built-in transformer forward pass.
    /// Falls back to demo mode when local model weights are not loaded.
    fn generateLocal(self: *Self, request: scheduler_mod.Request) !Result {
        // Local transformer integration point.
        // When model weights are loaded (via src/features/ai/llm/model.zig):
        //   1. Tokenize input via BPE tokenizer
        //   2. Run transformer forward pass with KV cache
        //   3. Sample output tokens via sampler
        //   4. Decode tokens back to text
        //
        // For now, delegate to demo with a marker in the output.
        // Local transformer integration point.
        // When model weights are loaded, this would run the transformer forward pass.
        // For now, delegate to demo backend.
        return self.generateDemo(request);
    }

    /// Demo generation: synthetic text seeded from input prompts.
    fn generateDemo(self: *Self, request: scheduler_mod.Request) !Result {
        const start = time_mod.timestampMs();

        const cache_ok = try self.kv_cache.allocate(request.id, request.max_tokens);
        if (!cache_ok) {
            return Result{
                .id = request.id,
                .text = "Error: insufficient KV cache capacity",
                .tokens = &.{},
                .finish_reason = .error_,
                .prompt_tokens = 0,
                .completion_tokens = 0,
                .latency_ms = 0,
                .ttft_ms = 0,
                .tokens_per_second = 0,
            };
        }
        defer self.kv_cache.free(request.id);

        const num_tokens = request.max_tokens;
        const tokens = try self.allocator.alloc(u32, num_tokens);
        const vocab_n = @min(self.config.vocab_size, 256);
        var local_sampler = self.sampler;
        local_sampler.params.temperature = request.temperature;
        local_sampler.params.top_p = request.top_p;
        local_sampler.params.top_k = request.top_k;

        // Seed logits from the prompt content so output depends on input.
        var prompt_hash: u64 = 0x517cc1b727220a95;
        for (request.prompt) |byte| {
            prompt_hash ^= @as(u64, byte);
            prompt_hash *%= 0x100000001b3;
        }

        var finish_reason: FinishReason = .length;
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

        // Build text output from demo vocabulary
        var total_len: usize = 0;
        for (tokens[0..actual_count], 0..) |tok, i| {
            const word = demo_vocabulary[tok % demo_vocabulary.len];
            total_len += word.len;
            if (i + 1 < actual_count) total_len += 1;
        }

        const text_buf = try self.allocator.alloc(u8, total_len);
        var pos: usize = 0;
        for (tokens[0..actual_count], 0..) |tok, i| {
            const word = demo_vocabulary[tok % demo_vocabulary.len];
            @memcpy(text_buf[pos..][0..word.len], word);
            pos += word.len;
            if (i + 1 < actual_count) {
                text_buf[pos] = ' ';
                pos += 1;
            }
        }

        const end = time_mod.timestampMs();
        const latency: f32 = @floatFromInt(end - start);
        const tps: f32 = if (latency > 0)
            @as(f32, @floatFromInt(actual_count)) / (latency / 1000.0)
        else
            0;

        self.total_requests += 1;
        self.total_tokens += actual_count;

        return Result{
            .id = request.id,
            .text = text_buf,
            .tokens = tokens,
            .finish_reason = finish_reason,
            .prompt_tokens = @intCast(@min(request.prompt.len, std.math.maxInt(u32))),
            .completion_tokens = actual_count,
            .latency_ms = latency,
            .ttft_ms = latency / @as(f32, @floatFromInt(@max(actual_count, 1))),
            .tokens_per_second = tps,
        };
    }

    pub fn submit(self: *Self, request: scheduler_mod.Request) !bool {
        return self.scheduler.submit(request);
    }

    pub fn getStats(self: *const Self) Stats {
        const avg_tps: f32 = if (self.total_requests > 0)
            @as(f32, @floatFromInt(self.total_tokens)) / @as(f32, @floatFromInt(self.total_requests))
        else
            0;

        return .{
            .total_requests = self.total_requests,
            .total_tokens_generated = self.total_tokens,
            .active_sequences = @intCast(self.kv_cache.activeSequences()),
            .cache_utilization = self.kv_cache.getUtilization(),
            .pending_requests = @intCast(self.scheduler.pendingCount()),
            .avg_tokens_per_second = avg_tps,
            .backend = self.config.backend,
        };
    }
};

test "engine init and stats" {
    const allocator = std.testing.allocator;

    var engine = try Engine.init(allocator, .{
        .kv_cache_pages = 100,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 8,
    });
    defer engine.deinit();

    const stats = engine.getStats();
    try std.testing.expectEqual(@as(u64, 0), stats.total_requests);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), stats.cache_utilization, 1e-5);
    try std.testing.expectEqual(Backend.demo, stats.backend);
}

test "engine generate demo" {
    const allocator = std.testing.allocator;

    var engine = try Engine.init(allocator, .{
        .kv_cache_pages = 100,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 8,
        .vocab_size = 256,
    });
    defer engine.deinit();

    const result = try engine.generate(.{
        .id = 1,
        .prompt = "Hello world",
        .max_tokens = 10,
        .temperature = 0.7,
        .top_p = 0.9,
        .top_k = 40,
    });
    defer allocator.free(result.tokens);
    defer allocator.free(result.text);

    try std.testing.expect(result.completion_tokens > 0);
    try std.testing.expect(result.text.len > 0);
    try std.testing.expectEqual(@as(u64, 1), engine.getStats().total_requests);
}

test "engine connector backend" {
    const allocator = std.testing.allocator;

    var engine = try Engine.init(allocator, .{
        .kv_cache_pages = 100,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 8,
        .backend = .connector,
        .model_id = "test-model",
    });
    defer engine.deinit();

    const result = try engine.generate(.{
        .id = 1,
        .prompt = "Explain HNSW",
        .max_tokens = 10,
    });
    defer allocator.free(result.text);

    try std.testing.expect(result.text.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, result.text, "test-model") != null);
    try std.testing.expectEqual(Backend.connector, engine.getStats().backend);
}

test "engine submit to scheduler" {
    const allocator = std.testing.allocator;

    var engine = try Engine.init(allocator, .{
        .kv_cache_pages = 10,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 2,
    });
    defer engine.deinit();

    const ok = try engine.submit(.{ .id = 1, .prompt = "test", .priority = 100 });
    try std.testing.expect(ok);
    try std.testing.expectEqual(@as(u32, 1), engine.getStats().pending_requests);
}
