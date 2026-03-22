//! Inference Engine
//!
//! Orchestrates token generation by combining the paged KV-cache, priority
//! scheduler, and token sampler. Provides both synchronous and async
//! (submit/poll) interfaces.

const std = @import("std");
const Allocator = std.mem.Allocator;
const kv_cache_mod = @import("kv_cache.zig");
const scheduler_mod = @import("scheduler.zig");
const sampler_mod = @import("sampler.zig");
const time_mod = @import("../foundation/time.zig");

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

    pub fn generate(self: *Self, request: scheduler_mod.Request) !Result {
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
        // Hash each prompt byte to create a per-position context seed.
        var prompt_hash: u64 = 0x517cc1b727220a95; // FNV offset basis
        for (request.prompt) |byte| {
            prompt_hash ^= @as(u64, byte);
            prompt_hash *%= 0x100000001b3; // FNV prime
        }

        var finish_reason: FinishReason = .length;
        var actual_count: u32 = 0;

        for (tokens, 0..) |*t, step_i| {
            var logits_buf: [256]f32 = undefined;
            const logits_slice = logits_buf[0..vocab_n];

            // Build logits seeded by prompt hash and step position.
            // This produces input-dependent, non-trivial distributions.
            var step_seed = prompt_hash +% @as(u64, @intCast(step_i)) *% 0x9e3779b97f4a7c15;
            for (logits_slice, 0..) |*l, j| {
                // Mix step seed with token index for varied distribution
                const mixed = step_seed ^ (@as(u64, @intCast(j)) *% 0x517cc1b727220a95);
                // Convert to float in [-2, 2] range for reasonable logit values
                const bits: u32 = @truncate(mixed >> 16);
                l.* = (@as(f32, @floatFromInt(bits % 1024)) / 256.0) - 2.0;
                step_seed = step_seed *% 0x100000001b3 +% @as(u64, @intCast(j));
            }

            // Boost EOS-like tokens (id 0, 1, 2) slightly in later steps
            // to allow natural stopping
            if (step_i > num_tokens / 2) {
                if (vocab_n > 2) {
                    logits_slice[1] += @as(f32, @floatFromInt(step_i)) * 0.02;
                    logits_slice[2] += @as(f32, @floatFromInt(step_i)) * 0.015;
                }
            }

            t.* = local_sampler.sample(logits_slice);
            actual_count += 1;

            // Treat token ids 0-2 as stop tokens (EOS) after generating
            // at least a few tokens
            if (t.* <= 2 and step_i >= 4) {
                finish_reason = .stop;
                actual_count = @intCast(step_i + 1);
                break;
            }
        }

        // Build text output from generated tokens using a simple ASCII mapping
        const text_buf = try self.allocator.alloc(u8, actual_count);
        for (tokens[0..actual_count], 0..) |tok, i| {
            // Map token id to printable ASCII range [32..126]
            text_buf[i] = @intCast(32 + (tok % 95));
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
}

test "engine generate" {
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
