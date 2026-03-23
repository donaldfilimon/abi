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
//!
//! ## Thread Safety
//!
//! `total_requests`, `total_tokens`, and `total_elapsed_ns` are
//! `std.atomic.Value(u64)` counters updated with `fetchAdd(.acq_rel)` so that
//! concurrent `generateAsync` callers never produce torn reads or lost updates.
//! `getStats()` reads them with `load(.acquire)` and requires no mutex.
//!
//! `in_flight_async` is guarded by `async_mu`.
//! `generation_mu` serialises access to the KV cache and sampler state inside
//! each `generate*` call.

const std = @import("std");
const Allocator = std.mem.Allocator;
const kv_cache_mod = @import("kv_cache.zig");
const scheduler_mod = @import("scheduler.zig");
const sampler_mod = @import("sampler.zig");
const time_mod = @import("../foundation/time.zig");
const sync_mod = @import("../foundation/sync.zig");

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

/// Generation result.
///
/// **Ownership**: The caller owns the returned `Result` and *must* call
/// `deinit()` with the same allocator that the engine was initialised with
/// when the result is no longer needed. Failing to do so will leak the
/// backing `text` and `tokens` buffers when the respective `*_owned` flags
/// are `true`.
pub const Result = struct {
    id: u64,
    text: []const u8,
    text_owned: bool = true,
    tokens: []const u32,
    tokens_owned: bool = true,
    finish_reason: FinishReason,
    prompt_tokens: u32,
    completion_tokens: u32,
    latency_ms: f32,
    ttft_ms: f32,
    tokens_per_second: f32,

    /// Release owned buffers. Safe to call multiple times.
    pub fn deinit(self: *Result, allocator: Allocator) void {
        if (self.text_owned) {
            allocator.free(self.text);
        }
        if (self.tokens_owned) {
            allocator.free(self.tokens);
        }
        self.text_owned = false;
        self.tokens_owned = false;
    }
};

/// Thread-safe container returned by `generateAsyncWithTimeout`.
///
/// The engine spawns a background thread that writes the result into this
/// container. The caller can either:
///   - call `wait()` / `waitTimeout()` to block until the result is ready, or
///   - call `deinit()` to abandon the result (the background thread will
///     clean up its own allocations).
///
/// Exactly one of {caller, background thread} frees the inner `Result`
/// buffers, coordinated by the `state` atomic.
pub const AsyncResult = struct {
    const Self = @This();

    const State = enum(u8) {
        /// Background thread has not yet written the result.
        pending = 0,
        /// Result is available for the caller to consume.
        ready = 1,
        /// Caller abandoned the result; background thread must free it.
        abandoned = 2,
    };

    allocator: Allocator,
    state: std.atomic.Value(u8),
    result: Result,

    /// Block until the result is available or `timeout_ms` elapses.
    /// Returns `null` on timeout. The caller owns the returned `Result`
    /// and must call `Result.deinit()` on it.
    pub fn waitTimeout(self: *Self, timeout_ms: u64) ?Result {
        const deadline_ns = time_mod.timestampNs() + timeout_ms * std.time.ns_per_ms;
        while (self.state.load(.acquire) == @intFromEnum(State.pending)) {
            if (time_mod.timestampNs() >= deadline_ns) return null;
            time_mod.sleepMs(1);
        }
        return self.result;
    }

    /// Block indefinitely until the result is available. The caller
    /// owns the returned `Result` and must call `Result.deinit()`.
    pub fn wait(self: *Self) Result {
        while (self.state.load(.acquire) == @intFromEnum(State.pending)) {
            time_mod.sleepMs(1);
        }
        return self.result;
    }

    /// Abandon a pending result. If the background thread has not yet
    /// finished, it will free the result and this container when it
    /// completes.
    ///
    /// **Must only be called when the result has NOT been consumed**
    /// (i.e. `waitTimeout` returned `null` or `wait` was never called).
    /// After calling `deinit()`, the caller must not dereference `self`.
    pub fn deinit(self: *Self) void {
        // Try to transition pending -> abandoned so the thread knows
        // it must free the result itself.
        _ = self.state.cmpxchgStrong(
            @intFromEnum(State.pending),
            @intFromEnum(State.abandoned),
            .acq_rel,
            .acquire,
        );
        // The heap-allocated AsyncResult is freed by the background
        // thread after it finishes (it always runs to completion).
    }

    /// Free the container after the result has been consumed via
    /// `wait()` or `waitTimeout()`. The caller must have already
    /// extracted and taken ownership of the inner `Result`.
    pub fn destroy(self: *Self) void {
        self.allocator.destroy(self);
    }
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

fn makeErrorResult(id: u64, message: []const u8) Result {
    return .{
        .id = id,
        .text = message,
        .text_owned = false,
        .tokens = &.{},
        .tokens_owned = false,
        .finish_reason = .error_,
        .prompt_tokens = 0,
        .completion_tokens = 0,
        .latency_ms = 0,
        .ttft_ms = 0,
        .tokens_per_second = 0,
    };
}

fn elapsedNsToMs(elapsed_ns: u64) f32 {
    return @as(f32, @floatFromInt(elapsed_ns)) / @as(f32, @floatFromInt(std.time.ns_per_ms));
}

fn tokensPerSecond(token_count: u32, elapsed_ns: u64) f32 {
    if (elapsed_ns == 0) return 0;
    return @as(f32, @floatFromInt(token_count)) * @as(f32, @floatFromInt(std.time.ns_per_s)) /
        @as(f32, @floatFromInt(elapsed_ns));
}

pub const Engine = struct {
    const Self = @This();

    allocator: Allocator,
    config: Config,
    kv_cache: kv_cache_mod.PagedKVCache,
    scheduler: scheduler_mod.Scheduler,
    sampler: sampler_mod.Sampler,

    /// Atomic counters — safe to update from concurrent generateAsync threads
    /// without holding generation_mu.
    total_requests: std.atomic.Value(u64),
    total_tokens: std.atomic.Value(u64),
    total_elapsed_ns: std.atomic.Value(u64),

    /// Serialises KV-cache access and sampler state within each generate call.
    generation_mu: sync_mod.Mutex = .{},
    /// Guards in_flight_async and closing_async.
    async_mu: sync_mod.Mutex = .{},
    in_flight_async: u32 = 0,
    closing_async: bool = false,

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
            .total_requests = std.atomic.Value(u64).init(0),
            .total_tokens = std.atomic.Value(u64).init(0),
            .total_elapsed_ns = std.atomic.Value(u64).init(0),
        };
    }

    pub fn deinit(self: *Self) void {
        self.async_mu.lock();
        self.closing_async = true;
        self.async_mu.unlock();

        while (true) {
            self.async_mu.lock();
            const in_flight = self.in_flight_async;
            self.async_mu.unlock();
            if (in_flight == 0) break;
            time_mod.sleepMs(1);
        }

        self.generation_mu.lock();
        self.generation_mu.unlock();

        self.kv_cache.deinit();
        self.scheduler.deinit();
    }

    /// Generate a response using the configured backend.
    pub fn generate(self: *Self, request: scheduler_mod.Request) !Result {
        self.async_mu.lock();
        const closing = self.closing_async;
        self.async_mu.unlock();
        if (closing) return error.ShutdownInProgress;

        self.generation_mu.lock();
        defer self.generation_mu.unlock();

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
        const start = time_mod.timestampNs();

        const cache_ok = try self.kv_cache.allocate(request.id, request.max_tokens);
        if (!cache_ok) {
            return makeErrorResult(request.id, "Error: insufficient KV cache capacity");
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
        errdefer self.allocator.free(response_text);

        const end = time_mod.timestampNs();
        const elapsed_ns = end - start;
        const latency = elapsedNsToMs(elapsed_ns);
        const token_count: u32 = @intCast(@min(response_text.len / 4, std.math.maxInt(u32)));

        _ = self.total_requests.fetchAdd(1, .acq_rel);
        _ = self.total_tokens.fetchAdd(token_count, .acq_rel);
        _ = self.total_elapsed_ns.fetchAdd(elapsed_ns, .acq_rel);

        return Result{
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
            .tokens_per_second = tokensPerSecond(token_count, elapsed_ns),
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
        const start = time_mod.timestampNs();

        const cache_ok = try self.kv_cache.allocate(request.id, request.max_tokens);
        if (!cache_ok) {
            return makeErrorResult(request.id, "Error: insufficient KV cache capacity");
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

        const end = time_mod.timestampNs();
        const elapsed_ns = end - start;
        const latency = elapsedNsToMs(elapsed_ns);
        const tps = tokensPerSecond(actual_count, elapsed_ns);

        _ = self.total_requests.fetchAdd(1, .acq_rel);
        _ = self.total_tokens.fetchAdd(actual_count, .acq_rel);
        _ = self.total_elapsed_ns.fetchAdd(elapsed_ns, .acq_rel);

        return Result{
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

    pub fn submit(self: *Self, request: scheduler_mod.Request) !bool {
        return self.scheduler.submit(request);
    }

    pub const AsyncCallback = *const fn (Result) void;

    fn beginAsyncJob(self: *Self) !void {
        self.async_mu.lock();
        defer self.async_mu.unlock();
        if (self.closing_async) return error.ShutdownInProgress;
        self.in_flight_async += 1;
    }

    fn finishAsyncJob(self: *Self) void {
        self.async_mu.lock();
        defer self.async_mu.unlock();
        std.debug.assert(self.in_flight_async > 0);
        self.in_flight_async -= 1;
    }

    /// Generate a response asynchronously on a separate thread, invoking a
    /// callback upon completion. The callback receives the result by value;
    /// the engine frees the result's owned buffers *after* the callback
    /// returns.
    ///
    /// **Note**: prefer `generateAsyncWithTimeout` for new code — it gives
    /// the caller explicit ownership and timeout support.
    pub fn generateAsync(self: *Self, request: scheduler_mod.Request, callback: AsyncCallback) !void {
        try self.beginAsyncJob();
        errdefer self.finishAsyncJob();

        const prompt_copy = try self.allocator.dupe(u8, request.prompt);
        errdefer self.allocator.free(prompt_copy);

        const profile_copy = try self.allocator.dupe(u8, request.profile);
        errdefer self.allocator.free(profile_copy);

        const AsyncContext = struct {
            engine: *Self,
            request: scheduler_mod.Request,
            callback: AsyncCallback,

            fn run(ctx: @This()) void {
                defer ctx.engine.finishAsyncJob();
                defer ctx.engine.allocator.free(ctx.request.profile);
                defer ctx.engine.allocator.free(ctx.request.prompt);

                var res = ctx.engine.generate(ctx.request) catch makeErrorResult(ctx.request.id, "Error: internal generation error");
                defer res.deinit(ctx.engine.allocator);
                ctx.callback(res);
            }
        };

        const ctx = AsyncContext{
            .engine = self,
            .request = .{
                .id = request.id,
                .prompt = prompt_copy,
                .max_tokens = request.max_tokens,
                .temperature = request.temperature,
                .top_p = request.top_p,
                .top_k = request.top_k,
                .profile = profile_copy,
                .priority = request.priority,
                .created_at = request.created_at,
                .stream = request.stream,
            },
            .callback = callback,
        };

        const thread = try std.Thread.spawn(.{}, AsyncContext.run, .{ctx});
        thread.detach();
    }

    /// Generate a response asynchronously, returning an `*AsyncResult` that
    /// the caller can poll or wait on with an optional timeout.
    ///
    /// **Ownership contract**:
    /// - On success the caller receives a heap-allocated `*AsyncResult`.
    /// - Call `waitTimeout(ms)` or `wait()` to retrieve the inner `Result`.
    /// - The returned `Result` is owned by the caller — call
    ///   `result.deinit(engine.allocator)` when done.
    /// - If the caller no longer needs the result (e.g. after a timeout),
    ///   call `async_result.deinit()`. The background thread will free the
    ///   inner buffers when it completes.
    /// - The `*AsyncResult` itself is freed by the background thread after
    ///   it finishes, so the caller must not dereference it after calling
    ///   `deinit()`.
    pub fn generateAsyncWithTimeout(self: *Self, request: scheduler_mod.Request) !*AsyncResult {
        try self.beginAsyncJob();
        errdefer self.finishAsyncJob();

        const prompt_copy = try self.allocator.dupe(u8, request.prompt);
        errdefer self.allocator.free(prompt_copy);

        const profile_copy = try self.allocator.dupe(u8, request.profile);
        errdefer self.allocator.free(profile_copy);

        const ar = try self.allocator.create(AsyncResult);
        errdefer self.allocator.destroy(ar);

        ar.* = .{
            .allocator = self.allocator,
            .state = std.atomic.Value(u8).init(@intFromEnum(AsyncResult.State.pending)),
            .result = undefined,
        };

        const TimeoutContext = struct {
            engine: *Self,
            request: scheduler_mod.Request,
            ar: *AsyncResult,

            fn run(ctx: @This()) void {
                defer ctx.engine.finishAsyncJob();
                defer ctx.engine.allocator.free(ctx.request.profile);
                defer ctx.engine.allocator.free(ctx.request.prompt);

                const res = ctx.engine.generate(ctx.request) catch makeErrorResult(ctx.request.id, "Error: internal generation error");

                // Write result data before attempting to publish.
                ctx.ar.result = res;

                // Atomically transition pending -> ready. If the caller
                // has already abandoned (pending -> abandoned), we own
                // the result and must free it.
                const prev = ctx.ar.state.cmpxchgStrong(
                    @intFromEnum(AsyncResult.State.pending),
                    @intFromEnum(AsyncResult.State.ready),
                    .acq_rel,
                    .acquire,
                );
                if (prev != null) {
                    // Caller abandoned — clean up owned buffers directly.
                    if (ctx.ar.result.text_owned) ctx.engine.allocator.free(ctx.ar.result.text);
                    if (ctx.ar.result.tokens_owned) ctx.engine.allocator.free(ctx.ar.result.tokens);
                    ctx.engine.allocator.destroy(ctx.ar);
                    return;
                }
            }
        };

        const ctx = TimeoutContext{
            .engine = self,
            .request = .{
                .id = request.id,
                .prompt = prompt_copy,
                .max_tokens = request.max_tokens,
                .temperature = request.temperature,
                .top_p = request.top_p,
                .top_k = request.top_k,
                .profile = profile_copy,
                .priority = request.priority,
                .created_at = request.created_at,
                .stream = request.stream,
            },
            .ar = ar,
        };

        const thread = try std.Thread.spawn(.{}, TimeoutContext.run, .{ctx});
        thread.detach();

        return ar;
    }

    fn isClosing(self: *Self) bool {
        self.async_mu.lock();
        defer self.async_mu.unlock();
        return self.closing_async;
    }

    /// Return a snapshot of engine statistics. Safe to call from any thread
    /// without holding any lock — counters are read with acquire ordering.
    pub fn getStats(self: *const Self) Stats {
        const reqs = self.total_requests.load(.acquire);
        const toks = self.total_tokens.load(.acquire);
        const ns = self.total_elapsed_ns.load(.acquire);

        const avg_tps: f32 = if (ns > 0)
            @as(f32, @floatFromInt(toks)) * @as(f32, @floatFromInt(std.time.ns_per_s)) /
                @as(f32, @floatFromInt(ns))
        else
            0;

        return .{
            .total_requests = reqs,
            .total_tokens_generated = toks,
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

    var result = try engine.generate(.{
        .id = 1,
        .prompt = "Hello world",
        .max_tokens = 10,
        .temperature = 0.7,
        .top_p = 0.9,
        .top_k = 40,
    });
    defer result.deinit(allocator);

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

    var result = try engine.generate(.{
        .id = 1,
        .prompt = "Explain HNSW",
        .max_tokens = 10,
    });
    defer result.deinit(allocator);

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

const AsyncPromptHelper = struct {
    var done: std.atomic.Value(bool) = .{ .raw = false };
    var result_id: std.atomic.Value(u64) = .{ .raw = 0 };
    var text_len: std.atomic.Value(usize) = .{ .raw = 0 };

    fn cb(res: Result) void {
        result_id.store(res.id, .release);
        text_len.store(res.text.len, .release);
        done.store(true, .release);
    }
};

test "engine generate async clones request prompt" {
    const allocator = std.testing.allocator;

    var engine = try Engine.init(allocator, .{
        .kv_cache_pages = 100,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 4,
        .vocab_size = 256,
    });
    defer engine.deinit();

    AsyncPromptHelper.done.store(false, .release);
    AsyncPromptHelper.result_id.store(0, .release);
    AsyncPromptHelper.text_len.store(0, .release);

    const prompt = try allocator.dupe(u8, "async prompt lifetime regression");
    try engine.generateAsync(.{
        .id = 42,
        .prompt = prompt,
        .max_tokens = 12,
        .priority = 100,
    }, AsyncPromptHelper.cb);
    allocator.free(prompt);

    const noise = try allocator.alloc(u8, 4096);
    defer allocator.free(noise);
    @memset(noise, 0xaa);

    var spins: usize = 0;
    while (!AsyncPromptHelper.done.load(.acquire) and spins < 5000) : (spins += 1) {
        time_mod.sleepMs(1);
    }

    try std.testing.expect(AsyncPromptHelper.done.load(.acquire));
    try std.testing.expectEqual(@as(u64, 42), AsyncPromptHelper.result_id.load(.acquire));
    try std.testing.expect(AsyncPromptHelper.text_len.load(.acquire) > 0);
}

test "engine average throughput uses elapsed time" {
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

    engine.total_requests.store(3, .release);
    engine.total_tokens.store(300, .release);
    engine.total_elapsed_ns.store(2 * std.time.ns_per_s, .release);

    const stats = engine.getStats();
    try std.testing.expectApproxEqAbs(@as(f32, 150.0), stats.avg_tokens_per_second, 0.001);
}

test "result deinit frees owned buffers (leak detection)" {
    const allocator = std.testing.allocator;

    var engine = try Engine.init(allocator, .{
        .kv_cache_pages = 100,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 4,
        .vocab_size = 256,
    });
    defer engine.deinit();

    // Generate and free — testing allocator detects leaks if deinit is skipped.
    var result = try engine.generate(.{
        .id = 99,
        .prompt = "leak check",
        .max_tokens = 8,
    });
    try std.testing.expect(result.text_owned);
    try std.testing.expect(result.tokens_owned);

    result.deinit(allocator);

    // After deinit, owned flags must be false (safe to call again).
    try std.testing.expect(!result.text_owned);
    try std.testing.expect(!result.tokens_owned);
}

test "result deinit is idempotent" {
    const allocator = std.testing.allocator;

    var engine = try Engine.init(allocator, .{
        .kv_cache_pages = 100,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 4,
        .vocab_size = 256,
    });
    defer engine.deinit();

    var result = try engine.generate(.{
        .id = 100,
        .prompt = "double free guard",
        .max_tokens = 6,
    });

    result.deinit(allocator);
    // Second deinit must be a no-op (no double-free).
    result.deinit(allocator);
}

test "error result has no owned buffers" {
    var err_result = makeErrorResult(0, "boom");
    try std.testing.expect(!err_result.text_owned);
    try std.testing.expect(!err_result.tokens_owned);
    // deinit on error result is a no-op — must not crash.
    err_result.deinit(std.testing.allocator);
}

test "generateAsyncWithTimeout returns result" {
    const allocator = std.testing.allocator;

    var engine = try Engine.init(allocator, .{
        .kv_cache_pages = 100,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 4,
        .vocab_size = 256,
    });
    defer engine.deinit();

    const ar = try engine.generateAsyncWithTimeout(.{
        .id = 77,
        .prompt = "timeout test",
        .max_tokens = 8,
    });

    // Wait with a generous timeout.
    const maybe_result = ar.waitTimeout(5000);
    try std.testing.expect(maybe_result != null);

    var result = maybe_result.?;
    try std.testing.expectEqual(@as(u64, 77), result.id);
    try std.testing.expect(result.text.len > 0);
    result.deinit(allocator);

    // Free the container now that we have consumed the result.
    ar.destroy();
}

test "generateAsyncWithTimeout abandon cleans up" {
    const allocator = std.testing.allocator;

    var engine = try Engine.init(allocator, .{
        .kv_cache_pages = 100,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 4,
        .vocab_size = 256,
    });
    defer engine.deinit();

    const ar = try engine.generateAsyncWithTimeout(.{
        .id = 88,
        .prompt = "abandon test",
        .max_tokens = 8,
    });

    // Immediately abandon — the background thread will free the result.
    ar.deinit();

    // Wait for the background thread to finish via engine.deinit()
    // (called by defer above). The testing allocator will detect
    // leaks if the thread fails to free the result.
}
