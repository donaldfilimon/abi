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
const build_options = @import("build_options");
const engine_async = @import("engine/async.zig");
const engine_backends = @import("engine/backends.zig");
const engine_types = @import("engine/types.zig");
const kv_cache_mod = @import("kv_cache.zig");
const scheduler_mod = @import("scheduler.zig");
const sampler_mod = @import("sampler.zig");
const time_mod = @import("../foundation/time.zig");
const sync_mod = @import("../foundation/sync.zig");

// Local LLM model — comptime-gated on feat_ai + feat_llm
const llm_model = if (build_options.feat_ai and build_options.feat_llm)
    @import("../features/ai/llm/model/llama.zig")
else
    struct {
        pub const LlamaModel = void;
    };

pub const Backend = engine_types.Backend;
pub const Config = engine_types.Config;
pub const FinishReason = engine_types.FinishReason;
pub const Result = engine_types.Result;
pub const AsyncResult = engine_types.AsyncResult;
pub const Stats = engine_types.Stats;

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
    /// Guards scheduler queue access.
    scheduler_mu: sync_mod.Mutex = .{},
    /// Guards in_flight_async and closing_async.
    async_mu: sync_mod.Mutex = .{},
    in_flight_async: u32 = 0,
    closing_async: bool = false,

    /// Loaded local LLM model (type-erased; null until loadModel is called).
    local_model: ?*anyopaque = null,

    /// Load a GGUF model for local inference. Requires feat_ai + feat_llm.
    /// Safe to call multiple times — frees any previously loaded model first.
    pub fn loadModel(self: *Self, path: []const u8) !void {
        if (comptime !(build_options.feat_ai and build_options.feat_llm)) {
            return error.LocalBackendNotAvailable;
        }
        // Free existing model if one is already loaded
        if (self.local_model) |prev_opaque| {
            const prev: *llm_model.LlamaModel = @ptrCast(@alignCast(prev_opaque));
            prev.deinit();
            self.allocator.destroy(prev);
            self.local_model = null;
        }
        const LlamaModel = llm_model.LlamaModel;
        const model_ptr = try self.allocator.create(LlamaModel);
        errdefer self.allocator.destroy(model_ptr);
        model_ptr.* = try LlamaModel.load(self.allocator, path);
        self.local_model = @ptrCast(model_ptr);
    }

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

        // Free local model if loaded
        if (comptime build_options.feat_ai and build_options.feat_llm) {
            if (self.local_model) |model_opaque| {
                const model: *llm_model.LlamaModel = @ptrCast(@alignCast(model_opaque));
                model.deinit();
                self.allocator.destroy(model);
                self.local_model = null;
            }
        }

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
        return engine_backends.generateConnector(self, request);
    }

    /// Local transformer generation: uses the built-in transformer forward pass.
    /// Falls back to demo mode when local model weights are not loaded.
    fn generateLocal(self: *Self, request: scheduler_mod.Request) !Result {
        return engine_backends.generateLocal(self, request);
    }

    /// Demo generation: synthetic text seeded from input prompts.
    fn generateDemo(self: *Self, request: scheduler_mod.Request) !Result {
        return engine_backends.generateDemo(self, request);
    }

    pub fn submit(self: *Self, request: scheduler_mod.Request) !bool {
        self.scheduler_mu.lock();
        defer self.scheduler_mu.unlock();
        return self.scheduler.submit(request);
    }

    pub const AsyncCallback = engine_async.AsyncCallback;

    fn beginAsyncJob(self: *Self) !void {
        return engine_async.beginAsyncJob(self);
    }

    fn finishAsyncJob(self: *Self) void {
        engine_async.finishAsyncJob(self);
    }

    pub fn generateAsync(self: *Self, request: scheduler_mod.Request, callback: AsyncCallback) !void {
        return engine_async.generateAsync(self, request, callback);
    }

    pub fn generateAsyncWithTimeout(self: *Self, request: scheduler_mod.Request) !*AsyncResult {
        return engine_async.generateAsyncWithTimeout(self, request);
    }

    pub fn getStats(self: *Self) Stats {
        const reqs = self.total_requests.load(.acquire);
        const toks = self.total_tokens.load(.acquire);
        const elapsed = self.total_elapsed_ns.load(.acquire);
        const pending = self.scheduler.pendingCount();
        const elapsed_s = @as(f32, @floatFromInt(elapsed)) / @as(f32, @floatFromInt(std.time.ns_per_s));
        const avg_tps: f32 = if (elapsed_s > 0) @as(f32, @floatFromInt(toks)) / elapsed_s else 0;

        return .{
            .total_requests = reqs,
            .total_tokens_generated = toks,
            .active_sequences = @intCast(self.kv_cache.activeSequences()),
            .cache_utilization = self.kv_cache.getUtilization(),
            .pending_requests = @intCast(pending),
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

test "engine connector backend: unsupported provider returns error" {
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

    // model_id "test-model" has no slash -> falls back to echo
    var result = try engine.generate(.{
        .id = 1,
        .prompt = "Explain HNSW",
        .max_tokens = 10,
    });
    defer if (result.text_owned) allocator.free(result.text);
    try std.testing.expect(std.mem.indexOf(u8, result.text, "Explain HNSW") != null);
    try std.testing.expectEqual(Backend.connector, engine.getStats().backend);
    result.deinit(allocator);

    try std.testing.expect(result.text.len > 0);
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

test "engine stats calculation" {
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
    // deinit() handles both result buffer cleanup and container
    // destruction on the happy path (state == ready).
    defer ar.deinit();

    // Wait with a generous timeout.
    const maybe_result = ar.waitTimeout(5000);
    try std.testing.expect(maybe_result != null);

    const result = maybe_result.?;
    try std.testing.expectEqual(@as(u64, 77), result.id);
    try std.testing.expect(result.text.len > 0);
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
