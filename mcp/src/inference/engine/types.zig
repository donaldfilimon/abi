const std = @import("std");
const Allocator = std.mem.Allocator;
const time_mod = @import("../../foundation/time.zig");

pub const Backend = enum {
    demo,
    connector,
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
    backend: Backend = .demo,
    model_id: []const u8 = "abi-demo",
    model_path: []const u8 = "",
};

pub const FinishReason = enum {
    stop,
    length,
    error_,
};

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

pub const AsyncResult = struct {
    const Self = @This();

    pub const State = enum(u8) {
        pending = 0,
        ready = 1,
        abandoned = 2,
    };

    allocator: Allocator,
    state: std.atomic.Value(u8),
    result: Result,

    pub fn waitTimeout(self: *Self, timeout_ms: u64) ?Result {
        const deadline_ns = time_mod.timestampNs() + timeout_ms * std.time.ns_per_ms;
        while (self.state.load(.acquire) == @intFromEnum(State.pending)) {
            if (time_mod.timestampNs() >= deadline_ns) return null;
            time_mod.sleepMs(1);
        }
        return self.result;
    }

    pub fn wait(self: *Self) Result {
        while (self.state.load(.acquire) == @intFromEnum(State.pending)) {
            time_mod.sleepMs(1);
        }
        return self.result;
    }

    pub fn deinit(self: *Self) void {
        const prev = self.state.cmpxchgStrong(
            @intFromEnum(State.pending),
            @intFromEnum(State.abandoned),
            .acq_rel,
            .acquire,
        );
        if (prev) |state| {
            if (state == @intFromEnum(State.ready)) {
                self.result.deinit(self.allocator);
                self.allocator.destroy(self);
            }
        }
    }

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

pub const demo_vocabulary = [_][]const u8{
    "the",      "a",       "is",      "of",     "and",    "to",      "in",     "that",   "it",     "for",
    "was",      "on",      "are",     "with",   "as",     "at",      "be",     "this",   "have",   "from",
    "data",     "model",   "query",   "result", "system", "process", "value",  "node",   "index",  "vector",
    "search",   "layer",   "output",  "input",  "token",  "state",   "memory", "cache",  "error",  "status",
    "response", "request", "context", "agent",  "neural", "graph",   "batch",  "stream", "config", "module",
};

pub fn makeErrorResult(id: u64, message: []const u8) Result {
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

pub fn elapsedNsToMs(elapsed_ns: u64) f32 {
    return @as(f32, @floatFromInt(elapsed_ns)) / @as(f32, @floatFromInt(std.time.ns_per_ms));
}

pub fn tokensPerSecond(token_count: u32, elapsed_ns: u64) f32 {
    if (elapsed_ns == 0) return 0;
    return @as(f32, @floatFromInt(token_count)) * @as(f32, @floatFromInt(std.time.ns_per_s)) /
        @as(f32, @floatFromInt(elapsed_ns));
}

test {
    std.testing.refAllDecls(@This());
}
