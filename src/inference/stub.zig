//! Inference stub — API-compatible no-ops when inference is disabled at compile time.

const std = @import("std");

// ── Sub-module namespace stubs ─────────────────────────────────────────────

pub const engine = struct {};
pub const scheduler = struct {};
pub const sampler = struct {};
pub const kv_cache = struct {};

// ── Re-exported type stubs ─────────────────────────────────────────────────

pub const Backend = enum { demo, connector, local };

pub const EngineConfig = struct {
    backend: Backend = .demo,
    max_tokens: u32 = 256,
    temperature: f32 = 0.7,
    model_name: []const u8 = "demo",
};

pub const FinishReason = enum { stop, length, error_reason };

pub const EngineResult = struct {
    request_id: u64 = 0,
    text: []const u8 = "",
    tokens_generated: u32 = 0,
    finish_reason: FinishReason = .error_reason,
    latency_ms: f32 = 0,
};

pub const EngineStats = struct {
    total_requests: u64 = 0,
    total_tokens: u64 = 0,
    avg_latency_ms: f32 = 0,
    tokens_per_second: f32 = 0,
};

pub const Engine = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, _: EngineConfig) !Engine {
        return .{ .allocator = allocator };
    }

    pub fn deinit(_: *Engine) void {}

    pub fn generate(_: *Engine, _: []const u8) error{FeatureDisabled}!EngineResult {
        return error.FeatureDisabled;
    }

    pub fn generateAsync(_: *Engine, _: []const u8, _: anytype) error{FeatureDisabled}!u64 {
        return error.FeatureDisabled;
    }

    pub fn getStats(_: *const Engine) EngineStats {
        return .{};
    }
};

pub const Request = struct {
    id: u64 = 0,
    prompt: []const u8 = "",
    priority: u8 = 0,
};

pub const Scheduler = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Scheduler {
        return .{ .allocator = allocator };
    }

    pub fn deinit(_: *Scheduler) void {}
};

pub const SamplingParams = struct {
    temperature: f32 = 0.7,
    top_k: u32 = 40,
    top_p: f32 = 0.95,
};

pub const Sampler = struct {
    params: SamplingParams = .{},

    pub fn init(params: SamplingParams) Sampler {
        return .{ .params = params };
    }

    pub fn sample(_: *const Sampler, _: []const f32) u32 {
        return 0;
    }
};

pub const PagedKVCacheConfig = struct {
    num_pages: u32 = 64,
    page_size: u32 = 16,
    num_heads: u32 = 8,
    head_dim: u32 = 64,
};

pub const PagedKVCache = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, _: PagedKVCacheConfig) !PagedKVCache {
        return .{ .allocator = allocator };
    }

    pub fn deinit(_: *PagedKVCache) void {}
};

test {
    std.testing.refAllDecls(@This());
}
