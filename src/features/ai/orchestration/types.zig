//! Shared public types for the orchestration module.

const std = @import("std");
const provider_types = @import("../llm/providers/types.zig");

pub const OrchestrationError = error{
    NoModelsAvailable,
    AllModelsFailed,
    ModelNotFound,
    DuplicateModelId,
    InvalidConfig,
    Timeout,
    InsufficientModelsForEnsemble,
    ModelDisabled,
    RateLimitExceeded,
    OutOfMemory,
    OrchestrationDisabled,
};

pub const ModelBackend = enum {
    openai,
    ollama,
    huggingface,
    anthropic,
    local,

    pub fn toString(self: ModelBackend) []const u8 {
        return @tagName(self);
    }

    pub fn toProviderId(self: ModelBackend) provider_types.ProviderId {
        return switch (self) {
            .openai => .openai,
            .ollama => .ollama,
            .huggingface => .openai,
            .anthropic => .anthropic,
            .local => .llama_cpp,
        };
    }
};

pub const Capability = enum {
    reasoning,
    coding,
    creative,
    analysis,
    summarization,
    translation,
    math,
    vision,
    embedding,

    pub fn toString(self: Capability) []const u8 {
        return @tagName(self);
    }
};

pub const RoutingStrategy = enum {
    round_robin,
    least_loaded,
    task_based,
    weighted,
    priority,
    cost_optimized,
    latency_optimized,

    pub fn toString(self: RoutingStrategy) []const u8 {
        return @tagName(self);
    }
};

pub const TaskType = enum {
    reasoning,
    coding,
    creative,
    analysis,
    summarization,
    translation,
    math,
    general,

    pub fn toString(self: TaskType) []const u8 {
        return @tagName(self);
    }

    pub fn detect(prompt: []const u8) TaskType {
        const lower = blk: {
            var buf: [512]u8 = undefined;
            const len = @min(prompt.len, buf.len);
            for (prompt[0..len], 0..) |c, i| {
                buf[i] = std.ascii.toLower(c);
            }
            break :blk buf[0..len];
        };

        if (std.mem.indexOf(u8, lower, "code") != null or
            std.mem.indexOf(u8, lower, "function") != null or
            std.mem.indexOf(u8, lower, "implement") != null or
            std.mem.indexOf(u8, lower, "debug") != null or
            std.mem.indexOf(u8, lower, "program") != null)
        {
            return .coding;
        }

        if (std.mem.indexOf(u8, lower, "calculate") != null or
            std.mem.indexOf(u8, lower, "solve") != null or
            std.mem.indexOf(u8, lower, "equation") != null or
            std.mem.indexOf(u8, lower, "math") != null)
        {
            return .math;
        }

        if (std.mem.indexOf(u8, lower, "summarize") != null or
            std.mem.indexOf(u8, lower, "summary") != null or
            std.mem.indexOf(u8, lower, "brief") != null or
            std.mem.indexOf(u8, lower, "tldr") != null)
        {
            return .summarization;
        }

        if (std.mem.indexOf(u8, lower, "translate") != null or
            std.mem.indexOf(u8, lower, "translation") != null)
        {
            return .translation;
        }

        if (std.mem.indexOf(u8, lower, "write") != null or
            std.mem.indexOf(u8, lower, "story") != null or
            std.mem.indexOf(u8, lower, "creative") != null or
            std.mem.indexOf(u8, lower, "poem") != null)
        {
            return .creative;
        }

        if (std.mem.indexOf(u8, lower, "analyze") != null or
            std.mem.indexOf(u8, lower, "analysis") != null or
            std.mem.indexOf(u8, lower, "examine") != null or
            std.mem.indexOf(u8, lower, "evaluate") != null)
        {
            return .analysis;
        }

        if (std.mem.indexOf(u8, lower, "reason") != null or
            std.mem.indexOf(u8, lower, "explain") != null or
            std.mem.indexOf(u8, lower, "why") != null or
            std.mem.indexOf(u8, lower, "think") != null)
        {
            return .reasoning;
        }

        return .general;
    }
};

pub const RouteResult = struct {
    model_id: []const u8,
    model_name: []const u8,
    backend: ModelBackend,
    prompt: []const u8,
    task_type: ?TaskType = null,
    confidence: f64 = 1.0,
    reason: ?[]const u8 = null,
};

pub const EnsembleMethod = enum {
    voting,
    averaging,
    weighted_average,
    best_of_n,
    concatenate,
    first_success,
    custom,

    pub fn toString(self: EnsembleMethod) []const u8 {
        return @tagName(self);
    }
};

pub const ModelResponse = struct {
    model_id: []const u8,
    response: []const u8,
    confidence: f64 = 1.0,
    latency_ms: u64 = 0,
    selected: bool = false,
};

pub const AggregationMetadata = struct {
    method: EnsembleMethod,
    total_responses: usize,
    failed_responses: usize,
    agreement_ratio: f64 = 0.0,
    avg_confidence: f64 = 0.0,
    std_deviation: f64 = 0.0,
};

pub const EnsembleResult = struct {
    response: []u8,
    model_count: usize,
    confidence: f64,
    individual_responses: ?[]const ModelResponse = null,
    metadata: ?AggregationMetadata = null,
};

pub const FallbackPolicy = enum {
    fail_fast,
    retry_then_fallback,
    immediate_fallback,
    circuit_breaker,

    pub fn toString(self: FallbackPolicy) []const u8 {
        return @tagName(self);
    }
};

pub const HealthStatus = enum {
    healthy,
    degraded,
    unhealthy,
    circuit_open,
    recovering,

    pub fn toString(self: HealthStatus) []const u8 {
        return @tagName(self);
    }

    pub fn isAvailable(self: HealthStatus) bool {
        return self == .healthy or self == .degraded;
    }
};

pub const OrchestrationConfig = struct {
    strategy: RoutingStrategy = .round_robin,
    enable_fallback: bool = true,
    enable_ensemble: bool = false,
    ensemble_method: EnsembleMethod = .voting,
    max_concurrent_requests: u32 = 100,
    timeout_ms: u64 = 30000,
    health_check_interval_ms: u64 = 60000,
    min_ensemble_models: u32 = 2,
    max_retries: u32 = 3,
    load_factor_weight: f32 = 0.5,

    pub fn defaults() OrchestrationConfig {
        return .{};
    }

    pub fn highAvailability() OrchestrationConfig {
        return .{
            .strategy = .least_loaded,
            .enable_fallback = true,
            .enable_ensemble = false,
            .max_concurrent_requests = 200,
            .max_retries = 5,
            .health_check_interval_ms = 30000,
        };
    }

    pub fn highQuality() OrchestrationConfig {
        return .{
            .strategy = .task_based,
            .enable_fallback = true,
            .enable_ensemble = true,
            .ensemble_method = .weighted_average,
            .min_ensemble_models = 3,
        };
    }
};

pub const ModelConfig = struct {
    id: []const u8,
    name: []const u8 = "",
    backend: ModelBackend = .openai,
    model_name: []const u8 = "",
    capabilities: []const Capability = &.{},
    priority: u32 = 100,
    weight: f32 = 1.0,
    max_tokens: u32 = 4096,
    cost_per_1k_tokens: f32 = 0.0,
    endpoint: ?[]const u8 = null,
    enabled: bool = true,
};

pub const ModelEntry = struct {
    config: ModelConfig,
    status: HealthStatus = .healthy,
    active_requests: u32 = 0,
    total_requests: u64 = 0,
    total_failures: u64 = 0,
    total_latency_ms: u64 = 0,
    last_request_time: i64 = 0,
    last_failure_time: i64 = 0,
    consecutive_failures: u32 = 0,

    pub fn avgLatencyMs(self: ModelEntry) f64 {
        if (self.total_requests == 0) return 0.0;
        return @as(f64, @floatFromInt(self.total_latency_ms)) /
            @as(f64, @floatFromInt(self.total_requests));
    }

    pub fn successRate(self: ModelEntry) f64 {
        if (self.total_requests == 0) return 1.0;
        const successes = self.total_requests - self.total_failures;
        return @as(f64, @floatFromInt(successes)) /
            @as(f64, @floatFromInt(self.total_requests));
    }

    pub fn loadFactor(self: ModelEntry, max_concurrent: u32) f64 {
        if (max_concurrent == 0) return 1.0;
        return @as(f64, @floatFromInt(self.active_requests)) /
            @as(f64, @floatFromInt(max_concurrent));
    }

    pub fn isAvailable(self: ModelEntry) bool {
        return self.config.enabled and self.status == .healthy;
    }
};

pub const OrchestratorStats = struct {
    total_models: u32 = 0,
    available_models: u32 = 0,
    total_requests: u64 = 0,
    total_failures: u64 = 0,
    active_requests: u32 = 0,

    pub fn successRate(self: OrchestratorStats) f64 {
        if (self.total_requests == 0) return 1.0;
        const successes = self.total_requests - self.total_failures;
        return @as(f64, @floatFromInt(successes)) /
            @as(f64, @floatFromInt(self.total_requests));
    }
};
