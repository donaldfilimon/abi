//! Orchestration stub types — extracted from stub.zig.

const std = @import("std");

// --- Errors ---

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
    FeatureDisabled,
};

// --- Config Types ---

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
        return .{};
    }
    pub fn highQuality() OrchestrationConfig {
        return .{};
    }
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
    pub fn toProviderId(_: ModelBackend) u8 {
        return 0;
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

// --- Router ---

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
    pub fn detect(_: []const u8) TaskType {
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

// --- Ensemble ---

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

pub const EnsembleResult = struct { response: []u8, model_count: usize, confidence: f64 };

// --- Fallback ---

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

// --- Model Entry ---

pub const ModelEntry = struct {
    config: ModelConfig,
    status: HealthStatus = .healthy,
    active_requests: u32 = 0,
    total_requests: u64 = 0,
    total_failures: u64 = 0,
    total_latency_ms: u64 = 0,
    pub fn avgLatencyMs(_: ModelEntry) f64 {
        return 0.0;
    }
    pub fn successRate(_: ModelEntry) f64 {
        return 1.0;
    }
    pub fn loadFactor(_: ModelEntry, _: u32) f64 {
        return 0.0;
    }
    pub fn isAvailable(_: ModelEntry) bool {
        return false;
    }
};

// --- Stats ---

pub const OrchestratorStats = struct {
    total_models: u32 = 0,
    available_models: u32 = 0,
    total_requests: u64 = 0,
    total_failures: u64 = 0,
    active_requests: u32 = 0,
    pub fn successRate(_: OrchestratorStats) f64 {
        return 0.0;
    }
};
