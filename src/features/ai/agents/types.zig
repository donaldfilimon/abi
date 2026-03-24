//! Shared public types for the AI agents module.

const std = @import("std");
const retry = @import("../../../foundation/mod.zig").utils.http_retry;
const config_module = @import("../../../core/config/mod.zig");
const provider_types = @import("../llm/providers/types.zig");

pub const Error = error{
    FeatureDisabled,
    AgentNotFound,
    ToolNotFound,
    ExecutionFailed,
    MaxAgentsReached,
};

pub const MIN_TEMPERATURE: f32 = 0.0;
pub const MAX_TEMPERATURE: f32 = 2.0;
pub const MIN_TOP_P: f32 = 0.0;
pub const MAX_TOP_P: f32 = 1.0;
pub const MAX_TOKENS_LIMIT: u32 = 128000;
pub const DEFAULT_TEMPERATURE: f32 = 0.7;
pub const DEFAULT_TOP_P: f32 = 0.9;
pub const DEFAULT_MAX_TOKENS: u32 = 1024;

pub const AgentError = error{
    InvalidConfiguration,
    OutOfMemory,
    ConnectorNotAvailable,
    GenerationFailed,
    ApiKeyMissing,
    HttpRequestFailed,
    InvalidApiResponse,
    RateLimitExceeded,
    Timeout,
    ConnectionRefused,
    ModelNotFound,
};

pub const AgentBackend = enum {
    echo,
    openai,
    anthropic,
    gemini,
    ollama,
    huggingface,
    codex,
    llama_cpp,
    local,
    provider_router,
};

pub const OperationContext = enum {
    initialization,
    configuration_validation,
    message_processing,
    response_generation,
    history_management,
    api_request,
    json_parsing,
    token_counting,

    pub fn toString(self: OperationContext) []const u8 {
        return switch (self) {
            .initialization => "initialization",
            .configuration_validation => "configuration validation",
            .message_processing => "message processing",
            .response_generation => "response generation",
            .history_management => "history management",
            .api_request => "API request",
            .json_parsing => "JSON parsing",
            .token_counting => "token counting",
        };
    }
};

pub const WorkloadType = enum {
    inference,
    training,
    embedding,
    fine_tuning,
    batch_inference,

    pub fn gpuIntensive(self: @This()) bool {
        return switch (self) {
            .training, .fine_tuning => true,
            else => false,
        };
    }

    pub fn memoryIntensive(self: @This()) bool {
        return switch (self) {
            .training, .fine_tuning, .batch_inference => true,
            else => false,
        };
    }

    pub fn name(self: @This()) []const u8 {
        return switch (self) {
            .inference => "Inference",
            .training => "Training",
            .embedding => "Embedding",
            .fine_tuning => "FineTuning",
            .batch_inference => "BatchInference",
        };
    }
};

pub const Priority = enum {
    low,
    normal,
    high,
    critical,

    pub fn weight(self: @This()) f32 {
        return switch (self) {
            .low => 0.25,
            .normal => 1.0,
            .high => 2.0,
            .critical => 4.0,
        };
    }

    pub fn name(self: @This()) []const u8 {
        return switch (self) {
            .low => "Low",
            .normal => "Normal",
            .high => "High",
            .critical => "Critical",
        };
    }
};

pub const ParameterType = enum { string, integer, boolean, array, object, number };

pub const ToolExecutionError = error{
    OutOfMemory,
    InvalidArguments,
    ExecutionFailed,
    Timeout,
    Cancelled,
    PermissionDenied,
    FileNotFound,
    ToolNotFound,
    InvalidState,
};

pub const ErrorContext = struct {
    @"error": AgentError,
    backend: AgentBackend,
    operation: OperationContext,
    http_status: ?u16 = null,
    endpoint: ?[]const u8 = null,
    model: ?[]const u8 = null,
    message: ?[]const u8 = null,
    retry_attempt: ?u32 = null,
    max_retries: ?u32 = null,

    pub fn apiError(
        err: AgentError,
        backend: AgentBackend,
        endpoint: []const u8,
        status: ?u16,
        model: ?[]const u8,
    ) ErrorContext {
        return .{
            .@"error" = err,
            .backend = backend,
            .operation = .api_request,
            .http_status = status,
            .endpoint = endpoint,
            .model = model,
        };
    }

    pub fn configError(err: AgentError, message: []const u8) ErrorContext {
        return .{
            .@"error" = err,
            .backend = .echo,
            .operation = .configuration_validation,
            .message = message,
        };
    }

    pub fn generationError(
        err: AgentError,
        backend: AgentBackend,
        model: []const u8,
        message: ?[]const u8,
    ) ErrorContext {
        return .{
            .@"error" = err,
            .backend = backend,
            .operation = .response_generation,
            .model = model,
            .message = message,
        };
    }

    pub fn retryError(
        err: AgentError,
        backend: AgentBackend,
        endpoint: []const u8,
        attempt: u32,
        max_attempts: u32,
    ) ErrorContext {
        return .{
            .@"error" = err,
            .backend = backend,
            .operation = .api_request,
            .endpoint = endpoint,
            .retry_attempt = attempt,
            .max_retries = max_attempts,
        };
    }

    pub fn format(self: ErrorContext, writer: anytype) !void {
        try writer.print("AgentError: {t} during {s}", .{
            self.@"error",
            self.operation.toString(),
        });

        try writer.print(" [backend={t}]", .{self.backend});

        if (self.model) |model| {
            try writer.print(" [model={s}]", .{model});
        }

        if (self.endpoint) |endpoint| {
            try writer.print(" [endpoint={s}]", .{endpoint});
        }

        if (self.http_status) |status| {
            try writer.print(" [http_status={d}]", .{status});
        }

        if (self.retry_attempt) |attempt| {
            if (self.max_retries) |max_attempts| {
                try writer.print(" [attempt={d}/{d}]", .{ attempt, max_attempts });
            }
        }

        if (self.message) |message| {
            try writer.print(" - {s}", .{message});
        }
    }

    pub fn formatToString(self: ErrorContext, allocator: std.mem.Allocator) ![]u8 {
        var aw = std.Io.Writer.Allocating.init(allocator);
        errdefer aw.deinit();
        try self.format(&aw.writer);
        return aw.toOwnedSlice();
    }

    pub fn log(self: ErrorContext) void {
        var buf: [512]u8 = undefined;
        var writer = std.Io.Writer.fixed(&buf);
        self.format(&writer) catch return;
        std.log.err("{s}", .{buf[0..writer.end]});
    }
};

pub const BackendMetrics = struct {
    success_count: u32 = 0,
    failure_count: u32 = 0,
    total_latency_ms: u64 = 0,
    total_quality: f32 = 0.0,

    pub fn record(self: *BackendMetrics, success: bool, latency_ms: u64, quality: f32) void {
        if (success) {
            self.success_count += 1;
        } else {
            self.failure_count += 1;
        }
        self.total_latency_ms += latency_ms;
        self.total_quality += quality;
    }

    pub fn totalCalls(self: BackendMetrics) u32 {
        return self.success_count + self.failure_count;
    }

    pub fn successRate(self: BackendMetrics) f32 {
        const total = self.totalCalls();
        if (total == 0) return 0.0;
        return @as(f32, @floatFromInt(self.success_count)) / @as(f32, @floatFromInt(total));
    }

    pub fn avgLatencyMs(self: BackendMetrics) f32 {
        const total = self.totalCalls();
        if (total == 0) return 0.0;
        return @as(f32, @floatFromInt(self.total_latency_ms)) / @as(f32, @floatFromInt(total));
    }

    pub fn avgQuality(self: BackendMetrics) f32 {
        const total = self.totalCalls();
        if (total == 0) return 0.0;
        return self.total_quality / @as(f32, @floatFromInt(total));
    }

    pub fn score(self: BackendMetrics) f32 {
        if (self.totalCalls() == 0) return 0.5;
        return self.successRate() * 0.4 +
            self.avgQuality() * 0.4 +
            (1.0 / (1.0 + self.avgLatencyMs() / 1000.0)) * 0.2;
    }
};

pub const AgentConfig = struct {
    name: []const u8 = "",
    enable_history: bool = true,
    temperature: f32 = DEFAULT_TEMPERATURE,
    top_p: f32 = DEFAULT_TOP_P,
    max_tokens: u32 = DEFAULT_MAX_TOKENS,
    backend: AgentBackend = .echo,
    model: []const u8 = "gpt-4",
    provider_backend: ?provider_types.ProviderId = null,
    provider_fallback: []const provider_types.ProviderId = &.{},
    provider_strict_backend: bool = false,
    provider_plugin_id: ?[]const u8 = null,
    system_prompt: ?[]const u8 = null,
    retry_config: retry.RetryConfig = .{},

    pub fn validate(self: AgentConfig) AgentError!void {
        if (self.name.len == 0) return AgentError.InvalidConfiguration;
        if (self.temperature < MIN_TEMPERATURE or self.temperature > MAX_TEMPERATURE) {
            return AgentError.InvalidConfiguration;
        }
        if (self.top_p < MIN_TOP_P or self.top_p > MAX_TOP_P) {
            return AgentError.InvalidConfiguration;
        }
        if (self.max_tokens == 0 or self.max_tokens > MAX_TOKENS_LIMIT) {
            return AgentError.InvalidConfiguration;
        }
    }
};

pub const Message = struct {
    role: Role,
    content: []const u8,

    pub const Role = enum {
        system,
        user,
        assistant,
    };
};

pub const Parameter = struct {
    name: []const u8,
    type: ParameterType,
    required: bool = false,
    description: []const u8 = "",
    enum_values: ?[]const []const u8 = null,
};

pub const ToolResult = struct {
    allocator: std.mem.Allocator,
    success: bool,
    output: []const u8,
    error_message: ?[]const u8,
    metadata: ?std.json.ObjectMap,

    pub fn init(allocator: std.mem.Allocator, success: bool, output: []const u8) ToolResult {
        return .{
            .allocator = allocator,
            .success = success,
            .output = output,
            .error_message = null,
            .metadata = null,
        };
    }

    pub fn fromError(allocator: std.mem.Allocator, err: []const u8) ToolResult {
        return .{
            .allocator = allocator,
            .success = false,
            .output = "",
            .error_message = err,
            .metadata = null,
        };
    }

    pub fn deinit(self: *ToolResult) void {
        if (self.metadata) |*obj| {
            obj.deinit();
        }
    }
};

pub const ToolContext = struct {
    allocator: std.mem.Allocator,
    working_directory: []const u8,
    environment: ?*const std.StringHashMapUnmanaged([]const u8),
    cancellation: ?*const std.atomic.Value(bool),
};

pub const Tool = struct {
    name: []const u8,
    description: []const u8,
    parameters: []const Parameter,
    execute: *const fn (*ToolContext, std.json.Value) ToolExecutionError!ToolResult,
};

pub const GpuAwareRequest = struct {
    prompt: []const u8,
    workload_type: WorkloadType,
    priority: Priority = .normal,
    max_tokens: u32 = 1024,
    temperature: f32 = 0.7,
    memory_hint_mb: ?u64 = null,
    preferred_backend: ?[]const u8 = null,
    model_id: ?[]const u8 = null,
    stream: bool = false,
    timeout_ms: u64 = 0,
};

pub const GpuAwareResponse = struct {
    content: []const u8,
    tokens_generated: u32,
    latency_ms: u64,
    gpu_backend_used: []const u8,
    gpu_memory_used_mb: u64,
    scheduling_confidence: f32,
    energy_estimate_wh: ?f32 = null,
    device_id: u32 = 0,
    truncated: bool = false,
    error_message: ?[]const u8 = null,
};

pub const GpuAgentStats = struct {
    total_requests: u64 = 0,
    gpu_accelerated: u64 = 0,
    cpu_fallback: u64 = 0,
    total_tokens: u64 = 0,
    total_latency_ms: u64 = 0,
    learning_episodes: u64 = 0,
    avg_scheduling_confidence: f32 = 0,
    avg_latency_ms: f32 = 0,
    failed_requests: u64 = 0,
    total_gpu_memory_mb: u64 = 0,

    pub fn updateConfidence(self: *GpuAgentStats, new_confidence: f32) void {
        if (self.gpu_accelerated == 0) {
            self.avg_scheduling_confidence = new_confidence;
        } else {
            const n = @as(f32, @floatFromInt(self.gpu_accelerated));
            self.avg_scheduling_confidence = (self.avg_scheduling_confidence * (n - 1) + new_confidence) / n;
        }
    }

    pub fn updateLatency(self: *GpuAgentStats, latency: u64) void {
        if (self.total_requests == 0) {
            self.avg_latency_ms = @floatFromInt(latency);
        } else {
            const n = @as(f32, @floatFromInt(self.total_requests));
            self.avg_latency_ms = (self.avg_latency_ms * (n - 1) + @as(f32, @floatFromInt(latency))) / n;
        }
    }

    pub fn successRate(self: GpuAgentStats) f32 {
        if (self.total_requests == 0) return 1.0;
        return @as(f32, @floatFromInt(self.total_requests - self.failed_requests)) /
            @as(f32, @floatFromInt(self.total_requests));
    }

    pub fn gpuUtilizationRate(self: GpuAgentStats) f32 {
        if (self.total_requests == 0) return 0.0;
        return @as(f32, @floatFromInt(self.gpu_accelerated)) /
            @as(f32, @floatFromInt(self.total_requests));
    }

    pub fn avgTokensPerRequest(self: GpuAgentStats) f32 {
        if (self.total_requests == 0) return 0.0;
        return @as(f32, @floatFromInt(self.total_tokens)) /
            @as(f32, @floatFromInt(self.total_requests));
    }
};

pub const BackendInfo = struct {
    name: []const u8 = "cpu",
    device_count: u32 = 0,
    total_memory_mb: u64 = 0,
    available_memory_mb: u64 = 0,
    is_healthy: bool = false,
};

pub const LearningStatsInfo = struct {
    episodes: usize = 0,
    avg_episode_reward: f32 = 0,
    exploration_rate: f32 = 0,
    replay_buffer_size: usize = 0,
};

pub const AgentStats = struct {
    history_length: usize,
    user_messages: usize,
    assistant_messages: usize,
    total_characters: usize,
    total_tokens_used: u64,
};

pub const Agent = struct {
    allocator: std.mem.Allocator,
    config: AgentConfig,
    history: std.ArrayListUnmanaged(Message) = .empty,
    total_tokens_used: u64 = 0,

    pub fn init(_: std.mem.Allocator, _: AgentConfig) AgentError!Agent {
        return error.ConnectorNotAvailable;
    }

    pub fn deinit(_: *Agent) void {}

    pub fn historyCount(self: *const Agent) usize {
        return self.history.items.len;
    }

    pub fn historySlice(self: *const Agent) []const Message {
        return self.history.items;
    }

    pub fn historyStrings(_: *const Agent, _: std.mem.Allocator) ?[]const []const u8 {
        return &.{};
    }

    pub fn clearHistory(_: *Agent) void {}

    pub fn chat(_: *Agent, _: []const u8, _: std.mem.Allocator) AgentError![]const u8 {
        return error.ConnectorNotAvailable;
    }

    pub fn process(_: *Agent, _: []const u8, _: std.mem.Allocator) AgentError![]const u8 {
        return error.ConnectorNotAvailable;
    }

    pub fn name(self: *const Agent) []const u8 {
        return self.config.name;
    }

    pub fn setTemperature(_: *Agent, _: f32) AgentError!void {
        return error.ConnectorNotAvailable;
    }

    pub fn setTopP(_: *Agent, _: f32) AgentError!void {
        return error.ConnectorNotAvailable;
    }

    pub fn setMaxTokens(_: *Agent, _: u32) AgentError!void {
        return error.ConnectorNotAvailable;
    }

    pub fn setBackend(_: *Agent, _: AgentBackend) void {}

    pub fn setModel(_: *Agent, _: []const u8) !void {}

    pub fn setHistoryEnabled(_: *Agent, _: bool) void {}

    pub fn getTotalTokensUsed(self: *const Agent) u64 {
        return self.total_tokens_used;
    }

    pub fn getStats(_: *const Agent) AgentStats {
        return .{
            .history_length = 0,
            .user_messages = 0,
            .assistant_messages = 0,
            .total_characters = 0,
            .total_tokens_used = 0,
        };
    }
};

pub const GpuAgent = struct {
    allocator: std.mem.Allocator,
    stats: GpuAgentStats = .{},

    pub fn init(_: std.mem.Allocator) !*@This() {
        return error.FeatureDisabled;
    }

    pub fn initWithConfig(
        _: std.mem.Allocator,
        _: struct {
            default_timeout_ms: u64 = 30000,
            enable_learning: bool = true,
        },
    ) !*@This() {
        return error.FeatureDisabled;
    }

    pub fn deinit(_: *@This()) void {}

    pub fn process(_: *@This(), _: GpuAwareRequest) !GpuAwareResponse {
        return error.FeatureDisabled;
    }

    pub fn getStats(self: *const @This()) GpuAgentStats {
        return self.stats;
    }

    pub fn isGpuEnabled(_: *const @This()) bool {
        return false;
    }

    pub fn isLearningEnabled(_: *const @This()) bool {
        return false;
    }

    pub fn endEpisode(_: *@This()) void {}

    pub fn resetStats(self: *@This()) void {
        self.stats = .{};
    }

    pub fn getBackendsSummary(_: *@This(), _: std.mem.Allocator) ![]const BackendInfo {
        return error.FeatureDisabled;
    }

    pub fn getLearningStats(_: *@This()) ?LearningStatsInfo {
        return null;
    }
};

pub const Context = struct {
    pub fn init(_: std.mem.Allocator, _: config_module.AgentsConfig) Error!*Context {
        return error.FeatureDisabled;
    }

    pub fn deinit(_: *Context) void {}

    pub fn createAgent(_: *Context, _: []const u8) Error!*Agent {
        return error.FeatureDisabled;
    }

    pub fn getAgent(_: *Context, _: []const u8) ?*Agent {
        return null;
    }

    pub fn getToolRegistry(_: *Context) Error!*ToolRegistry {
        return error.FeatureDisabled;
    }

    pub fn registerTool(_: *Context, _: Tool) Error!void {
        return error.FeatureDisabled;
    }
};

pub const ToolRegistry = struct {
    allocator: std.mem.Allocator,
    tools: std.StringHashMapUnmanaged(*const Tool),

    pub fn init(allocator: std.mem.Allocator) ToolRegistry {
        return .{
            .allocator = allocator,
            .tools = .empty,
        };
    }

    pub fn deinit(self: *ToolRegistry) void {
        self.tools.deinit(self.allocator);
    }

    pub fn register(self: *ToolRegistry, tool: *const Tool) !void {
        try self.tools.put(self.allocator, tool.name, tool);
    }

    pub fn get(self: *ToolRegistry, name: []const u8) ?*const Tool {
        return self.tools.get(name);
    }

    pub fn count(self: *const ToolRegistry) usize {
        return self.tools.count();
    }

    pub fn contains(self: *const ToolRegistry, name: []const u8) bool {
        return self.tools.contains(name);
    }
};
