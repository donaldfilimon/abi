//! Stub implementation for enhanced streaming when AI features are disabled.

const std = @import("std");
const transformer = @import("../transformer/mod.zig");
const stub_root = @This();

pub const sse = struct {
    pub const SseEvent = stub_root.SseEvent;
    pub const SseEncoder = stub_root.SseEncoder;
    pub const SseDecoder = stub_root.SseDecoder;
    pub const SseConfig = stub_root.SseConfig;
};

pub const backpressure = struct {
    pub const BackpressureController = stub_root.BackpressureController;
    pub const BackpressureStrategy = stub_root.BackpressureStrategy;
    pub const BackpressureConfig = stub_root.BackpressureConfig;
    pub const FlowState = stub_root.FlowState;
    pub const BackpressureStats = stub_root.BackpressureStats;
    pub const RateLimiter = stub_root.RateLimiter;
};

pub const buffer = struct {
    pub const TokenBuffer = stub_root.TokenBuffer;
    pub const BufferConfig = stub_root.BufferConfig;
    pub const BufferStrategy = stub_root.BufferStrategy;
    pub const BufferStats = stub_root.BufferStats;
    pub const CoalescingBuffer = stub_root.CoalescingBuffer;
};

pub const generator = struct {
    pub const StreamingGenerator = stub_root.StreamingGenerator;
    pub const StreamingError = stub_root.StreamingError;
    pub const StreamState = stub_root.StreamState;
    pub const GenerationConfig = stub_root.GenerationConfig;
    pub const streamInference = stub_root.streamInference;
    pub const formatStreamOutput = stub_root.formatStreamOutput;
    pub const createChunkedStream = stub_root.createChunkedStream;
};

// Streaming server stubs
pub const server = struct {
    pub const StreamingServer = stub_root.StreamingServer;
    pub const ServerConfig = stub_root.ServerConfig;
    pub const StreamingServerError = stub_root.StreamingServerError;
};

pub const websocket = struct {
    pub const WebSocketHandler = stub_root.WebSocketHandler;
    pub const WebSocketConfig = stub_root.WebSocketConfig;
    pub const Opcode = stub_root.WebSocketOpcode;
    pub const CloseCode = stub_root.WebSocketCloseCode;
    pub const computeAcceptKey = stub_root.computeWebSocketAcceptKey;
};

pub const backends = struct {
    pub const BackendType = stub_root.BackendType;
    pub const BackendRouter = stub_root.BackendRouter;
    pub const Backend = stub_root.Backend;
    pub const GenerationConfig = stub_root.BackendGenerationConfig;
};

pub const formats = struct {
    pub const openai = stub_root.formats_openai;
};

// Error recovery and resilience stubs
pub const recovery = struct {
    pub const StreamRecovery = stub_root.StreamRecovery;
    pub const RecoveryConfig = stub_root.RecoveryConfig;
    pub const RecoveryEvent = stub_root.RecoveryEvent;
    pub const RecoveryCallback = stub_root.RecoveryCallback;
};

pub const circuit_breaker = struct {
    pub const CircuitBreaker = stub_root.CircuitBreaker;
    pub const CircuitBreakerConfig = stub_root.CircuitBreakerConfig;
    pub const CircuitState = stub_root.CircuitState;
};

pub const retry_config = struct {
    pub const StreamingRetryConfig = stub_root.StreamingRetryConfig;
    pub const StreamingRetryableErrors = stub_root.StreamingRetryableErrors;
};

pub const session_cache = struct {
    pub const SessionCache = stub_root.SessionCache;
    pub const SessionCacheConfig = stub_root.SessionCacheConfig;
    pub const CachedToken = stub_root.CachedToken;
};

pub const streaming_metrics = struct {
    pub const StreamingMetrics = stub_root.StreamingMetrics;
    pub const StreamingMetricsConfig = stub_root.StreamingMetricsConfig;
};

/// Stub SSE event.
pub const SseEvent = struct {
    event: ?[]const u8 = null,
    data: []const u8 = "",
    id: ?[]const u8 = null,
    retry: ?u32 = null,

    pub fn deinit(self: *SseEvent, allocator: std.mem.Allocator) void {
        _ = allocator;
        self.* = undefined;
    }
};

/// Stub SSE config.
pub const SseConfig = struct {
    event_prefix: []const u8 = "",
    include_timestamp: bool = false,
    include_id: bool = true,
    retry_ms: ?u32 = null,
};

/// Stub SSE encoder.
pub const SseEncoder = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: SseConfig) SseEncoder {
        _ = config;
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *SseEncoder) void {
        self.* = undefined;
    }

    pub fn encode(self: *SseEncoder, event: StreamEvent) ![]u8 {
        _ = self;
        _ = event;
        return error.StreamingDisabled;
    }
};

/// Stub SSE decoder.
pub const SseDecoder = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) SseDecoder {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *SseDecoder) void {
        self.* = undefined;
    }

    pub fn feed(self: *SseDecoder, data: []const u8) ![]SseEvent {
        _ = self;
        _ = data;
        return error.StreamingDisabled;
    }
};

/// Stub backpressure strategy.
pub const BackpressureStrategy = enum {
    drop,
    block,
    buffer,
    sample,
    adaptive,
};

/// Stub backpressure config.
pub const BackpressureConfig = struct {
    strategy: BackpressureStrategy = .buffer,
    high_watermark: usize = 100,
    low_watermark: usize = 25,
    max_buffer: usize = 1000,
};

/// Stub flow state.
pub const FlowState = enum {
    normal,
    throttled,
    blocked,
    recovering,
};

/// Stub backpressure controller.
pub const BackpressureController = struct {
    pub fn init(config: BackpressureConfig) BackpressureController {
        _ = config;
        return .{};
    }

    pub fn checkFlow(self: *BackpressureController) FlowState {
        _ = self;
        return .normal;
    }

    pub fn produce(self: *BackpressureController) void {
        _ = self;
    }

    pub fn consume(self: *BackpressureController) void {
        _ = self;
    }
};

/// Stub backpressure statistics.
pub const BackpressureStats = struct {
    pending_count: usize = 0,
    dropped_count: usize = 0,
    total_processed: u64 = 0,
    current_tps: f64 = 0,
    state: FlowState = .normal,
    utilization: f64 = 0,
};

/// Stub rate limiter.
pub const RateLimiter = struct {
    tokens_per_second: f64 = 0,
    bucket_size: ?f64 = null,

    pub fn init(tokens_per_second: f64, bucket_size: ?f64) RateLimiter {
        return .{
            .tokens_per_second = tokens_per_second,
            .bucket_size = bucket_size,
        };
    }

    pub fn tryAcquire(_: *RateLimiter) bool {
        return false;
    }

    pub fn acquire(_: *RateLimiter) void {}

    pub fn getAvailable(_: *RateLimiter) f64 {
        return 0;
    }

    pub fn reset(_: *RateLimiter) void {}
};

/// Stub buffer strategy.
pub const BufferStrategy = enum {
    fifo,
    lifo,
    priority,
    ring,
};

/// Stub buffer config.
pub const BufferConfig = struct {
    strategy: BufferStrategy = .fifo,
    capacity: usize = 100,
    flush_threshold: usize = 50,
    coalesce: bool = false,
};

/// Stub stream token.
pub const StreamToken = struct {
    id: u32 = 0,
    text: []const u8 = "",
    log_prob: ?f32 = null,
    is_end: bool = false,
    timestamp_ns: i128 = 0,
    sequence_index: usize = 0,

    pub fn clone(self: StreamToken, allocator: std.mem.Allocator) !StreamToken {
        return .{
            .id = self.id,
            .text = try allocator.dupe(u8, self.text),
            .log_prob = self.log_prob,
            .is_end = self.is_end,
            .timestamp_ns = self.timestamp_ns,
            .sequence_index = self.sequence_index,
        };
    }

    pub fn deinit(self: *StreamToken, allocator: std.mem.Allocator) void {
        allocator.free(self.text);
        self.* = undefined;
    }
};

/// Stub stream event type.
pub const StreamEventType = enum {
    token,
    start,
    end,
    error_event,
    metadata,
    heartbeat,
};

/// Stub stream event.
pub const StreamEvent = struct {
    event_type: StreamEventType = .token,
    token: ?StreamToken = null,
    metadata: ?[]const u8 = null,
    error_message: ?[]const u8 = null,
    timestamp_ns: i128 = 0,

    pub fn tokenEvent(token: StreamToken) StreamEvent {
        return .{
            .event_type = .token,
            .token = token,
        };
    }

    pub fn startEvent() StreamEvent {
        return .{ .event_type = .start };
    }

    pub fn endEvent() StreamEvent {
        return .{ .event_type = .end };
    }

    pub fn errorEvent(message: []const u8) StreamEvent {
        return .{
            .event_type = .error_event,
            .error_message = message,
        };
    }

    pub fn heartbeatEvent() StreamEvent {
        return .{ .event_type = .heartbeat };
    }
};

const GeneratorStreamToken = struct {
    id: u32,
    text: []const u8,
    log_prob: ?f32 = null,
    is_end: bool = false,
};

/// Stub streaming generator error set.
pub const StreamingError = error{
    StreamingDisabled,
    StreamClosed,
    InvalidState,
    GenerationFailed,
};

/// Stub streaming state.
pub const StreamState = enum {
    idle,
    generating,
    paused,
    completed,
    failed,
};

/// Stub generation config.
pub const GenerationConfig = struct {
    max_tokens: u32 = 256,
    temperature: f32 = 0.8,
    top_p: f32 = 0.9,
    top_k: u32 = 40,
    repeat_penalty: f32 = 1.1,
    presence_penalty: f32 = 0.0,
    frequency_penalty: f32 = 0.0,
    stop_tokens: []const []const u8 = &.{},
};

/// Stub streaming generator.
pub const StreamingGenerator = struct {
    allocator: std.mem.Allocator,
    config: GenerationConfig,
    state: StreamState,

    pub fn init(
        allocator: std.mem.Allocator,
        model: *transformer.TransformerModel,
        config: GenerationConfig,
    ) StreamingGenerator {
        _ = model;
        return .{
            .allocator = allocator,
            .config = config,
            .state = .idle,
        };
    }

    pub fn deinit(self: *StreamingGenerator) void {
        _ = self;
    }

    pub fn start(self: *StreamingGenerator, prompt: []const u8) StreamingError!void {
        _ = self;
        _ = prompt;
        return error.StreamingDisabled;
    }

    pub fn next(self: *StreamingGenerator) StreamingError!?GeneratorStreamToken {
        _ = self;
        return error.StreamingDisabled;
    }

    pub fn pause(self: *StreamingGenerator) void {
        _ = self;
    }

    pub fn resumeGeneration(self: *StreamingGenerator) void {
        _ = self;
    }

    pub fn cancel(self: *StreamingGenerator) void {
        _ = self;
    }

    pub fn reset(self: *StreamingGenerator, new_config: GenerationConfig) void {
        self.config = new_config;
        self.state = .idle;
    }

    pub fn getGeneratedText(self: *StreamingGenerator, allocator: std.mem.Allocator) ![]u8 {
        _ = self;
        _ = allocator;
        return error.StreamingDisabled;
    }

    pub fn tokenCount(self: *const StreamingGenerator) usize {
        _ = self;
        return 0;
    }

    pub fn isComplete(self: *const StreamingGenerator) bool {
        return self.state == .completed;
    }
};

pub fn streamInference(
    allocator: std.mem.Allocator,
    model: *transformer.TransformerModel,
    prompt: []const u8,
    config: GenerationConfig,
    callback: anytype,
) !void {
    _ = allocator;
    _ = model;
    _ = prompt;
    _ = config;
    _ = callback;
    return error.StreamingDisabled;
}

pub fn formatStreamOutput(tokens: []const GeneratorStreamToken, allocator: std.mem.Allocator) ![]u8 {
    _ = tokens;
    _ = allocator;
    return error.StreamingDisabled;
}

pub fn createChunkedStream(
    allocator: std.mem.Allocator,
    tokens: []const GeneratorStreamToken,
    chunk_size: usize,
) ![]const []const u8 {
    _ = allocator;
    _ = tokens;
    _ = chunk_size;
    return error.StreamingDisabled;
}

/// Stub token buffer.
pub const TokenBuffer = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: BufferConfig) TokenBuffer {
        _ = config;
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *TokenBuffer) void {
        self.* = undefined;
    }

    pub fn push(self: *TokenBuffer, token: StreamToken) !void {
        _ = self;
        _ = token;
        return error.StreamingDisabled;
    }

    pub fn pop(self: *TokenBuffer) ?StreamToken {
        _ = self;
        return null;
    }

    pub fn peek(self: *const TokenBuffer) ?StreamToken {
        _ = self;
        return null;
    }

    pub fn len(self: *const TokenBuffer) usize {
        _ = self;
        return 0;
    }

    pub fn isEmpty(self: *const TokenBuffer) bool {
        _ = self;
        return true;
    }

    pub fn isFull(self: *const TokenBuffer) bool {
        _ = self;
        return false;
    }

    pub fn shouldFlush(self: *const TokenBuffer) bool {
        _ = self;
        return false;
    }

    pub fn clear(self: *TokenBuffer) void {
        _ = self;
    }

    pub fn getStats(self: *const TokenBuffer) BufferStats {
        _ = self;
        return .{};
    }

    pub fn flushAsText(self: *TokenBuffer) ![]u8 {
        _ = self;
        return error.StreamingDisabled;
    }
};

/// Stub buffer stats.
pub const BufferStats = struct {
    current_size: usize = 0,
    capacity: usize = 0,
    total_pushed: u64 = 0,
    total_popped: u64 = 0,
    total_dropped: u64 = 0,
    utilization: f64 = 0,
};

/// Stub coalescing buffer.
pub const CoalescingBuffer = struct {
    allocator: std.mem.Allocator,
    max_length: usize,

    pub fn init(allocator: std.mem.Allocator, max_length: usize) CoalescingBuffer {
        return .{
            .allocator = allocator,
            .max_length = max_length,
        };
    }

    pub fn deinit(self: *CoalescingBuffer) void {
        _ = self;
    }

    pub fn add(self: *CoalescingBuffer, text: []const u8) !?[]u8 {
        _ = self;
        _ = text;
        return error.StreamingDisabled;
    }

    pub fn flush(self: *CoalescingBuffer) ![]u8 {
        _ = self;
        return error.StreamingDisabled;
    }

    pub fn len(self: *const CoalescingBuffer) usize {
        _ = self;
        return 0;
    }

    pub fn isEmpty(self: *const CoalescingBuffer) bool {
        _ = self;
        return true;
    }
};

/// Stub stream config.
pub const StreamConfig = struct {
    sse_config: SseConfig = .{},
    backpressure_config: BackpressureConfig = .{},
    buffer_config: BufferConfig = .{},
};

/// Stub stream stats.
pub const StreamStats = struct {
    /// Total tokens generated.
    total_tokens: usize = 0,
    /// Total characters generated.
    total_chars: usize = 0,
    /// Tokens per second.
    tokens_per_second: f64 = 0,
    /// Start time (elapsed ns from timer).
    start_time_ns: u64 = 0,
    /// End time (elapsed ns from timer).
    end_time_ns: u64 = 0,
    /// Number of pauses.
    pause_count: usize = 0,
    /// Time spent paused (ns).
    pause_duration_ns: u64 = 0,

    pub fn duration_ms(self: *const StreamStats) f64 {
        const duration_ns = self.end_time_ns - self.start_time_ns - self.pause_duration_ns;
        return @as(f64, @floatFromInt(duration_ns)) / 1_000_000.0;
    }
};

/// Stub enhanced streaming generator.
pub const EnhancedStreamingGenerator = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: StreamConfig) EnhancedStreamingGenerator {
        _ = config;
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *EnhancedStreamingGenerator) void {
        self.* = undefined;
    }

    pub fn start(self: *EnhancedStreamingGenerator) !void {
        _ = self;
        return error.StreamingDisabled;
    }

    pub fn emit(self: *EnhancedStreamingGenerator, token: StreamToken) !?[]u8 {
        _ = self;
        _ = token;
        return error.StreamingDisabled;
    }

    /// Flush buffered tokens.
    pub fn flush(self: *EnhancedStreamingGenerator) ![][]u8 {
        _ = self;
        return error.StreamingDisabled;
    }

    pub fn complete(self: *EnhancedStreamingGenerator) ![]u8 {
        _ = self;
        return error.StreamingDisabled;
    }

    pub fn getStats(self: *const EnhancedStreamingGenerator) StreamStats {
        _ = self;
        return .{};
    }
};

/// Stub factory function.
pub fn createSseStream(allocator: std.mem.Allocator, chunks: []const []const u8) ![]u8 {
    _ = allocator;
    _ = chunks;
    return error.StreamingDisabled;
}

// ============================================================================
// Streaming Server Stubs (new in streaming inference API)
// ============================================================================

/// Stub streaming server error.
pub const StreamingServerError = std.mem.Allocator.Error || error{
    StreamingDisabled,
    InvalidAddress,
    InvalidRequest,
    Unauthorized,
    BackendError,
    StreamError,
    WebSocketError,
    RequestTooLarge,
    UnsupportedBackend,
    ModelReloadFailed,
    ModelReloadTimeout,
    CircuitBreakerOpen,
};

/// Stub server config.
pub const ServerConfig = struct {
    /// Listen address (e.g., "127.0.0.1:8080")
    address: []const u8 = "127.0.0.1:8080",
    /// Bearer token for authentication (null = no auth required)
    auth_token: ?[]const u8 = null,
    /// Allow health endpoint without auth
    allow_health_without_auth: bool = true,
    /// Default backend for inference
    default_backend: BackendType = .local,
    /// Heartbeat interval in milliseconds (0 = disabled)
    heartbeat_interval_ms: u64 = 15000,
    /// Maximum concurrent streams
    max_concurrent_streams: u32 = 100,
    /// Enable OpenAI-compatible endpoints
    enable_openai_compat: bool = true,
    /// Enable WebSocket support
    enable_websocket: bool = true,
    /// Path to default local model (optional, for local backend)
    default_model_path: ?[]const u8 = null,
    /// Pre-load model on server start (reduces first-request latency)
    preload_model: bool = false,
    /// Enable error recovery (circuit breakers, retry, session caching)
    enable_recovery: bool = true,
    /// Recovery configuration (only used if enable_recovery is true)
    recovery_config: RecoveryConfig = .{},
};

/// Stub streaming server.
pub const StreamingServer = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: ServerConfig) !StreamingServer {
        _ = config;
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *StreamingServer) void {
        self.* = undefined;
    }

    pub fn serve(self: *StreamingServer) !void {
        _ = self;
        return error.StreamingDisabled;
    }
};

/// Stub WebSocket opcode.
pub const WebSocketOpcode = enum(u4) {
    continuation = 0x0,
    text = 0x1,
    binary = 0x2,
    close = 0x8,
    ping = 0x9,
    pong = 0xA,
};

/// Stub WebSocket close code.
pub const WebSocketCloseCode = enum(u16) {
    normal = 1000,
    going_away = 1001,
    protocol_error = 1002,
    _,
};

/// Stub WebSocket config.
pub const WebSocketConfig = struct {
    max_message_size: usize = 16 * 1024 * 1024,
    ping_interval_ms: u64 = 30000,
    enable_compression: bool = false,
};

/// Stub WebSocket handler.
pub const WebSocketHandler = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: WebSocketConfig) !WebSocketHandler {
        _ = config;
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *WebSocketHandler) void {
        self.* = undefined;
    }

    pub fn sendText(self: *WebSocketHandler, text: []const u8) ![]u8 {
        _ = self;
        _ = text;
        return error.StreamingDisabled;
    }
};

/// Stub compute WebSocket accept key.
pub fn computeWebSocketAcceptKey(allocator: std.mem.Allocator, client_key: []const u8) ![]u8 {
    _ = allocator;
    _ = client_key;
    return error.StreamingDisabled;
}

/// Stub backend type.
pub const BackendType = enum {
    local,
    openai,
    ollama,
    anthropic,

    pub fn fromString(s: []const u8) ?BackendType {
        if (std.mem.eql(u8, s, "local")) return .local;
        if (std.mem.eql(u8, s, "openai")) return .openai;
        if (std.mem.eql(u8, s, "ollama")) return .ollama;
        if (std.mem.eql(u8, s, "anthropic")) return .anthropic;
        return null;
    }
};

/// Stub backend generation config.
pub const BackendGenerationConfig = struct {
    max_tokens: u32 = 1024,
    temperature: f32 = 0.7,
    top_p: f32 = 0.9,
    top_k: u32 = 0,
    model: ?[]const u8 = null,
};

/// Stub backend.
pub const Backend = struct {
    allocator: std.mem.Allocator,
    backend_type: BackendType,

    pub fn init(allocator: std.mem.Allocator, backend_type: BackendType) !Backend {
        return .{ .allocator = allocator, .backend_type = backend_type };
    }

    pub fn deinit(self: *Backend) void {
        self.* = undefined;
    }

    pub fn generate(self: *Backend, prompt: []const u8, config: BackendGenerationConfig) ![]u8 {
        _ = self;
        _ = prompt;
        _ = config;
        return error.StreamingDisabled;
    }
};

/// Stub backend router.
pub const BackendRouter = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !BackendRouter {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *BackendRouter) void {
        self.* = undefined;
    }

    pub fn getBackend(self: *BackendRouter, backend_type: BackendType) !*Backend {
        _ = self;
        _ = backend_type;
        return error.StreamingDisabled;
    }

    pub fn listModelsJson(self: *BackendRouter, allocator: std.mem.Allocator) ![]u8 {
        _ = self;
        _ = allocator;
        return error.StreamingDisabled;
    }
};

/// Stub OpenAI formats module.
pub const formats_openai = struct {
    pub const Role = enum { system, user, assistant, tool };
    pub const ChatMessage = struct { role: Role, content: []const u8 };

    pub const ChatCompletionRequest = struct {
        model: []const u8,
        messages: []ChatMessage,
        max_tokens: u32 = 1024,
        temperature: f32 = 0.7,
        stream: bool = false,

        pub fn deinit(self: *const ChatCompletionRequest, allocator: std.mem.Allocator) void {
            _ = self;
            _ = allocator;
        }
    };

    pub fn parseRequest(allocator: std.mem.Allocator, json: []const u8) !ChatCompletionRequest {
        _ = allocator;
        _ = json;
        return error.StreamingDisabled;
    }

    pub fn formatStreamChunk(
        allocator: std.mem.Allocator,
        content: []const u8,
        model: []const u8,
        index: u32,
        is_end: bool,
    ) ![]u8 {
        _ = allocator;
        _ = content;
        _ = model;
        _ = index;
        _ = is_end;
        return error.StreamingDisabled;
    }

    pub fn formatResponse(allocator: std.mem.Allocator, content: []const u8, model: []const u8) ![]u8 {
        _ = allocator;
        _ = content;
        _ = model;
        return error.StreamingDisabled;
    }
};

// ============================================================================
// Error Recovery and Resilience Stubs
// ============================================================================

/// Stub recovery event.
pub const RecoveryEvent = enum {
    backend_failure,
    backend_recovery,
    circuit_opened,
    circuit_closed,
    retry_attempt,
    retry_exhausted,
    session_restored,
};

/// Stub recovery callback.
pub const RecoveryCallback = *const fn (event: RecoveryEvent, backend: BackendType, context: ?*anyopaque) void;

/// Stub recovery config.
pub const RecoveryConfig = struct {
    /// Enable recovery features.
    enabled: bool = true,
    /// Retry configuration.
    retry: StreamingRetryConfig = .{},
    /// Circuit breaker configuration (applied per-backend).
    circuit_breaker: CircuitBreakerConfig = .{},
    /// Session cache configuration.
    session_cache: SessionCacheConfig = .{},
    /// Metrics configuration.
    metrics: StreamingMetricsConfig = .{},

    /// Use local backend defaults (faster timeouts).
    pub fn forLocalBackend() RecoveryConfig {
        return .{
            .retry = StreamingRetryConfig.forLocalBackend(),
            .circuit_breaker = .{
                .failure_threshold = 3,
                .timeout_ms = 10_000,
            },
        };
    }

    /// Use external backend defaults (more tolerant).
    pub fn forExternalBackend() RecoveryConfig {
        return .{
            .retry = StreamingRetryConfig.forExternalBackend(),
            .circuit_breaker = .{
                .failure_threshold = 5,
                .timeout_ms = 120_000,
            },
        };
    }
};

/// Stub stream recovery.
pub const StreamRecovery = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: RecoveryConfig) !StreamRecovery {
        _ = config;
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *StreamRecovery) void {
        self.* = undefined;
    }

    pub fn isBackendAvailable(self: *StreamRecovery, backend: BackendType) bool {
        _ = self;
        _ = backend;
        return false;
    }

    pub fn recordSuccess(self: *StreamRecovery, backend: BackendType) void {
        _ = self;
        _ = backend;
    }

    pub fn recordFailure(self: *StreamRecovery, backend: BackendType) void {
        _ = self;
        _ = backend;
    }

    pub fn getCircuitState(self: *StreamRecovery, backend: BackendType) CircuitState {
        _ = self;
        _ = backend;
        return .open;
    }
};

/// Stub circuit state.
pub const CircuitState = enum {
    closed,
    open,
    half_open,
};

/// Stub circuit breaker config.
pub const CircuitBreakerConfig = struct {
    failure_threshold: u32 = 5,
    success_threshold: u32 = 2,
    timeout_ms: u64 = 60_000,
    half_open_max_requests: u32 = 3,
};

/// Stub circuit breaker.
pub const CircuitBreaker = struct {
    state: CircuitState = .open,

    pub fn init(config: CircuitBreakerConfig) CircuitBreaker {
        _ = config;
        return .{};
    }

    pub fn deinit(self: *CircuitBreaker) void {
        self.* = undefined;
    }

    pub fn allowRequest(self: *CircuitBreaker) bool {
        _ = self;
        return false;
    }

    pub fn recordSuccess(self: *CircuitBreaker) void {
        _ = self;
    }

    pub fn recordFailure(self: *CircuitBreaker) void {
        _ = self;
    }

    pub fn getState(self: *const CircuitBreaker) CircuitState {
        return self.state;
    }

    pub fn reset(self: *CircuitBreaker) void {
        _ = self;
    }
};

/// Stub streaming retryable errors.
pub const StreamingRetryableErrors = struct {
    connection_reset: bool = true,
    timeout: bool = true,
    server_error: bool = true,
    rate_limited: bool = true,
};

/// Stub base retry config.
pub const BaseRetryConfig = struct {
    max_retries: u32 = 3,
    initial_delay_ns: u64 = 100_000_000,
    max_delay_ns: u64 = 5_000_000_000,
    multiplier: f64 = 2.0,
    jitter: bool = true,
    jitter_factor: f64 = 0.25,
    total_timeout_ns: u64 = 0,
};

/// Stub streaming retry config.
pub const StreamingRetryConfig = struct {
    /// Base retry config for connection/backend operations.
    base: BaseRetryConfig = .{},
    /// Enable retry for streaming operations.
    enabled: bool = true,
    /// Timeout for single token generation (milliseconds).
    token_timeout_ms: u64 = 30_000,
    /// Timeout for entire stream (milliseconds, 0 = unlimited).
    total_timeout_ms: u64 = 300_000,
    /// Backend-specific timeout for initial connection (milliseconds).
    backend_timeout_ms: u64 = 60_000,
    /// WebSocket ping/pong timeout (milliseconds).
    websocket_timeout_ms: u64 = 30_000,

    /// Create a config optimized for local backends.
    pub fn forLocalBackend() StreamingRetryConfig {
        return .{
            .base = .{
                .max_retries = 2,
                .initial_delay_ns = 50_000_000,
                .max_delay_ns = 1_000_000_000,
                .multiplier = 2.0,
                .jitter = true,
                .jitter_factor = 0.1,
                .total_timeout_ns = 0,
            },
            .token_timeout_ms = 10_000,
            .backend_timeout_ms = 10_000,
        };
    }

    /// Create a config optimized for external API backends.
    pub fn forExternalBackend() StreamingRetryConfig {
        return .{
            .base = .{
                .max_retries = 3,
                .initial_delay_ns = 200_000_000,
                .max_delay_ns = 10_000_000_000,
                .multiplier = 2.0,
                .jitter = true,
                .jitter_factor = 0.25,
                .total_timeout_ns = 0,
            },
            .token_timeout_ms = 60_000,
            .backend_timeout_ms = 120_000,
        };
    }

    /// Convert token timeout to nanoseconds.
    pub fn tokenTimeoutNs(self: StreamingRetryConfig) u64 {
        return self.token_timeout_ms * std.time.ns_per_ms;
    }

    /// Convert total timeout to nanoseconds.
    pub fn totalTimeoutNs(self: StreamingRetryConfig) u64 {
        return self.total_timeout_ms * std.time.ns_per_ms;
    }

    /// Convert backend timeout to nanoseconds.
    pub fn backendTimeoutNs(self: StreamingRetryConfig) u64 {
        return self.backend_timeout_ms * std.time.ns_per_ms;
    }
};

/// Stub cached token.
pub const CachedToken = struct {
    /// SSE event ID for this token.
    event_id: u64,
    /// Token text (owned by cache).
    text: []const u8,
    /// Timestamp when cached (milliseconds).
    timestamp_ms: i64,
};

/// Stub session cache config.
pub const SessionCacheConfig = struct {
    /// Maximum number of sessions to cache.
    max_sessions: usize = 100,
    /// Maximum tokens to cache per session.
    max_tokens_per_session: usize = 100,
    /// Time-to-live for cached sessions (milliseconds).
    ttl_ms: u64 = 300_000,
    /// Cleanup interval for expired sessions (milliseconds).
    cleanup_interval_ms: u64 = 60_000,
};

/// Stub session cache.
pub const SessionCache = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: SessionCacheConfig) SessionCache {
        _ = config;
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *SessionCache) void {
        self.* = undefined;
    }

    pub fn storeToken(
        self: *SessionCache,
        session_id: []const u8,
        event_id: u64,
        token: []const u8,
        backend: BackendType,
        prompt_hash: u64,
    ) !void {
        _ = self;
        _ = session_id;
        _ = event_id;
        _ = token;
        _ = backend;
        _ = prompt_hash;
        return error.StreamingDisabled;
    }

    pub fn getTokensAfter(
        self: *SessionCache,
        session_id: []const u8,
        after_event_id: u64,
    ) ![]CachedToken {
        _ = self;
        _ = session_id;
        _ = after_event_id;
        return error.StreamingDisabled;
    }

    pub fn invalidateSession(self: *SessionCache, session_id: []const u8) void {
        _ = self;
        _ = session_id;
    }

    pub fn cleanup(self: *SessionCache) void {
        _ = self;
    }
};

/// Stub streaming metrics config.
pub const StreamingMetricsConfig = struct {
    /// Enable per-backend metrics.
    enable_backend_metrics: bool = true,
    /// Enable session cache metrics.
    enable_cache_metrics: bool = true,
    /// Enable recovery metrics.
    enable_recovery_metrics: bool = true,
};

/// Stub streaming metrics.
pub const StreamingMetrics = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: StreamingMetricsConfig) StreamingMetrics {
        _ = config;
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *StreamingMetrics) void {
        self.* = undefined;
    }

    pub fn recordLatency(self: *StreamingMetrics, backend: BackendType, latency_ms: f64) void {
        _ = self;
        _ = backend;
        _ = latency_ms;
    }

    pub fn recordTokens(self: *StreamingMetrics, backend: BackendType, count: usize) void {
        _ = self;
        _ = backend;
        _ = count;
    }

    pub fn recordError(self: *StreamingMetrics, backend: BackendType, error_type: []const u8) void {
        _ = self;
        _ = backend;
        _ = error_type;
    }

    pub fn getStats(self: *const StreamingMetrics, backend: BackendType) BackendMetricStats {
        _ = self;
        _ = backend;
        return .{};
    }

    pub fn reset(self: *StreamingMetrics) void {
        _ = self;
    }
};

/// Stub backend metric stats (used by StreamingMetrics).
pub const BackendMetricStats = struct {
    total_requests: u64 = 0,
    total_tokens: u64 = 0,
    total_errors: u64 = 0,
    avg_latency_ms: f64 = 0,
    p50_latency_ms: f64 = 0,
    p99_latency_ms: f64 = 0,
    tokens_per_second: f64 = 0,
};
