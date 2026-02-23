//! Stub implementation for enhanced streaming when AI features are disabled.

const std = @import("std");
const transformer = @import("../transformer/mod.zig");
const stub_root = @This();

// ── Sub-module re-exports (API parity with mod.zig) ────────────────────────

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

// ── Error types ────────────────────────────────────────────────────────────

pub const StreamingError = error{ FeatureDisabled, StreamClosed, InvalidState, GenerationFailed };

pub const StreamingServerError = std.mem.Allocator.Error || error{
    FeatureDisabled,
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

// ── Enums ──────────────────────────────────────────────────────────────────

pub const BackpressureStrategy = enum { drop, block, buffer, sample, adaptive };
pub const FlowState = enum { normal, throttled, blocked, recovering };
pub const BufferStrategy = enum { fifo, lifo, priority, ring };
pub const StreamEventType = enum { token, start, end, error_event, metadata, heartbeat };
pub const StreamState = enum { idle, generating, paused, completed, failed };
pub const CircuitState = enum { closed, open, half_open };
pub const RecoveryEvent = enum { backend_failure, backend_recovery, circuit_opened, circuit_closed, retry_attempt, retry_exhausted, session_restored };
pub const WebSocketOpcode = enum(u4) { continuation = 0x0, text = 0x1, binary = 0x2, close = 0x8, ping = 0x9, pong = 0xA };
pub const WebSocketCloseCode = enum(u16) { normal = 1000, going_away = 1001, protocol_error = 1002, _ };

pub const BackendType = enum {
    local,
    openai,
    ollama,
    anthropic,
    pub fn fromString(s: []const u8) ?BackendType {
        const map = .{ .{ "local", .local }, .{ "openai", .openai }, .{ "ollama", .ollama }, .{ "anthropic", .anthropic } };
        inline for (map) |entry| {
            if (std.mem.eql(u8, s, entry[0])) return entry[1];
        }
        return null;
    }
};

// ── Config structs ─────────────────────────────────────────────────────────

pub const SseConfig = struct {
    event_prefix: []const u8 = "",
    include_timestamp: bool = false,
    include_id: bool = true,
    retry_ms: ?u32 = null,
};

pub const BackpressureConfig = struct {
    strategy: BackpressureStrategy = .buffer,
    high_watermark: usize = 100,
    low_watermark: usize = 25,
    max_buffer: usize = 1000,
};

pub const BufferConfig = struct {
    strategy: BufferStrategy = .fifo,
    capacity: usize = 100,
    flush_threshold: usize = 50,
    coalesce: bool = false,
};

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

pub const WebSocketConfig = struct {
    max_message_size: usize = 16 * 1024 * 1024,
    ping_interval_ms: u64 = 30000,
    enable_compression: bool = false,
};

pub const BackendGenerationConfig = struct {
    max_tokens: u32 = 1024,
    temperature: f32 = 0.7,
    top_p: f32 = 0.9,
    top_k: u32 = 0,
    model: ?[]const u8 = null,
};

pub const CircuitBreakerConfig = struct {
    failure_threshold: u32 = 5,
    success_threshold: u32 = 2,
    timeout_ms: u64 = 60_000,
    half_open_max_requests: u32 = 3,
};

pub const StreamingMetricsConfig = struct {
    enable_backend_metrics: bool = true,
    enable_cache_metrics: bool = true,
    enable_recovery_metrics: bool = true,
};

pub const SessionCacheConfig = struct {
    max_sessions: usize = 100,
    max_tokens_per_session: usize = 100,
    ttl_ms: u64 = 300_000,
    cleanup_interval_ms: u64 = 60_000,
};

pub const StreamingRetryableErrors = struct {
    connection_reset: bool = true,
    timeout: bool = true,
    server_error: bool = true,
    rate_limited: bool = true,
};

pub const BaseRetryConfig = struct {
    max_retries: u32 = 3,
    initial_delay_ns: u64 = 100_000_000,
    max_delay_ns: u64 = 5_000_000_000,
    multiplier: f64 = 2.0,
    jitter: bool = true,
    jitter_factor: f64 = 0.25,
    total_timeout_ns: u64 = 0,
};

pub const StreamingRetryConfig = struct {
    base: BaseRetryConfig = .{},
    enabled: bool = true,
    token_timeout_ms: u64 = 30_000,
    total_timeout_ms: u64 = 300_000,
    backend_timeout_ms: u64 = 60_000,
    websocket_timeout_ms: u64 = 30_000,

    pub fn forLocalBackend() StreamingRetryConfig {
        return .{ .base = .{ .max_retries = 2, .initial_delay_ns = 50_000_000, .max_delay_ns = 1_000_000_000, .multiplier = 2.0, .jitter = true, .jitter_factor = 0.1 }, .token_timeout_ms = 10_000, .backend_timeout_ms = 10_000 };
    }
    pub fn forExternalBackend() StreamingRetryConfig {
        return .{ .base = .{ .max_retries = 3, .initial_delay_ns = 200_000_000, .max_delay_ns = 10_000_000_000, .multiplier = 2.0, .jitter = true, .jitter_factor = 0.25 }, .token_timeout_ms = 60_000, .backend_timeout_ms = 120_000 };
    }
    pub fn tokenTimeoutNs(self: StreamingRetryConfig) u64 {
        return self.token_timeout_ms * std.time.ns_per_ms;
    }
    pub fn totalTimeoutNs(self: StreamingRetryConfig) u64 {
        return self.total_timeout_ms * std.time.ns_per_ms;
    }
    pub fn backendTimeoutNs(self: StreamingRetryConfig) u64 {
        return self.backend_timeout_ms * std.time.ns_per_ms;
    }
};

pub const ServerConfig = struct {
    address: []const u8 = "127.0.0.1:8080",
    auth_token: ?[]const u8 = null,
    allow_health_without_auth: bool = true,
    default_backend: BackendType = .local,
    heartbeat_interval_ms: u64 = 15000,
    max_concurrent_streams: u32 = 100,
    enable_openai_compat: bool = true,
    enable_websocket: bool = true,
    default_model_path: ?[]const u8 = null,
    preload_model: bool = false,
    enable_recovery: bool = true,
    recovery_config: RecoveryConfig = .{},
};

pub const StreamConfig = struct {
    sse_config: SseConfig = .{},
    backpressure_config: BackpressureConfig = .{},
    buffer_config: BufferConfig = .{},
};

pub const RecoveryConfig = struct {
    enabled: bool = true,
    retry: StreamingRetryConfig = .{},
    circuit_breaker: CircuitBreakerConfig = .{},
    session_cache: SessionCacheConfig = .{},
    metrics: StreamingMetricsConfig = .{},

    pub fn forLocalBackend() RecoveryConfig {
        return .{ .retry = StreamingRetryConfig.forLocalBackend(), .circuit_breaker = .{ .failure_threshold = 3, .timeout_ms = 10_000 } };
    }
    pub fn forExternalBackend() RecoveryConfig {
        return .{ .retry = StreamingRetryConfig.forExternalBackend(), .circuit_breaker = .{ .failure_threshold = 5, .timeout_ms = 120_000 } };
    }
};

pub const RecoveryCallback = *const fn (event: RecoveryEvent, backend: BackendType, context: ?*anyopaque) void;

// ── Stats structs ──────────────────────────────────────────────────────────

pub const BackpressureStats = struct { pending_count: usize = 0, dropped_count: usize = 0, total_processed: u64 = 0, current_tps: f64 = 0, state: FlowState = .normal, utilization: f64 = 0 };
pub const BufferStats = struct { current_size: usize = 0, capacity: usize = 0, total_pushed: u64 = 0, total_popped: u64 = 0, total_dropped: u64 = 0, utilization: f64 = 0 };
pub const StreamStats = struct {
    total_tokens: usize = 0,
    total_chars: usize = 0,
    tokens_per_second: f64 = 0,
    start_time_ns: u64 = 0,
    end_time_ns: u64 = 0,
    pause_count: usize = 0,
    pause_duration_ns: u64 = 0,
    pub fn duration_ms(self: *const StreamStats) f64 {
        return @as(f64, @floatFromInt(self.end_time_ns - self.start_time_ns - self.pause_duration_ns)) / 1_000_000.0;
    }
};
pub const BackendMetricStats = struct { total_requests: u64 = 0, total_tokens: u64 = 0, total_errors: u64 = 0, avg_latency_ms: f64 = 0, p50_latency_ms: f64 = 0, p99_latency_ms: f64 = 0, tokens_per_second: f64 = 0 };

// ── Data structs ───────────────────────────────────────────────────────────

pub const SseEvent = struct {
    event: ?[]const u8 = null,
    data: []const u8 = "",
    id: ?[]const u8 = null,
    retry: ?u32 = null,
    pub fn deinit(self: *SseEvent, _: std.mem.Allocator) void {
        self.* = undefined;
    }
};

pub const StreamToken = struct {
    id: u32 = 0,
    text: []const u8 = "",
    log_prob: ?f32 = null,
    is_end: bool = false,
    timestamp_ns: i128 = 0,
    sequence_index: usize = 0,
    pub fn clone(self: StreamToken, allocator: std.mem.Allocator) !StreamToken {
        return .{ .id = self.id, .text = try allocator.dupe(u8, self.text), .log_prob = self.log_prob, .is_end = self.is_end, .timestamp_ns = self.timestamp_ns, .sequence_index = self.sequence_index };
    }
    pub fn deinit(self: *StreamToken, allocator: std.mem.Allocator) void {
        allocator.free(self.text);
        self.* = undefined;
    }
};

const GeneratorStreamToken = struct { id: u32, text: []const u8, log_prob: ?f32 = null, is_end: bool = false };

pub const StreamEvent = struct {
    event_type: StreamEventType = .token,
    token: ?StreamToken = null,
    metadata: ?[]const u8 = null,
    error_message: ?[]const u8 = null,
    timestamp_ns: i128 = 0,
    pub fn tokenEvent(token: StreamToken) StreamEvent {
        return .{ .event_type = .token, .token = token };
    }
    pub fn startEvent() StreamEvent {
        return .{ .event_type = .start };
    }
    pub fn endEvent() StreamEvent {
        return .{ .event_type = .end };
    }
    pub fn errorEvent(message: []const u8) StreamEvent {
        return .{ .event_type = .error_event, .error_message = message };
    }
    pub fn heartbeatEvent() StreamEvent {
        return .{ .event_type = .heartbeat };
    }
};

pub const CachedToken = struct { event_id: u64, text: []const u8, timestamp_ms: i64 };

// ── Stub type impls (all methods return error.FeatureDisabled or no-op) ──

pub const SseEncoder = struct {
    allocator: std.mem.Allocator,
    pub fn init(allocator: std.mem.Allocator, _: SseConfig) SseEncoder {
        return .{ .allocator = allocator };
    }
    pub fn deinit(self: *SseEncoder) void {
        self.* = undefined;
    }
    pub fn encode(_: *SseEncoder, _: StreamEvent) ![]u8 {
        return error.FeatureDisabled;
    }
};

pub const SseDecoder = struct {
    allocator: std.mem.Allocator,
    pub fn init(allocator: std.mem.Allocator) SseDecoder {
        return .{ .allocator = allocator };
    }
    pub fn deinit(self: *SseDecoder) void {
        self.* = undefined;
    }
    pub fn feed(_: *SseDecoder, _: []const u8) ![]SseEvent {
        return error.FeatureDisabled;
    }
};

pub const BackpressureController = struct {
    pub fn init(_: BackpressureConfig) BackpressureController {
        return .{};
    }
    pub fn checkFlow(_: *BackpressureController) FlowState {
        return .normal;
    }
    pub fn produce(_: *BackpressureController) void {}
    pub fn consume(_: *BackpressureController) void {}
};

pub const RateLimiter = struct {
    tokens_per_second: f64 = 0,
    bucket_size: ?f64 = null,
    pub fn init(tps: f64, bs: ?f64) RateLimiter {
        return .{ .tokens_per_second = tps, .bucket_size = bs };
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

pub const TokenBuffer = struct {
    allocator: std.mem.Allocator,
    pub fn init(allocator: std.mem.Allocator, _: BufferConfig) TokenBuffer {
        return .{ .allocator = allocator };
    }
    pub fn deinit(self: *TokenBuffer) void {
        self.* = undefined;
    }
    pub fn push(_: *TokenBuffer, _: StreamToken) !void {
        return error.FeatureDisabled;
    }
    pub fn pop(_: *TokenBuffer) ?StreamToken {
        return null;
    }
    pub fn peek(_: *const TokenBuffer) ?StreamToken {
        return null;
    }
    pub fn len(_: *const TokenBuffer) usize {
        return 0;
    }
    pub fn isEmpty(_: *const TokenBuffer) bool {
        return true;
    }
    pub fn isFull(_: *const TokenBuffer) bool {
        return false;
    }
    pub fn shouldFlush(_: *const TokenBuffer) bool {
        return false;
    }
    pub fn clear(_: *TokenBuffer) void {}
    pub fn getStats(_: *const TokenBuffer) BufferStats {
        return .{};
    }
    pub fn flushAsText(_: *TokenBuffer) ![]u8 {
        return error.FeatureDisabled;
    }
};

pub const CoalescingBuffer = struct {
    allocator: std.mem.Allocator,
    max_length: usize,
    pub fn init(allocator: std.mem.Allocator, max_length: usize) CoalescingBuffer {
        return .{ .allocator = allocator, .max_length = max_length };
    }
    pub fn deinit(_: *CoalescingBuffer) void {}
    pub fn add(_: *CoalescingBuffer, _: []const u8) !?[]u8 {
        return error.FeatureDisabled;
    }
    pub fn flush(_: *CoalescingBuffer) ![]u8 {
        return error.FeatureDisabled;
    }
    pub fn len(_: *const CoalescingBuffer) usize {
        return 0;
    }
    pub fn isEmpty(_: *const CoalescingBuffer) bool {
        return true;
    }
};

pub const StreamingGenerator = struct {
    allocator: std.mem.Allocator,
    config: GenerationConfig,
    state: StreamState,
    pub fn init(allocator: std.mem.Allocator, _: *transformer.TransformerModel, config: GenerationConfig) StreamingGenerator {
        return .{ .allocator = allocator, .config = config, .state = .idle };
    }
    pub fn deinit(_: *StreamingGenerator) void {}
    pub fn start(_: *StreamingGenerator, _: []const u8) StreamingError!void {
        return error.FeatureDisabled;
    }
    pub fn next(_: *StreamingGenerator) StreamingError!?GeneratorStreamToken {
        return error.FeatureDisabled;
    }
    pub fn pause(_: *StreamingGenerator) void {}
    pub fn resumeGeneration(_: *StreamingGenerator) void {}
    pub fn cancel(_: *StreamingGenerator) void {}
    pub fn reset(self: *StreamingGenerator, new_config: GenerationConfig) void {
        self.config = new_config;
        self.state = .idle;
    }
    pub fn getGeneratedText(_: *StreamingGenerator, _: std.mem.Allocator) ![]u8 {
        return error.FeatureDisabled;
    }
    pub fn tokenCount(_: *const StreamingGenerator) usize {
        return 0;
    }
    pub fn isComplete(self: *const StreamingGenerator) bool {
        return self.state == .completed;
    }
};

pub const EnhancedStreamingGenerator = struct {
    allocator: std.mem.Allocator,
    pub fn init(allocator: std.mem.Allocator, _: StreamConfig) EnhancedStreamingGenerator {
        return .{ .allocator = allocator };
    }
    pub fn deinit(self: *EnhancedStreamingGenerator) void {
        self.* = undefined;
    }
    pub fn start(_: *EnhancedStreamingGenerator) !void {
        return error.FeatureDisabled;
    }
    pub fn emit(_: *EnhancedStreamingGenerator, _: StreamToken) !?[]u8 {
        return error.FeatureDisabled;
    }
    pub fn flush(_: *EnhancedStreamingGenerator) ![][]u8 {
        return error.FeatureDisabled;
    }
    pub fn complete(_: *EnhancedStreamingGenerator) ![]u8 {
        return error.FeatureDisabled;
    }
    pub fn getStats(_: *const EnhancedStreamingGenerator) StreamStats {
        return .{};
    }
};

pub const StreamingServer = struct {
    allocator: std.mem.Allocator,
    pub fn init(allocator: std.mem.Allocator, _: ServerConfig) !StreamingServer {
        return .{ .allocator = allocator };
    }
    pub fn deinit(self: *StreamingServer) void {
        self.* = undefined;
    }
    pub fn serve(_: *StreamingServer) !void {
        return error.FeatureDisabled;
    }
};

pub const WebSocketHandler = struct {
    allocator: std.mem.Allocator,
    pub fn init(allocator: std.mem.Allocator, _: WebSocketConfig) !WebSocketHandler {
        return .{ .allocator = allocator };
    }
    pub fn deinit(self: *WebSocketHandler) void {
        self.* = undefined;
    }
    pub fn sendText(_: *WebSocketHandler, _: []const u8) ![]u8 {
        return error.FeatureDisabled;
    }
};

pub const Backend = struct {
    allocator: std.mem.Allocator,
    backend_type: BackendType,
    pub fn init(allocator: std.mem.Allocator, bt: BackendType) !Backend {
        return .{ .allocator = allocator, .backend_type = bt };
    }
    pub fn deinit(self: *Backend) void {
        self.* = undefined;
    }
    pub fn generate(_: *Backend, _: []const u8, _: BackendGenerationConfig) ![]u8 {
        return error.FeatureDisabled;
    }
};

pub const BackendRouter = struct {
    allocator: std.mem.Allocator,
    pub fn init(allocator: std.mem.Allocator) !BackendRouter {
        return .{ .allocator = allocator };
    }
    pub fn deinit(self: *BackendRouter) void {
        self.* = undefined;
    }
    pub fn getBackend(_: *BackendRouter, _: BackendType) !*Backend {
        return error.FeatureDisabled;
    }
    pub fn listModelsJson(_: *BackendRouter, _: std.mem.Allocator) ![]u8 {
        return error.FeatureDisabled;
    }
};

pub const CircuitBreaker = struct {
    state: CircuitState = .open,
    pub fn init(_: CircuitBreakerConfig) CircuitBreaker {
        return .{};
    }
    pub fn deinit(self: *CircuitBreaker) void {
        self.* = undefined;
    }
    pub fn allowRequest(_: *CircuitBreaker) bool {
        return false;
    }
    pub fn recordSuccess(_: *CircuitBreaker) void {}
    pub fn recordFailure(_: *CircuitBreaker) void {}
    pub fn getState(self: *const CircuitBreaker) CircuitState {
        return self.state;
    }
    pub fn reset(_: *CircuitBreaker) void {}
};

pub const StreamRecovery = struct {
    allocator: std.mem.Allocator,
    pub fn init(allocator: std.mem.Allocator, _: RecoveryConfig) !StreamRecovery {
        return .{ .allocator = allocator };
    }
    pub fn deinit(self: *StreamRecovery) void {
        self.* = undefined;
    }
    pub fn isBackendAvailable(_: *StreamRecovery, _: BackendType) bool {
        return false;
    }
    pub fn recordSuccess(_: *StreamRecovery, _: BackendType) void {}
    pub fn recordFailure(_: *StreamRecovery, _: BackendType) void {}
    pub fn getCircuitState(_: *StreamRecovery, _: BackendType) CircuitState {
        return .open;
    }
};

pub const SessionCache = struct {
    allocator: std.mem.Allocator,
    pub fn init(allocator: std.mem.Allocator, _: SessionCacheConfig) SessionCache {
        return .{ .allocator = allocator };
    }
    pub fn deinit(self: *SessionCache) void {
        self.* = undefined;
    }
    pub fn storeToken(_: *SessionCache, _: []const u8, _: u64, _: []const u8, _: BackendType, _: u64) !void {
        return error.FeatureDisabled;
    }
    pub fn getTokensAfter(_: *SessionCache, _: []const u8, _: u64) ![]CachedToken {
        return error.FeatureDisabled;
    }
    pub fn invalidateSession(_: *SessionCache, _: []const u8) void {}
    pub fn cleanup(_: *SessionCache) void {}
};

pub const StreamingMetrics = struct {
    allocator: std.mem.Allocator,
    pub fn init(allocator: std.mem.Allocator, _: StreamingMetricsConfig) StreamingMetrics {
        return .{ .allocator = allocator };
    }
    pub fn deinit(self: *StreamingMetrics) void {
        self.* = undefined;
    }
    pub fn recordLatency(_: *StreamingMetrics, _: BackendType, _: f64) void {}
    pub fn recordTokens(_: *StreamingMetrics, _: BackendType, _: usize) void {}
    pub fn recordError(_: *StreamingMetrics, _: BackendType, _: []const u8) void {}
    pub fn getStats(_: *const StreamingMetrics, _: BackendType) BackendMetricStats {
        return .{};
    }
    pub fn reset(_: *StreamingMetrics) void {}
};

// ── Free functions ─────────────────────────────────────────────────────────

pub fn streamInference(_: std.mem.Allocator, _: *transformer.TransformerModel, _: []const u8, _: GenerationConfig, _: anytype) !void {
    return error.FeatureDisabled;
}
pub fn formatStreamOutput(_: []const GeneratorStreamToken, _: std.mem.Allocator) ![]u8 {
    return error.FeatureDisabled;
}
pub fn createChunkedStream(_: std.mem.Allocator, _: []const GeneratorStreamToken, _: usize) ![]const []const u8 {
    return error.FeatureDisabled;
}
pub fn createSseStream(_: std.mem.Allocator, _: []const []const u8) ![]u8 {
    return error.FeatureDisabled;
}
pub fn computeWebSocketAcceptKey(_: std.mem.Allocator, _: []const u8) ![]u8 {
    return error.FeatureDisabled;
}

// ── OpenAI format stubs ────────────────────────────────────────────────────

pub const formats_openai = struct {
    pub const Role = enum { system, user, assistant, tool };
    pub const ChatMessage = struct { role: Role, content: []const u8 };
    pub const ChatCompletionRequest = struct {
        model: []const u8,
        messages: []ChatMessage,
        max_tokens: u32 = 1024,
        temperature: f32 = 0.7,
        stream: bool = false,
        pub fn deinit(_: *const ChatCompletionRequest, _: std.mem.Allocator) void {}
    };
    pub fn parseRequest(_: std.mem.Allocator, _: []const u8) !ChatCompletionRequest {
        return error.FeatureDisabled;
    }
    pub fn formatStreamChunk(_: std.mem.Allocator, _: []const u8, _: []const u8, _: u32, _: bool) ![]u8 {
        return error.FeatureDisabled;
    }
    pub fn formatResponse(_: std.mem.Allocator, _: []const u8, _: []const u8) ![]u8 {
        return error.FeatureDisabled;
    }
};
