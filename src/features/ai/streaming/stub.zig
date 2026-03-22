//! Stub implementation for enhanced streaming when AI features are disabled.

const std = @import("std");
const transformer = @import("../transformer/mod.zig");
const types = @import("types.zig");

// ── Re-exported types ──────────────────────────────────────────────────────

pub const StreamingError = types.StreamingError;
pub const StreamingServerError = types.StreamingServerError;
pub const BackpressureStrategy = types.BackpressureStrategy;
pub const FlowState = types.FlowState;
pub const BufferStrategy = types.BufferStrategy;
pub const StreamEventType = types.StreamEventType;
pub const StreamState = types.StreamState;
pub const CircuitState = types.CircuitState;
pub const RecoveryEvent = types.RecoveryEvent;
pub const WebSocketOpcode = types.WebSocketOpcode;
pub const WebSocketCloseCode = types.WebSocketCloseCode;
pub const BackendType = types.BackendType;
pub const SseConfig = types.SseConfig;
pub const BackpressureConfig = types.BackpressureConfig;
pub const BufferConfig = types.BufferConfig;
pub const GenerationConfig = types.GenerationConfig;
pub const WebSocketConfig = types.WebSocketConfig;
pub const BackendGenerationConfig = types.BackendGenerationConfig;
pub const CircuitBreakerConfig = types.CircuitBreakerConfig;
pub const StreamingMetricsConfig = types.StreamingMetricsConfig;
pub const SessionCacheConfig = types.SessionCacheConfig;
pub const StreamingRetryableErrors = types.StreamingRetryableErrors;
pub const BaseRetryConfig = types.BaseRetryConfig;
pub const StreamingRetryConfig = types.StreamingRetryConfig;
pub const ServerConfig = types.ServerConfig;
pub const StreamConfig = types.StreamConfig;
pub const RecoveryConfig = types.RecoveryConfig;
pub const RecoveryCallback = types.RecoveryCallback;
pub const BackpressureStats = types.BackpressureStats;
pub const BufferStats = types.BufferStats;
pub const StreamStats = types.StreamStats;
pub const BackendMetricStats = types.BackendMetricStats;
pub const SseEvent = types.SseEvent;
pub const StreamToken = types.StreamToken;
pub const StreamEvent = types.StreamEvent;
pub const CachedToken = types.CachedToken;
pub const SseEncoder = types.SseEncoder;
pub const SseDecoder = types.SseDecoder;
pub const BackpressureController = types.BackpressureController;
pub const RateLimiter = types.RateLimiter;
pub const TokenBuffer = types.TokenBuffer;
pub const CoalescingBuffer = types.CoalescingBuffer;
pub const StreamingGenerator = types.StreamingGenerator;
pub const EnhancedStreamingGenerator = types.EnhancedStreamingGenerator;
pub const StreamingServer = types.StreamingServer;
pub const WebSocketHandler = types.WebSocketHandler;
pub const Backend = types.Backend;
pub const BackendRouter = types.BackendRouter;
pub const CircuitBreaker = types.CircuitBreaker;
pub const StreamRecovery = types.StreamRecovery;
pub const SessionCache = types.SessionCache;
pub const StreamingMetrics = types.StreamingMetrics;
pub const formats_openai = types.formats_openai;

// ── Sub-module re-exports (API parity with mod.zig) ────────────────────────

const stub_root = @This();

pub const sse = struct {
    pub const SseEvent = types.SseEvent;
    pub const SseEncoder = types.SseEncoder;
    pub const SseDecoder = types.SseDecoder;
    pub const SseConfig = types.SseConfig;
};

pub const backpressure = struct {
    pub const BackpressureController = types.BackpressureController;
    pub const BackpressureStrategy = types.BackpressureStrategy;
    pub const BackpressureConfig = types.BackpressureConfig;
    pub const FlowState = types.FlowState;
    pub const BackpressureStats = types.BackpressureStats;
    pub const RateLimiter = types.RateLimiter;
};

pub const buffer = struct {
    pub const TokenBuffer = types.TokenBuffer;
    pub const BufferConfig = types.BufferConfig;
    pub const BufferStrategy = types.BufferStrategy;
    pub const BufferStats = types.BufferStats;
    pub const CoalescingBuffer = types.CoalescingBuffer;
};

pub const generator = struct {
    pub const StreamingGenerator = types.StreamingGenerator;
    pub const StreamingError = types.StreamingError;
    pub const StreamState = types.StreamState;
    pub const GenerationConfig = types.GenerationConfig;
    pub const streamInference = stub_root.streamInference;
    pub const formatStreamOutput = stub_root.formatStreamOutput;
    pub const createChunkedStream = stub_root.createChunkedStream;
};

pub const server = struct {
    pub const StreamingServer = types.StreamingServer;
    pub const ServerConfig = types.ServerConfig;
    pub const StreamingServerError = types.StreamingServerError;
};

pub const websocket = struct {
    pub const WebSocketHandler = types.WebSocketHandler;
    pub const WebSocketConfig = types.WebSocketConfig;
    pub const Opcode = types.WebSocketOpcode;
    pub const CloseCode = types.WebSocketCloseCode;
    pub const computeAcceptKey = stub_root.computeWebSocketAcceptKey;
};

pub const backends = struct {
    pub const BackendType = types.BackendType;
    pub const BackendRouter = types.BackendRouter;
    pub const Backend = types.Backend;
    pub const GenerationConfig = types.BackendGenerationConfig;
};

pub const formats = struct {
    pub const openai = types.formats_openai;
};

pub const recovery = struct {
    pub const StreamRecovery = types.StreamRecovery;
    pub const RecoveryConfig = types.RecoveryConfig;
    pub const RecoveryEvent = types.RecoveryEvent;
    pub const RecoveryCallback = types.RecoveryCallback;
};

pub const circuit_breaker = struct {
    pub const CircuitBreaker = types.CircuitBreaker;
    pub const CircuitBreakerConfig = types.CircuitBreakerConfig;
    pub const CircuitState = types.CircuitState;
};

pub const retry_config = struct {
    pub const StreamingRetryConfig = types.StreamingRetryConfig;
    pub const StreamingRetryableErrors = types.StreamingRetryableErrors;
};

pub const session_cache = struct {
    pub const SessionCache = types.SessionCache;
    pub const SessionCacheConfig = types.SessionCacheConfig;
    pub const CachedToken = types.CachedToken;
};

pub const streaming_metrics = struct {
    pub const StreamingMetrics = types.StreamingMetrics;
    pub const StreamingMetricsConfig = types.StreamingMetricsConfig;
};

// ── Free functions ─────────────────────────────────────────────────────────

pub fn streamInference(_: std.mem.Allocator, _: *transformer.TransformerModel, _: []const u8, _: types.GenerationConfig, _: anytype) !void {
    return error.FeatureDisabled;
}
pub fn formatStreamOutput(_: []const types.GeneratorStreamToken, _: std.mem.Allocator) ![]u8 {
    return error.FeatureDisabled;
}
pub fn createChunkedStream(_: std.mem.Allocator, _: []const types.GeneratorStreamToken, _: usize) ![]const []const u8 {
    return error.FeatureDisabled;
}
pub fn createSseStream(_: std.mem.Allocator, _: []const []const u8) ![]u8 {
    return error.FeatureDisabled;
}
pub fn computeWebSocketAcceptKey(_: std.mem.Allocator, _: []const u8) ![]u8 {
    return error.FeatureDisabled;
}

test {
    std.testing.refAllDecls(@This());
}
