//! Enhanced streaming support for AI model responses.
//!
//! This module provides comprehensive streaming infrastructure for real-time
//! AI model output, including HTTP servers, protocol handlers, and resilience
//! features for production deployments.
//!
//! ## Features
//!
//! - **Basic Streaming**: Token-by-token generation from transformer models
//! - **SSE Protocol**: Server-Sent Events encoding/decoding for HTTP streaming
//! - **WebSocket Support**: Bidirectional streaming with cancellation
//! - **Backpressure Control**: Flow management to prevent memory exhaustion
//! - **Token Buffering**: Configurable buffering strategies (FIFO, ring, etc.)
//! - **HTTP Server**: OpenAI-compatible API with SSE and WebSocket endpoints
//! - **Backend Routing**: Support for local GGUF, OpenAI, Ollama, Anthropic
//! - **Circuit Breakers**: Per-backend failure isolation and recovery
//! - **Retry Logic**: Exponential backoff with jitter for transient failures
//! - **Session Caching**: Reconnection recovery via Last-Event-ID
//!
//! ## Error Handling
//!
//! The streaming module implements a comprehensive error handling strategy:
//!
//! | Error Type | Handling Strategy |
//! |------------|-------------------|
//! | Connection errors | Retry with exponential backoff |
//! | Backend failures | Circuit breaker isolation + fallback |
//! | Rate limiting | Backpressure + client notification |
//! | Stream interruption | Session cache for resumption |
//! | Timeout | Configurable per-token and total timeouts |
//!
//! ## Quick Start
//!
//! ### HTTP Server
//!
//! ```zig
//! const streaming = @import("abi").ai.streaming;
//!
//! var server = try streaming.StreamingServer.init(allocator, .{
//!     .address = "0.0.0.0:8080",
//!     .auth_token = "secret-token",
//!     .default_model_path = "./models/llama-7b.gguf",
//!     .preload_model = true,
//!     .enable_recovery = true,
//! });
//! defer server.deinit();
//!
//! try server.serve(); // Blocking
//! ```
//!
//! ### Basic Generator
//!
//! ```zig
//! var generator = streaming.StreamingGenerator.init(allocator, &model, .{
//!     .max_tokens = 256,
//!     .temperature = 0.7,
//! });
//! defer generator.deinit();
//!
//! try generator.start("Hello, world!");
//! while (try generator.next()) |token| {
//!     std.debug.print("{s}", .{token.text});
//!     if (token.is_end) break;
//! }
//! ```
//!
//! ### Recovery with Circuit Breaker
//!
//! ```zig
//! var recovery = try streaming.StreamRecovery.init(allocator, .{
//!     .circuit_breaker = .{ .failure_threshold = 5 },
//! });
//! defer recovery.deinit();
//!
//! if (recovery.isBackendAvailable(.openai)) {
//!     // Safe to make request
//!     recovery.recordSuccess(.openai);
//! } else {
//!     // Circuit is open, use fallback
//! }
//! ```
//!
//! ## Endpoints (when using StreamingServer)
//!
//! | Endpoint | Method | Description |
//! |----------|--------|-------------|
//! | `/v1/chat/completions` | POST | OpenAI-compatible chat completions |
//! | `/api/stream` | POST | Custom ABI streaming endpoint |
//! | `/api/stream/ws` | GET | WebSocket upgrade for bidirectional streaming |
//! | `/health` | GET | Health check (optionally unauthenticated) |
//! | `/metrics` | GET | Prometheus-style metrics snapshot |
//! | `/admin/reload` | POST | Hot-reload model without restart |
//! | `/v1/models` | GET | List available models (OpenAI-compatible) |

const std = @import("std");
const time = @import("../../../services/shared/time.zig");
const sync = @import("../../../services/shared/sync.zig");
pub const sse = @import("sse.zig");
pub const backpressure = @import("backpressure.zig");
pub const buffer = @import("buffer.zig");
pub const generator = @import("generator.zig");

// Streaming inference server
pub const server = @import("server.zig");
pub const websocket = @import("websocket.zig");
pub const backends = @import("backends/mod.zig");
pub const formats = @import("formats/mod.zig");

// Error recovery and resilience
pub const recovery = @import("recovery.zig");
pub const circuit_breaker = @import("circuit_breaker.zig");
pub const retry_config = @import("retry_config.zig");
pub const session_cache = @import("session_cache.zig");
pub const streaming_metrics = @import("metrics.zig");

// Server types

/// HTTP streaming server with SSE and WebSocket support.
pub const StreamingServer = server.StreamingServer;

/// Configuration for the streaming server.
pub const ServerConfig = server.ServerConfig;

/// Errors that can occur during server operations.
pub const StreamingServerError = server.StreamingServerError;

// WebSocket types

/// Handler for WebSocket connections with streaming support.
pub const WebSocketHandler = websocket.WebSocketHandler;

/// Configuration for WebSocket behavior.
pub const WebSocketConfig = websocket.WebSocketConfig;

/// WebSocket frame opcode (text, binary, ping, pong, close).
pub const WebSocketOpcode = websocket.Opcode;

/// WebSocket close status codes.
pub const WebSocketCloseCode = websocket.CloseCode;

/// Compute the Sec-WebSocket-Accept header value for handshake.
pub const computeWebSocketAcceptKey = websocket.computeAcceptKey;

// Backend types

/// Type of inference backend (local, openai, ollama, anthropic).
pub const BackendType = backends.BackendType;

/// Router for directing requests to appropriate backends.
pub const BackendRouter = backends.BackendRouter;

/// Interface to a specific inference backend.
pub const Backend = backends.Backend;

/// Generation configuration passed to backends.
pub const BackendGenerationConfig = backends.GenerationConfig;

// Recovery types

/// Manager for resilient streaming with retry, circuit breakers, and caching.
pub const StreamRecovery = recovery.StreamRecovery;

/// Configuration for stream recovery behavior.
pub const RecoveryConfig = recovery.RecoveryConfig;

/// Events emitted during recovery operations.
pub const RecoveryEvent = recovery.RecoveryEvent;

/// Callback for receiving recovery event notifications.
pub const RecoveryCallback = recovery.RecoveryCallback;

/// Per-backend circuit breaker for failure isolation.
pub const CircuitBreaker = circuit_breaker.CircuitBreaker;

/// Configuration for circuit breaker thresholds and timeouts.
pub const CircuitBreakerConfig = circuit_breaker.CircuitBreakerConfig;

/// State of a circuit breaker (closed, open, half_open).
pub const CircuitState = circuit_breaker.CircuitState;

/// Configuration for retry behavior with exponential backoff.
pub const StreamingRetryConfig = retry_config.StreamingRetryConfig;

/// Specification of which errors should trigger retry.
pub const StreamingRetryableErrors = retry_config.StreamingRetryableErrors;

/// Cache for storing tokens to enable stream resumption.
pub const SessionCache = session_cache.SessionCache;

/// Configuration for session cache behavior.
pub const SessionCacheConfig = session_cache.SessionCacheConfig;

/// A cached token for stream recovery.
pub const CachedToken = session_cache.CachedToken;

/// Metrics collector for streaming operations.
pub const StreamingMetrics = streaming_metrics.StreamingMetrics;

/// Configuration for metrics collection.
pub const StreamingMetricsConfig = streaming_metrics.StreamingMetricsConfig;

// Basic generator types (from generator.zig)

/// Token-by-token streaming generator for transformer models.
pub const StreamingGenerator = generator.StreamingGenerator;

/// Errors that can occur during streaming generation.
pub const StreamingError = generator.StreamingError;

/// State of a streaming generation session.
pub const StreamState = generator.StreamState;

/// Configuration for text generation parameters.
pub const GenerationConfig = generator.GenerationConfig;

/// Stream inference with a callback for each token.
pub const streamInference = generator.streamInference;

/// Format an array of stream tokens into a single string.
pub const formatStreamOutput = generator.formatStreamOutput;

/// Split tokens into chunks for batched transmission.
pub const createChunkedStream = generator.createChunkedStream;

// SSE types

/// Server-Sent Event data structure.
pub const SseEvent = sse.SseEvent;

/// Encoder for converting events to SSE wire format.
pub const SseEncoder = sse.SseEncoder;

/// Decoder for parsing SSE wire format.
pub const SseDecoder = sse.SseDecoder;

/// Configuration for SSE encoding behavior.
pub const SseConfig = sse.SseConfig;

// Backpressure types

/// Controller for managing flow control and backpressure.
pub const BackpressureController = backpressure.BackpressureController;

/// Strategy for handling backpressure (drop, block, buffer, sample).
pub const BackpressureStrategy = backpressure.BackpressureStrategy;

/// Configuration for backpressure thresholds.
pub const BackpressureConfig = backpressure.BackpressureConfig;

/// Current flow state (normal, throttled, blocked, recovering).
pub const FlowState = backpressure.FlowState;

/// Statistics about backpressure operations.
pub const BackpressureStats = backpressure.BackpressureStats;

/// Token bucket rate limiter for request throttling.
pub const RateLimiter = backpressure.RateLimiter;

// Buffer types

/// Buffer for temporarily storing tokens during streaming.
pub const TokenBuffer = buffer.TokenBuffer;

/// Configuration for token buffer behavior.
pub const BufferConfig = buffer.BufferConfig;

/// Strategy for buffer management (fifo, lifo, ring, priority).
pub const BufferStrategy = buffer.BufferStrategy;

/// Statistics about buffer usage.
pub const BufferStats = buffer.BufferStats;

/// Buffer that coalesces multiple tokens into larger chunks.
pub const CoalescingBuffer = buffer.CoalescingBuffer;

/// Stream token from AI model.
pub const StreamToken = struct {
    /// Token ID.
    id: u32,
    /// Decoded text.
    text: []const u8,
    /// Log probability (optional).
    log_prob: ?f32 = null,
    /// Is this the final token?
    is_end: bool = false,
    /// Timestamp in nanoseconds.
    timestamp_ns: i128 = 0,
    /// Token index in sequence.
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

/// Stream event types.
pub const StreamEventType = enum {
    token,
    start,
    end,
    error_event,
    metadata,
    heartbeat,
};

/// Stream event.
pub const StreamEvent = struct {
    event_type: StreamEventType,
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

/// Stream statistics.
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

/// Enhanced streaming generator with SSE support.
///
/// Combines token generation with SSE encoding, backpressure control, and
/// buffering for production-ready streaming. Suitable for use with HTTP
/// servers and real-time client connections.
///
/// ## Features
///
/// - Automatic SSE encoding of tokens
/// - Backpressure-aware flow control
/// - Token buffering during throttled periods
/// - Statistics tracking (tokens/second, total tokens, etc.)
///
/// ## Example
///
/// ```zig
/// var gen = EnhancedStreamingGenerator.init(allocator, .{});
/// defer gen.deinit();
///
/// try gen.start();
/// while (generating) {
///     if (try gen.emit(token)) |sse_data| {
///         try connection.write(sse_data);
///         allocator.free(sse_data);
///     }
/// }
/// const final = try gen.complete();
/// defer allocator.free(final);
/// ```
///
/// ## Error Handling
///
/// - `error.AlreadyStarted`: `start()` called when already started
/// - `error.NotStarted`: `emit()` called before `start()`
/// - `error.Completed`: `emit()` called after `complete()`
/// - `error.AlreadyCompleted`: `complete()` called twice
pub const EnhancedStreamingGenerator = struct {
    allocator: std.mem.Allocator,
    sse_encoder: SseEncoder,
    backpressure_ctrl: BackpressureController,
    token_buffer: TokenBuffer,
    stats: StreamStats,
    timer: time.Timer,
    is_started: bool,
    is_completed: bool,

    /// Initialize enhanced streaming generator with the given configuration.
    /// Returns error.TimerUnavailable if platform timer cannot be started.
    pub fn init(allocator: std.mem.Allocator, config: StreamConfig) error{TimerUnavailable}!EnhancedStreamingGenerator {
        return .{
            .allocator = allocator,
            .sse_encoder = SseEncoder.init(allocator, config.sse_config),
            .backpressure_ctrl = BackpressureController.init(config.backpressure_config) catch return error.TimerUnavailable,
            .token_buffer = TokenBuffer.init(allocator, config.buffer_config),
            .stats = .{},
            .timer = time.Timer.start() catch return error.TimerUnavailable,
            .is_started = false,
            .is_completed = false,
        };
    }

    /// Deinitialize.
    pub fn deinit(self: *EnhancedStreamingGenerator) void {
        self.sse_encoder.deinit();
        self.token_buffer.deinit();
        self.* = undefined;
    }

    /// Start streaming.
    pub fn start(self: *EnhancedStreamingGenerator) !void {
        if (self.is_started) return error.AlreadyStarted;

        self.is_started = true;
        self.stats.start_time_ns = self.timer.read();
    }

    /// Process and emit a token.
    pub fn emit(self: *EnhancedStreamingGenerator, token: StreamToken) !?[]u8 {
        if (!self.is_started) return error.NotStarted;
        if (self.is_completed) return error.Completed;

        // Check backpressure
        const flow = self.backpressure_ctrl.checkFlow();
        if (flow == .blocked) {
            // Buffer the token for later
            try self.token_buffer.push(token);
            return null;
        }

        // Update stats
        self.stats.total_tokens += 1;
        self.stats.total_chars += token.text.len;

        // Encode as SSE
        const event = StreamEvent.tokenEvent(token);
        return try self.sse_encoder.encode(event);
    }

    /// Flush buffered tokens.
    pub fn flush(self: *EnhancedStreamingGenerator) ![][]u8 {
        var results = std.ArrayListUnmanaged([]u8){};
        errdefer {
            for (results.items) |item| self.allocator.free(item);
            results.deinit(self.allocator);
        }

        while (self.token_buffer.pop()) |token| {
            const event = StreamEvent.tokenEvent(token);
            const encoded = try self.sse_encoder.encode(event);
            try results.append(self.allocator, encoded);

            self.stats.total_tokens += 1;
            self.stats.total_chars += token.text.len;
        }

        return results.toOwnedSlice(self.allocator);
    }

    /// Complete the stream.
    pub fn complete(self: *EnhancedStreamingGenerator) ![]u8 {
        if (self.is_completed) return error.AlreadyCompleted;

        self.is_completed = true;
        self.stats.end_time_ns = self.timer.read();

        const duration_ns = self.stats.end_time_ns - self.stats.start_time_ns;
        if (duration_ns > 0) {
            self.stats.tokens_per_second = @as(f64, @floatFromInt(self.stats.total_tokens)) *
                1_000_000_000.0 / @as(f64, @floatFromInt(duration_ns));
        }

        const event = StreamEvent.endEvent();
        return try self.sse_encoder.encode(event);
    }

    /// Get streaming statistics.
    pub fn getStats(self: *const EnhancedStreamingGenerator) StreamStats {
        return self.stats;
    }
};

/// Stream configuration.
pub const StreamConfig = struct {
    sse_config: sse.SseConfig = .{},
    backpressure_config: BackpressureConfig = .{},
    buffer_config: BufferConfig = .{},
};

/// Create a simple SSE stream from text chunks.
///
/// Encodes an array of text strings as a complete SSE event stream,
/// suitable for sending as an HTTP response body. Includes start,
/// token, and end events.
///
/// ## Parameters
///
/// - `allocator`: Memory allocator for the output buffer
/// - `chunks`: Array of text chunks to encode as token events
///
/// ## Returns
///
/// Complete SSE stream as a byte array, owned by the caller.
///
/// ## Example
///
/// ```zig
/// const chunks = &.{ "Hello", " ", "world", "!" };
/// const sse_stream = try createSseStream(allocator, chunks);
/// defer allocator.free(sse_stream);
///
/// // Send as HTTP response body with Content-Type: text/event-stream
/// try response.write(sse_stream);
/// ```
pub fn createSseStream(
    allocator: std.mem.Allocator,
    chunks: []const []const u8,
) ![]u8 {
    var output = std.ArrayListUnmanaged(u8){};
    errdefer output.deinit(allocator);

    var encoder = SseEncoder.init(allocator, .{});
    defer encoder.deinit();

    // Start event
    const start = try encoder.encode(StreamEvent.startEvent());
    defer allocator.free(start);
    try output.appendSlice(allocator, start);

    // Token events
    for (chunks, 0..) |chunk, i| {
        const token = StreamToken{
            .id = @intCast(i),
            .text = chunk,
            .sequence_index = i,
        };
        const event = StreamEvent.tokenEvent(token);
        const encoded = try encoder.encode(event);
        defer allocator.free(encoded);
        try output.appendSlice(allocator, encoded);
    }

    // End event
    const end = try encoder.encode(StreamEvent.endEvent());
    defer allocator.free(end);
    try output.appendSlice(allocator, end);

    return output.toOwnedSlice(allocator);
}

test "stream token" {
    const token = StreamToken{
        .id = 1,
        .text = "hello",
        .is_end = false,
    };

    try std.testing.expectEqual(@as(u32, 1), token.id);
    try std.testing.expectEqualStrings("hello", token.text);
}

test "stream event creation" {
    const token_event = StreamEvent.tokenEvent(.{
        .id = 1,
        .text = "test",
    });
    try std.testing.expectEqual(StreamEventType.token, token_event.event_type);

    const start_event = StreamEvent.startEvent();
    try std.testing.expectEqual(StreamEventType.start, start_event.event_type);

    const end_event = StreamEvent.endEvent();
    try std.testing.expectEqual(StreamEventType.end, end_event.event_type);
}

test {
    _ = sse;
    _ = backpressure;
    _ = buffer;
    _ = generator;
    _ = server;
    _ = websocket;
    _ = backends;
    _ = formats;
    _ = recovery;
    _ = circuit_breaker;
    _ = retry_config;
    _ = session_cache;
    _ = streaming_metrics;
    _ = @import("request_types.zig");
    _ = @import("server_test.zig");
}
