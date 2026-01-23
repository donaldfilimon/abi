//! Enhanced streaming support for AI model responses.
//!
//! Provides advanced streaming capabilities including:
//! - Basic streaming generator for transformer models
//! - Server-Sent Events (SSE) encoding/decoding
//! - Backpressure control for flow management
//! - Token buffering strategies
//! - Stream transformations

const std = @import("std");
pub const sse = @import("sse.zig");
pub const backpressure = @import("backpressure.zig");
pub const buffer = @import("buffer.zig");
pub const generator = @import("generator.zig");

// Basic generator types (from generator.zig)
pub const StreamingGenerator = generator.StreamingGenerator;
pub const StreamingError = generator.StreamingError;
pub const StreamState = generator.StreamState;
pub const GenerationConfig = generator.GenerationConfig;
pub const streamInference = generator.streamInference;
pub const formatStreamOutput = generator.formatStreamOutput;
pub const createChunkedStream = generator.createChunkedStream;

// SSE types
pub const SseEvent = sse.SseEvent;
pub const SseEncoder = sse.SseEncoder;
pub const SseDecoder = sse.SseDecoder;
pub const SseConfig = sse.SseConfig;

// Backpressure types
pub const BackpressureController = backpressure.BackpressureController;
pub const BackpressureStrategy = backpressure.BackpressureStrategy;
pub const BackpressureConfig = backpressure.BackpressureConfig;
pub const FlowState = backpressure.FlowState;
pub const BackpressureStats = backpressure.BackpressureStats;
pub const RateLimiter = backpressure.RateLimiter;

// Buffer types
pub const TokenBuffer = buffer.TokenBuffer;
pub const BufferConfig = buffer.BufferConfig;
pub const BufferStrategy = buffer.BufferStrategy;
pub const BufferStats = buffer.BufferStats;
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
pub const EnhancedStreamingGenerator = struct {
    allocator: std.mem.Allocator,
    sse_encoder: SseEncoder,
    backpressure_ctrl: BackpressureController,
    token_buffer: TokenBuffer,
    stats: StreamStats,
    timer: std.time.Timer,
    is_started: bool,
    is_completed: bool,

    /// Initialize enhanced streaming generator.
    pub fn init(allocator: std.mem.Allocator, config: StreamConfig) EnhancedStreamingGenerator {
        return .{
            .allocator = allocator,
            .sse_encoder = SseEncoder.init(allocator, config.sse_config),
            .backpressure_ctrl = BackpressureController.init(config.backpressure_config),
            .token_buffer = TokenBuffer.init(allocator, config.buffer_config),
            .stats = .{},
            .timer = std.time.Timer.start() catch unreachable,
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
