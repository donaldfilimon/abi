//! Stub implementation for enhanced streaming when AI features are disabled.

const std = @import("std");

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

    pub fn tokenEvent(token: StreamToken) StreamEvent {
        return .{ .event_type = .token, .token = token };
    }

    pub fn startEvent() StreamEvent {
        return .{ .event_type = .start };
    }

    pub fn endEvent() StreamEvent {
        return .{ .event_type = .end };
    }
};

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
};

/// Stub stream config.
pub const StreamConfig = struct {
    sse_config: SseConfig = .{},
    backpressure_config: BackpressureConfig = .{},
    buffer_config: BufferConfig = .{},
};

/// Stub stream stats.
pub const StreamStats = struct {
    total_tokens: usize = 0,
    total_chars: usize = 0,
    tokens_per_second: f64 = 0,
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
