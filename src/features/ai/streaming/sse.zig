//! Server-Sent Events (SSE) encoding and decoding.
//!
//! Provides utilities for encoding stream events to SSE format
//! and decoding SSE streams back to events.

const std = @import("std");
const mod = @import("mod.zig");
const StreamEvent = mod.StreamEvent;
const StreamEventType = mod.StreamEventType;
const StreamToken = mod.StreamToken;

/// SSE configuration.
pub const SseConfig = struct {
    /// Event type prefix.
    event_prefix: []const u8 = "",
    /// Include timestamps.
    include_timestamp: bool = false,
    /// Include event IDs.
    include_id: bool = true,
    /// Retry interval hint (ms).
    retry_ms: ?u32 = null,
    /// Comment prefix for heartbeats.
    comment_prefix: []const u8 = ":",
};

/// A parsed SSE event.
pub const SseEvent = struct {
    /// Event type (optional).
    event: ?[]const u8,
    /// Event data.
    data: []const u8,
    /// Event ID (optional).
    id: ?[]const u8,
    /// Retry interval (optional).
    retry: ?u32,

    pub fn deinit(self: *SseEvent, allocator: std.mem.Allocator) void {
        if (self.event) |e| allocator.free(e);
        allocator.free(self.data);
        if (self.id) |i| allocator.free(i);
        self.* = undefined;
    }
};

/// SSE encoder for stream events.
pub const SseEncoder = struct {
    allocator: std.mem.Allocator,
    config: SseConfig,
    event_counter: u64,

    /// Initialize SSE encoder.
    pub fn init(allocator: std.mem.Allocator, config: SseConfig) SseEncoder {
        return .{
            .allocator = allocator,
            .config = config,
            .event_counter = 0,
        };
    }

    /// Deinitialize.
    pub fn deinit(self: *SseEncoder) void {
        self.* = undefined;
    }

    /// Encode a stream event to SSE format.
    pub fn encode(self: *SseEncoder, event: StreamEvent) ![]u8 {
        var output = std.ArrayListUnmanaged(u8).empty;
        errdefer output.deinit(self.allocator);

        // Event ID
        if (self.config.include_id) {
            try output.appendSlice(self.allocator, "id: ");
            var buf: [20]u8 = undefined;
            const id_str = std.fmt.bufPrint(&buf, "{d}", .{self.event_counter}) catch "0";
            try output.appendSlice(self.allocator, id_str);
            try output.append(self.allocator, '\n');
            self.event_counter += 1;
        }

        // Event type
        const event_name = self.eventTypeName(event.event_type);
        if (event_name.len > 0) {
            try output.appendSlice(self.allocator, "event: ");
            if (self.config.event_prefix.len > 0) {
                try output.appendSlice(self.allocator, self.config.event_prefix);
            }
            try output.appendSlice(self.allocator, event_name);
            try output.append(self.allocator, '\n');
        }

        // Retry hint
        if (self.config.retry_ms) |retry| {
            try output.appendSlice(self.allocator, "retry: ");
            var buf: [20]u8 = undefined;
            const retry_str = std.fmt.bufPrint(&buf, "{d}", .{retry}) catch "1000";
            try output.appendSlice(self.allocator, retry_str);
            try output.append(self.allocator, '\n');
        }

        // Data
        const data = try self.serializeEventData(event);
        defer self.allocator.free(data);

        // Split data by newlines and prefix each with "data: "
        var lines = std.mem.splitSequence(u8, data, "\n");
        while (lines.next()) |line| {
            try output.appendSlice(self.allocator, "data: ");
            try output.appendSlice(self.allocator, line);
            try output.append(self.allocator, '\n');
        }

        // End with double newline
        try output.append(self.allocator, '\n');

        return output.toOwnedSlice(self.allocator);
    }

    /// Encode a heartbeat comment.
    pub fn encodeHeartbeat(self: *SseEncoder) ![]u8 {
        var output = std.ArrayListUnmanaged(u8).empty;
        errdefer output.deinit(self.allocator);

        try output.appendSlice(self.allocator, self.config.comment_prefix);
        try output.appendSlice(self.allocator, " heartbeat\n\n");

        return output.toOwnedSlice(self.allocator);
    }

    /// Get event type name.
    fn eventTypeName(_: *const SseEncoder, event_type: StreamEventType) []const u8 {
        return switch (event_type) {
            .token => "token",
            .start => "start",
            .end => "end",
            .error_event => "error",
            .metadata => "metadata",
            .heartbeat => "",
        };
    }

    /// Serialize event data to JSON-like format.
    fn serializeEventData(self: *SseEncoder, event: StreamEvent) ![]u8 {
        var output = std.ArrayListUnmanaged(u8).empty;
        errdefer output.deinit(self.allocator);

        try output.append(self.allocator, '{');

        switch (event.event_type) {
            .token => {
                if (event.token) |token| {
                    try output.appendSlice(self.allocator, "\"text\":\"");
                    // Escape JSON string
                    for (token.text) |c| {
                        switch (c) {
                            '"' => try output.appendSlice(self.allocator, "\\\""),
                            '\\' => try output.appendSlice(self.allocator, "\\\\"),
                            '\n' => try output.appendSlice(self.allocator, "\\n"),
                            '\r' => try output.appendSlice(self.allocator, "\\r"),
                            '\t' => try output.appendSlice(self.allocator, "\\t"),
                            else => try output.append(self.allocator, c),
                        }
                    }
                    try output.appendSlice(self.allocator, "\",\"id\":");
                    var buf: [20]u8 = undefined;
                    const id_str = std.fmt.bufPrint(&buf, "{d}", .{token.id}) catch "0";
                    try output.appendSlice(self.allocator, id_str);

                    if (token.is_end) {
                        try output.appendSlice(self.allocator, ",\"done\":true");
                    }
                }
            },
            .start => {
                try output.appendSlice(self.allocator, "\"status\":\"started\"");
            },
            .end => {
                try output.appendSlice(self.allocator, "\"status\":\"completed\"");
            },
            .error_event => {
                try output.appendSlice(self.allocator, "\"error\":\"");
                if (event.error_message) |msg| {
                    try output.appendSlice(self.allocator, msg);
                }
                try output.append(self.allocator, '"');
            },
            .metadata => {
                try output.appendSlice(self.allocator, "\"metadata\":");
                if (event.metadata) |meta| {
                    try output.appendSlice(self.allocator, meta);
                } else {
                    try output.appendSlice(self.allocator, "null");
                }
            },
            .heartbeat => {
                try output.appendSlice(self.allocator, "\"type\":\"heartbeat\"");
            },
        }

        try output.append(self.allocator, '}');

        return output.toOwnedSlice(self.allocator);
    }
};

/// SSE decoder for parsing SSE streams.
pub const SseDecoder = struct {
    allocator: std.mem.Allocator,
    buffer: std.ArrayListUnmanaged(u8),
    last_event_id: ?[]u8,

    /// Initialize SSE decoder.
    pub fn init(allocator: std.mem.Allocator) SseDecoder {
        return .{
            .allocator = allocator,
            .buffer = .{},
            .last_event_id = null,
        };
    }

    /// Deinitialize.
    pub fn deinit(self: *SseDecoder) void {
        self.buffer.deinit(self.allocator);
        if (self.last_event_id) |id| self.allocator.free(id);
        self.* = undefined;
    }

    /// Feed data to the decoder and get parsed events.
    pub fn feed(self: *SseDecoder, data: []const u8) ![]SseEvent {
        try self.buffer.appendSlice(self.allocator, data);

        var events = std.ArrayListUnmanaged(SseEvent).empty;
        errdefer {
            for (events.items) |*e| e.deinit(self.allocator);
            events.deinit(self.allocator);
        }

        // Parse complete events (ending with double newline)
        while (std.mem.indexOf(u8, self.buffer.items, "\n\n")) |end| {
            const event_data = self.buffer.items[0..end];

            if (try self.parseEvent(event_data)) |event| {
                try events.append(self.allocator, event);
            }

            // Remove parsed data from buffer
            const remaining = self.buffer.items[end + 2 ..];
            std.mem.copyForwards(u8, self.buffer.items[0..remaining.len], remaining);
            self.buffer.shrinkRetainingCapacity(remaining.len);
        }

        return events.toOwnedSlice(self.allocator);
    }

    /// Parse a single SSE event.
    fn parseEvent(self: *SseDecoder, data: []const u8) !?SseEvent {
        var event_type: ?[]u8 = null;
        var event_data = std.ArrayListUnmanaged(u8).empty;
        defer event_data.deinit(self.allocator);
        var event_id: ?[]u8 = null;
        var retry: ?u32 = null;

        var lines = std.mem.splitSequence(u8, data, "\n");
        while (lines.next()) |line| {
            if (line.len == 0) continue;

            // Skip comments
            if (line[0] == ':') continue;

            // Parse field
            if (std.mem.indexOf(u8, line, ":")) |colon_pos| {
                const field = line[0..colon_pos];
                var value = line[colon_pos + 1 ..];

                // Skip leading space
                if (value.len > 0 and value[0] == ' ') {
                    value = value[1..];
                }

                if (std.mem.eql(u8, field, "event")) {
                    event_type = try self.allocator.dupe(u8, value);
                } else if (std.mem.eql(u8, field, "data")) {
                    if (event_data.items.len > 0) {
                        try event_data.append(self.allocator, '\n');
                    }
                    try event_data.appendSlice(self.allocator, value);
                } else if (std.mem.eql(u8, field, "id")) {
                    event_id = try self.allocator.dupe(u8, value);
                    if (self.last_event_id) |old| self.allocator.free(old);
                    self.last_event_id = try self.allocator.dupe(u8, value);
                } else if (std.mem.eql(u8, field, "retry")) {
                    retry = std.fmt.parseInt(u32, value, 10) catch null;
                }
            }
        }

        if (event_data.items.len == 0 and event_type == null) {
            if (event_id) |id| self.allocator.free(id);
            return null;
        }

        return SseEvent{
            .event = event_type,
            .data = try event_data.toOwnedSlice(self.allocator),
            .id = event_id,
            .retry = retry,
        };
    }

    /// Get last event ID for reconnection.
    pub fn getLastEventId(self: *const SseDecoder) ?[]const u8 {
        return self.last_event_id;
    }

    /// Clear the buffer.
    pub fn clear(self: *SseDecoder) void {
        self.buffer.clearRetainingCapacity();
    }
};

test "sse encoder basic" {
    const allocator = std.testing.allocator;
    var encoder = SseEncoder.init(allocator, .{});
    defer encoder.deinit();

    const event = StreamEvent.tokenEvent(.{
        .id = 1,
        .text = "hello",
    });

    const encoded = try encoder.encode(event);
    defer allocator.free(encoded);

    try std.testing.expect(std.mem.indexOf(u8, encoded, "event: token") != null);
    try std.testing.expect(std.mem.indexOf(u8, encoded, "data: ") != null);
    try std.testing.expect(std.mem.indexOf(u8, encoded, "hello") != null);
}

test "sse encoder escaping" {
    const allocator = std.testing.allocator;
    var encoder = SseEncoder.init(allocator, .{});
    defer encoder.deinit();

    const event = StreamEvent.tokenEvent(.{
        .id = 1,
        .text = "line1\nline2",
    });

    const encoded = try encoder.encode(event);
    defer allocator.free(encoded);

    try std.testing.expect(std.mem.indexOf(u8, encoded, "\\n") != null);
}

test "sse decoder basic" {
    const allocator = std.testing.allocator;
    var decoder = SseDecoder.init(allocator);
    defer decoder.deinit();

    const sse_data = "id: 1\nevent: token\ndata: {\"text\":\"hello\"}\n\n";

    const events = try decoder.feed(sse_data);
    defer {
        for (events) |*e| {
            var event = e.*;
            event.deinit(allocator);
        }
        allocator.free(events);
    }

    try std.testing.expectEqual(@as(usize, 1), events.len);
    try std.testing.expectEqualStrings("token", events[0].event.?);
}

test "sse decoder partial data" {
    const allocator = std.testing.allocator;
    var decoder = SseDecoder.init(allocator);
    defer decoder.deinit();

    // Feed partial data
    const events1 = try decoder.feed("id: 1\nevent: token\n");
    defer allocator.free(events1);
    try std.testing.expectEqual(@as(usize, 0), events1.len);

    // Complete the event
    const events2 = try decoder.feed("data: test\n\n");
    defer {
        for (events2) |*e| {
            var event = e.*;
            event.deinit(allocator);
        }
        allocator.free(events2);
    }
    try std.testing.expectEqual(@as(usize, 1), events2.len);
}

test "sse heartbeat" {
    const allocator = std.testing.allocator;
    var encoder = SseEncoder.init(allocator, .{});
    defer encoder.deinit();

    const heartbeat = try encoder.encodeHeartbeat();
    defer allocator.free(heartbeat);

    try std.testing.expect(std.mem.indexOf(u8, heartbeat, "heartbeat") != null);
}
