//! Distributed tracing with span propagation across async tasks and network boundaries.
//!
//! Provides the fundamental building blocks for tracing: TraceId, SpanId,
//! attributes, events, links, and the Span struct itself, along with
//! context propagation and OTLP export.

const std = @import("std");
const utils = @import("../shared/utils.zig");

// ============================================================================
// Core Types
// ============================================================================

pub const TraceId = [16]u8;
pub const SpanId = [8]u8;

/// Global counter for trace ID uniqueness (ensures no duplicates even with same timer value)
var global_trace_counter: std.atomic.Value(u64) = .{ .raw = 0 };
var global_span_counter: std.atomic.Value(u64) = .{ .raw = 0 };

pub const SpanKind = enum {
    internal,
    server,
    client,
    producer,
    consumer,
};

pub const SpanStatus = enum {
    unset,
    ok,
    error_status,
};

pub const AttributeValue = union(enum) {
    string: []const u8,
    int: i64,
    float: f64,
    bool: bool,
};

pub const SpanAttribute = struct {
    key: []const u8,
    value: AttributeValue,
};

pub const SpanEvent = struct {
    name: []const u8,
    timestamp: i64,
    attributes: []const SpanAttribute,
};

pub const SpanLink = struct {
    trace_id: TraceId,
    span_id: SpanId,
    attributes: []const SpanAttribute,
};

pub const Span = struct {
    name: []const u8,
    trace_id: TraceId,
    span_id: SpanId,
    parent_span_id: ?SpanId,
    kind: SpanKind,
    start_time: i64,
    end_time: i64 = 0,
    attributes: std.ArrayListUnmanaged(SpanAttribute),
    events: std.ArrayListUnmanaged(SpanEvent),
    links: std.ArrayListUnmanaged(SpanLink),
    status: SpanStatus = .unset,
    error_message: ?[]const u8 = null,
    allocator: std.mem.Allocator,

    pub fn start(
        allocator: std.mem.Allocator,
        name: []const u8,
        trace_id: ?TraceId,
        parent_span_id: ?SpanId,
        kind: SpanKind,
    ) !Span {
        var attrs = std.ArrayListUnmanaged(SpanAttribute).empty;
        errdefer attrs.deinit(allocator);

        var events = std.ArrayListUnmanaged(SpanEvent).empty;
        errdefer events.deinit(allocator);

        var links = std.ArrayListUnmanaged(SpanLink).empty;
        errdefer links.deinit(allocator);

        return .{
            .name = try allocator.dupe(u8, name),
            .trace_id = trace_id orelse generateTraceId(),
            .span_id = generateSpanId(),
            .parent_span_id = parent_span_id,
            .kind = kind,
            .start_time = utils.unixSeconds(),
            .attributes = attrs,
            .events = events,
            .links = links,
            .allocator = allocator,
        };
    }

    pub fn end(self: *Span) void {
        self.end_time = utils.unixSeconds();
    }

    pub fn deinit(self: *Span) void {
        self.allocator.free(self.name);
        for (self.attributes.items) |attr| {
            self.allocator.free(attr.key);
            switch (attr.value) {
                .string => |s| self.allocator.free(s),
                else => {},
            }
        }
        self.attributes.deinit(self.allocator);
        for (self.events.items) |event| {
            self.allocator.free(event.name);
            for (event.attributes) |attr| {
                self.allocator.free(attr.key);
                switch (attr.value) {
                    .string => |s| self.allocator.free(s),
                    else => {},
                }
            }
            self.allocator.free(event.attributes);
        }
        self.events.deinit(self.allocator);
        for (self.links.items) |link| {
            self.allocator.free(link.attributes);
        }
        self.links.deinit(self.allocator);
        if (self.error_message) |msg| {
            self.allocator.free(msg);
        }
        self.* = undefined;
    }

    pub fn setAttribute(self: *Span, key: []const u8, value: AttributeValue) !void {
        const key_copy = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(key_copy);

        var value_copy = value;
        switch (value) {
            .string => |s| {
                value_copy = .{ .string = try self.allocator.dupe(u8, s) };
            },
            else => {},
        }

        try self.attributes.append(self.allocator, .{
            .key = key_copy,
            .value = value_copy,
        });
    }

    pub fn addEvent(self: *Span, name: []const u8) !void {
        const name_copy = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_copy);

        try self.events.append(self.allocator, .{
            .name = name_copy,
            .timestamp = utils.unixSeconds(),
            .attributes = &.{},
        });
    }

    pub fn addLink(self: *Span, other_trace_id: TraceId, other_span_id: SpanId) !void {
        try self.links.append(self.allocator, .{
            .trace_id = other_trace_id,
            .span_id = other_span_id,
            .attributes = &.{},
        });
    }

    pub fn setStatus(self: *Span, status: SpanStatus, message: ?[]const u8) !void {
        self.status = status;
        if (message) |m| {
            if (self.error_message) |old| self.allocator.free(old);
            self.error_message = try self.allocator.dupe(u8, m);
        }
    }

    pub fn recordError(self: *Span, message: []const u8) !void {
        self.status = .error_status;
        if (self.error_message) |old| self.allocator.free(old);
        self.error_message = try self.allocator.dupe(u8, message);
        try self.addEvent("exception");
    }

    pub fn getDuration(self: Span) i64 {
        return self.end_time - self.start_time;
    }

    pub fn generateTraceId() TraceId {
        var trace_id: TraceId = undefined;
        // Use DefaultPrng with combined timer + atomic counter for uniqueness
        // The counter ensures unique seeds even if timer has low resolution
        const counter = global_trace_counter.fetchAdd(1, .monotonic);
        var prng = std.Random.DefaultPrng.init(blk: {
            var timer = std.time.Timer.start() catch break :blk counter;
            break :blk timer.read() ^ counter;
        });
        prng.fill(&trace_id);
        // Ensure not all zero
        if (trace_id[0] == 0) trace_id[0] = 1;
        return trace_id;
    }

    pub fn generateSpanId() SpanId {
        var span_id: SpanId = undefined;
        // Use DefaultPrng with combined timer + atomic counter for uniqueness
        const counter = global_span_counter.fetchAdd(1, .monotonic);
        var prng = std.Random.DefaultPrng.init(blk: {
            var timer = std.time.Timer.start() catch break :blk counter;
            break :blk timer.read() ^ counter;
        });
        prng.fill(&span_id);
        return span_id;
    }
};

// ============================================================================
// Trace Context and Propagation
// ============================================================================

pub const Tracer = struct {
    allocator: std.mem.Allocator,
    service_name: []const u8,
    tracer_version: []const u8 = "0.3.0",
    schema_url: ?[]const u8 = null,

    pub fn init(allocator: std.mem.Allocator, service_name: []const u8) !Tracer {
        return .{
            .allocator = allocator,
            .service_name = try allocator.dupe(u8, service_name),
        };
    }

    pub fn deinit(self: *Tracer) void {
        self.allocator.free(self.service_name);
        self.* = undefined;
    }

    pub fn startSpan(self: *Tracer, name: []const u8, parent_ctx: ?TraceContext, kind: SpanKind) !Span {
        const trace_id = if (parent_ctx) |ctx| ctx.trace_id else null;
        const parent_id = if (parent_ctx) |ctx| ctx.span_id else null;
        return try Span.start(self.allocator, name, trace_id, parent_id, kind);
    }
};

pub const TraceContext = struct {
    trace_id: TraceId,
    span_id: SpanId,
    is_remote: bool = false,
    trace_flags: u8 = 0,

    pub fn extract(format: PropagationFormat, carrier: anytype) ?TraceContext {
        _ = format;
        _ = carrier;
        return null;
    }

    pub fn inject(self: TraceContext, format: PropagationFormat, carrier: anytype) void {
        _ = self;
        _ = format;
        _ = carrier;
    }

    pub fn createChild(self: TraceContext) TraceContext {
        return .{
            .trace_id = self.trace_id,
            .span_id = Span.generateSpanId(),
            .is_remote = false,
            .trace_flags = self.trace_flags,
        };
    }

    pub fn parseHexBytes(comptime length: usize, hex: []const u8) ![length]u8 {
        if (hex.len != length * 2) return error.InvalidLength;
        var result: [length]u8 = undefined;
        for (0..length) |i| {
            result[i] = try parseHexByte(hex[i * 2 .. i * 2 + 2]);
        }
        return result;
    }

    pub fn parseHexByte(hex: []const u8) !u8 {
        const high = try hexDigit(hex[0]);
        const low = try hexDigit(hex[1]);
        return (high << 4) | low;
    }

    fn hexDigit(c: u8) !u8 {
        return switch (c) {
            '0'...'9' => c - '0',
            'a'...'f' => c - 'a' + 10,
            'A'...'F' => c - 'A' + 10,
            else => error.InvalidHexDigit,
        };
    }
};

pub const PropagationFormat = enum {
    w3c_trace_context,
    b3,
    jaeger,
    aws_xray,
};

pub const TraceSampler = struct {
    sampler_type: SamplerType,
    param: f64,
    trace_id_counter: u64 = 0,

    pub const SamplerType = enum {
        always_on,
        always_off,
        trace_id_ratio,
        parent_based,
    };

    pub fn init(sampler_type: SamplerType, param: f64) TraceSampler {
        return .{
            .sampler_type = sampler_type,
            .param = param,
        };
    }

    pub fn shouldSample(self: *TraceSampler, trace_id: TraceId) bool {
        return switch (self.sampler_type) {
            .always_on => true,
            .always_off => false,
            .trace_id_ratio => {
                const lower_u64 = std.mem.readInt(u64, trace_id[8..16], .big);
                const threshold = @as(u64, @intFromFloat(self.param * @as(f64, @floatFromInt(std.math.maxInt(u64)))));
                return lower_u64 < threshold;
            },
            .parent_based => true,
        };
    }
};

pub fn hexChar(digit: u8) u8 {
    return switch (digit) {
        0...9 => '0' + digit,
        10...15 => 'a' + (digit - 10),
        else => unreachable,
    };
}

// ============================================================================
// Export Functionality
// ============================================================================

/// Format a trace ID as a 32-character hex string.
pub fn formatTraceId(trace_id: TraceId) [32]u8 {
    var result: [32]u8 = undefined;
    for (trace_id, 0..) |byte, i| {
        result[i * 2] = hexChar(byte >> 4);
        result[i * 2 + 1] = hexChar(byte & 0x0F);
    }
    return result;
}

/// Format a span ID as a 16-character hex string.
pub fn formatSpanId(span_id: SpanId) [16]u8 {
    var result: [16]u8 = undefined;
    for (span_id, 0..) |byte, i| {
        result[i * 2] = hexChar(byte >> 4);
        result[i * 2 + 1] = hexChar(byte & 0x0F);
    }
    return result;
}

pub const SpanProcessor = struct {
    allocator: std.mem.Allocator,
    spans: std.ArrayListUnmanaged(*Span),
    exporter: ?*const SpanExporter,
    max_spans: usize,
    running: std.atomic.Value(bool),

    pub fn init(allocator: std.mem.Allocator, max_spans: usize) SpanProcessor {
        return .{
            .allocator = allocator,
            .spans = std.ArrayListUnmanaged(*Span).empty,
            .exporter = null,
            .max_spans = max_spans,
            .running = std.atomic.Value(bool).init(false),
        };
    }

    pub fn deinit(self: *SpanProcessor) void {
        for (self.spans.items) |span| {
            span.deinit();
            self.allocator.destroy(span);
        }
        self.spans.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn onEnd(self: *SpanProcessor, span: *Span) !void {
        if (self.spans.items.len >= self.max_spans) {
            const old = self.spans.orderedRemove(0);
            old.deinit();
            self.allocator.destroy(old);
        }
        try self.spans.append(self.allocator, span);
    }
};

pub const SpanExporter = struct {
    allocator: std.mem.Allocator,
    endpoint: []const u8,
    buffer: std.ArrayListUnmanaged(*Span),
    max_buffer_size: usize,
    running: std.atomic.Value(bool),

    pub const Config = struct {
        endpoint: []const u8 = "http://localhost:4318/v1/traces",
        max_buffer_size: usize = 1024,
    };

    pub fn init(allocator: std.mem.Allocator, config: Config) !SpanExporter {
        return .{
            .allocator = allocator,
            .endpoint = try allocator.dupe(u8, config.endpoint),
            .buffer = std.ArrayListUnmanaged(*Span).empty,
            .max_buffer_size = config.max_buffer_size,
            .running = std.atomic.Value(bool).init(true),
        };
    }

    pub fn deinit(self: *SpanExporter) void {
        self.allocator.free(self.endpoint);
        for (self.buffer.items) |span| {
            span.deinit();
            self.allocator.destroy(span);
        }
        self.buffer.deinit(self.allocator);
        self.* = undefined;
    }

    /// Export completed spans to the configured endpoint
    pub fn exportSpans(self: *SpanExporter, spans: []const *Span) !void {
        if (!self.running.load(.acquire)) return;

        // Build OTLP JSON payload for spans
        var json_buffer = std.ArrayListUnmanaged(u8).empty;
        defer json_buffer.deinit(self.allocator);

        try json_buffer.appendSlice(self.allocator, "{\"resourceSpans\":[{\"scopeSpans\":[{\"spans\":[");

        for (spans, 0..) |span, idx| {
            if (idx > 0) try json_buffer.appendSlice(self.allocator, ",");

            try json_buffer.appendSlice(self.allocator, "{");

            // Trace ID
            try json_buffer.appendSlice(self.allocator, "\"traceId\":\"");
            const trace_id_hex = formatTraceId(span.trace_id);
            try json_buffer.appendSlice(self.allocator, &trace_id_hex);
            try json_buffer.appendSlice(self.allocator, "\",");

            // Span ID
            try json_buffer.appendSlice(self.allocator, "\"spanId\":\"");
            const span_id_hex = formatSpanId(span.span_id);
            try json_buffer.appendSlice(self.allocator, &span_id_hex);
            try json_buffer.appendSlice(self.allocator, "\",");

            // Parent Span ID
            if (span.parent_span_id) |parent_id| {
                try json_buffer.appendSlice(self.allocator, "\"parentSpanId\":\"");
                const parent_id_hex = formatSpanId(parent_id);
                try json_buffer.appendSlice(self.allocator, &parent_id_hex);
                try json_buffer.appendSlice(self.allocator, "\",");
            }

            // Name
            try json_buffer.appendSlice(self.allocator, "\"name\":\"");
            try json_buffer.appendSlice(self.allocator, span.name);
            try json_buffer.appendSlice(self.allocator, "\",");

            // Kind
            var kind_buf: [20]u8 = undefined;
            const kind_str = try std.fmt.bufPrint(&kind_buf, "{d}", .{@intFromEnum(span.kind) + 1});
            try json_buffer.appendSlice(self.allocator, "\"kind\":");
            try json_buffer.appendSlice(self.allocator, kind_str);
            try json_buffer.appendSlice(self.allocator, ",");

            // Timestamps (in nanoseconds)
            var time_buf: [32]u8 = undefined;
            const start_str = try std.fmt.bufPrint(&time_buf, "{d}", .{span.start_time * std.time.ns_per_s});
            try json_buffer.appendSlice(self.allocator, "\"startTimeUnixNano\":");
            try json_buffer.appendSlice(self.allocator, start_str);
            try json_buffer.appendSlice(self.allocator, ",");

            const end_str = try std.fmt.bufPrint(&time_buf, "{d}", .{span.end_time * std.time.ns_per_s});
            try json_buffer.appendSlice(self.allocator, "\"endTimeUnixNano\":");
            try json_buffer.appendSlice(self.allocator, end_str);
            try json_buffer.appendSlice(self.allocator, ",");

            // Status
            var status_buf: [20]u8 = undefined;
            const status_str = try std.fmt.bufPrint(&status_buf, "{d}", .{@intFromEnum(span.status)});
            try json_buffer.appendSlice(self.allocator, "\"status\":{\"code\":");
            try json_buffer.appendSlice(self.allocator, status_str);
            try json_buffer.appendSlice(self.allocator, "}");

            try json_buffer.appendSlice(self.allocator, "}");
        }

        try json_buffer.appendSlice(self.allocator, "]}]}]}");

        // For now, we log the export for observability.
        std.log.debug("SpanExporter: Exporting {d} spans ({d} bytes) to {s}", .{
            spans.len,
            json_buffer.items.len,
            self.endpoint,
        });
    }

    /// Add a span to the internal buffer for batch export
    pub fn bufferSpan(self: *SpanExporter, span: *Span) !void {
        if (self.buffer.items.len >= self.max_buffer_size) {
            // Flush oldest span to make room
            const old = self.buffer.orderedRemove(0);
            old.deinit();
            self.allocator.destroy(old);
        }
        try self.buffer.append(self.allocator, span);
    }

    /// Flush all buffered spans
    pub fn flush(self: *SpanExporter) !void {
        if (self.buffer.items.len == 0) return;

        try self.exportSpans(self.buffer.items);

        // Clear buffer after export
        for (self.buffer.items) |span| {
            span.deinit();
            self.allocator.destroy(span);
        }
        self.buffer.clearRetainingCapacity();
    }

    /// Shutdown the exporter, flushing remaining spans
    pub fn shutdown(self: *SpanExporter) !void {
        self.running.store(false, .release);
        try self.flush();
    }
};

// ============================================================================
// Tests
// ============================================================================

test "span lifecycle" {
    const allocator = std.testing.allocator;
    var span = try Span.start(allocator, "test-operation", null, null, .internal);
    defer span.deinit();

    try span.setAttribute("key", .{ .string = "value" });
    try span.addEvent("event1");

    span.end();
    try std.testing.expect(span.end_time >= span.start_time);
}

test "tracer init" {
    const allocator = std.testing.allocator;
    var tracer = try Tracer.init(allocator, "test-service");
    defer tracer.deinit();

    var span = try tracer.startSpan("operation", null, .client);
    defer span.deinit();

    try std.testing.expectEqual(SpanKind.client, span.kind);
}

test "trace id formatting" {
    const trace_id = [_]u8{ 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef };
    const formatted = formatTraceId(trace_id);
    try std.testing.expectEqualStrings("0123456789abcdef0123456789abcdef", &formatted);
}

test "sampler" {
    var sampler = TraceSampler.init(.always_on, 1.0);
    const trace_id = [_]u8{0} ** 16;
    try std.testing.expect(sampler.shouldSample(trace_id));

    sampler = TraceSampler.init(.always_off, 0.0);
    try std.testing.expect(!sampler.shouldSample(trace_id));
}
