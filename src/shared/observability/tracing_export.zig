//! Span processing and export functionality for distributed tracing.
//!
//! Provides SpanProcessor for buffering and SpanExporter for sending
//! spans to OTLP-compatible endpoints.

const std = @import("std");
const span_types = @import("tracing_span.zig");
const context = @import("tracing_context.zig");

pub const TraceId = span_types.TraceId;
pub const SpanId = span_types.SpanId;
pub const Span = span_types.Span;

pub const hexChar = context.hexChar;

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
        var aw = std.Io.Writer.Allocating.init(self.allocator);
        errdefer aw.deinit();
        const writer = &aw.writer;
        try writer.writeAll("{\"resourceSpans\":[{\"scopeSpans\":[{\"spans\":[");

        for (spans, 0..) |span, idx| {
            if (idx > 0) try writer.writeAll(",");

            try writer.writeAll("{");

            // Trace ID
            try writer.writeAll("\"traceId\":\"");
            const trace_id_hex = formatTraceId(span.trace_id);
            try writer.writeAll(&trace_id_hex);
            try writer.writeAll("\",");

            // Span ID
            try writer.writeAll("\"spanId\":\"");
            const span_id_hex = formatSpanId(span.span_id);
            try writer.writeAll(&span_id_hex);
            try writer.writeAll("\",");

            // Parent Span ID
            if (span.parent_span_id) |parent_id| {
                try writer.writeAll("\"parentSpanId\":\"");
                const parent_id_hex = formatSpanId(parent_id);
                try writer.writeAll(&parent_id_hex);
                try writer.writeAll("\",");
            }

            // Name
            try std.fmt.format(writer, "\"name\":\"{s}\",", .{span.name});

            // Kind
            try std.fmt.format(writer, "\"kind\":{d},", .{@intFromEnum(span.kind) + 1});

            // Timestamps (in nanoseconds)
            try std.fmt.format(writer, "\"startTimeUnixNano\":{d},", .{span.start_time * std.time.ns_per_s});
            try std.fmt.format(writer, "\"endTimeUnixNano\":{d},", .{span.end_time * std.time.ns_per_s});

            // Status
            try std.fmt.format(writer, "\"status\":{{\"code\":{d}}}", .{@intFromEnum(span.status)});

            try writer.writeAll("}");
        }

        try writer.writeAll("]}]}]}");
        const payload = try aw.toOwnedSlice();
        defer self.allocator.free(payload);

        // In a production implementation, this would send to the OTLP endpoint via HTTP.
        // For now, we log the export for observability.
        std.log.debug("SpanExporter: Exporting {d} spans ({d} bytes) to {s}", .{
            spans.len,
            payload.len,
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

test "trace id formatting" {
    const trace_id = [_]u8{ 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef };
    const formatted = formatTraceId(trace_id);
    try std.testing.expectEqualStrings("0123456789abcdef0123456789abcdef", &formatted);
}
