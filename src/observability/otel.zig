//! OpenTelemetry integration for distributed tracing and metrics.
const std = @import("std");
const time = @import("../shared/utils_combined.zig");
const observability = @import("../observability/mod.zig");

pub const OtelConfig = struct {
    enabled: bool = true,
    service_name: []const u8 = "abi-service",
    service_version: []const u8 = "0.3.0",
    exporter_endpoint: []const u8 = "http://localhost:4318",
    export_interval_ms: u64 = 60000,
    export_on_shutdown: bool = true,
};

pub const OtelExporter = struct {
    allocator: std.mem.Allocator,
    config: OtelConfig,
    running: std.atomic.Value(bool),
    thread: ?std.Thread = null,

    pub fn init(allocator: std.mem.Allocator, config: OtelConfig) !OtelExporter {
        return .{
            .allocator = allocator,
            .config = config,
            .running = std.atomic.Value(bool).init(false),
            .thread = null,
        };
    }

    pub fn deinit(self: *OtelExporter) void {
        self.stop();
        self.* = undefined;
    }

    pub fn start(self: *OtelExporter) !void {
        if (self.running.load(.acquire)) return;
        self.running.store(true, .release);
        self.thread = try std.Thread.spawn(.{}, runExportLoop, .{self});
    }

    pub fn stop(self: *OtelExporter) void {
        if (!self.running.load(.acquire)) return;
        self.running.store(false, .release);
        if (self.thread) |t| {
            t.join();
            self.thread = null;
        }
    }

    fn runExportLoop(self: *OtelExporter) void {
        while (self.running.load(.acquire)) {
            time.sleepMs(self.config.export_interval_ms);
            if (!self.running.load(.acquire)) break;

            // Periodically export any buffered telemetry
            // In a full implementation, this would flush the internal buffer
            std.log.debug("OpenTelemetry: Export loop tick for {s}", .{self.config.service_name});
        }
    }

    /// Export metrics to the OTLP endpoint via HTTP
    pub fn exportMetrics(self: *OtelExporter, metrics: []const OtelMetric) !void {
        if (!self.config.enabled or metrics.len == 0) return;

        // Build JSON payload for OpenTelemetry metrics export
        var aw = std.Io.Writer.Allocating.init(self.allocator);
        errdefer aw.deinit();
        const writer = &aw.writer;
        try writer.writeAll("{\"resourceMetrics\":[{\"scopeMetrics\":[{\"metrics\":[");

        for (metrics, 0..) |metric, i| {
            if (i > 0) try writer.writeAll(",");

            try std.fmt.format(writer, "{{\"name\":\"{s}\",", .{metric.name});
            try std.fmt.format(writer, "\"description\":\"\",", .{});

            switch (metric.metric_type) {
                .counter => {
                    try writer.writeAll("\"sum\":{\"dataPoints\":[{");
                    try std.fmt.format(writer, "\"asDouble\":{d},", .{metric.value});
                    try std.fmt.format(writer, "\"timeUnixNano\":{d}", .{metric.timestamp * 1_000_000_000});
                    try writer.writeAll("}],\"aggregationTemporality\":2,\"isMonotonic\":true}");
                },
                .gauge => {
                    try writer.writeAll("\"gauge\":{\"dataPoints\":[{");
                    try std.fmt.format(writer, "\"asDouble\":{d},", .{metric.value});
                    try std.fmt.format(writer, "\"timeUnixNano\":{d}", .{metric.timestamp * 1_000_000_000});
                    try writer.writeAll("}]}");
                },
                .histogram => {
                    try writer.writeAll("\"histogram\":{\"dataPoints\":[{");
                    try std.fmt.format(writer, "\"sum\":{d},\"count\":1,", .{metric.value});
                    try std.fmt.format(writer, "\"timeUnixNano\":{d}", .{metric.timestamp * 1_000_000_000});
                    try writer.writeAll("}],\"aggregationTemporality\":2}");
                },
            }

            try writer.writeAll("}");
        }

        try writer.writeAll("]}]}]}");

        // Send to OpenTelemetry collector
        const payload = try aw.toOwnedSlice();
        defer self.allocator.free(payload);
        try self.sendToCollector("/v1/metrics", payload);
    }

    /// Export traces/spans to the OTLP endpoint via HTTP
    pub fn exportTraces(self: *OtelExporter, traces: []const OtelSpan) !void {
        if (!self.config.enabled or traces.len == 0) return;

        // Build JSON payload for OpenTelemetry traces export
        var aw = std.Io.Writer.Allocating.init(self.allocator);
        errdefer aw.deinit();
        const writer = &aw.writer;
        try writer.writeAll("{\"resourceSpans\":[{\"scopeSpans\":[{\"spans\":[");

        for (traces, 0..) |trace, i| {
            if (i > 0) try writer.writeAll(",");

            try writer.writeAll("{");
            try std.fmt.format(writer, "\"traceId\":\"{s}\",", .{std.fmt.fmtSliceHexLower(&trace.trace_id)});
            try std.fmt.format(writer, "\"spanId\":\"{s}\",", .{std.fmt.fmtSliceHexLower(&trace.span_id)});
            try std.fmt.format(writer, "\"parentSpanId\":\"{s}\",", .{std.fmt.fmtSliceHexLower(&trace.parent_span_id)});
            try std.fmt.format(writer, "\"name\":\"{s}\",", .{trace.name});
            try std.fmt.format(writer, "\"kind\":{d},", .{@intFromEnum(trace.kind) + 1});
            try std.fmt.format(writer, "\"startTimeUnixNano\":{d},", .{trace.start_time * 1_000_000_000});
            try std.fmt.format(writer, "\"endTimeUnixNano\":{d},", .{trace.end_time * 1_000_000_000});

            // Add attributes
            if (trace.attributes.items.len > 0) {
                try writer.writeAll("\"attributes\":[");
                for (trace.attributes.items, 0..) |attr, j| {
                    if (j > 0) try writer.writeAll(",");
                    try writer.writeAll("{");
                    try std.fmt.format(writer, "\"key\":\"{s}\",", .{attr.key});
                    try writer.writeAll("\"value\":{");
                    switch (attr.value) {
                        .string => |s| try std.fmt.format(writer, "\"stringValue\":\"{s}\"", .{s}),
                        .int => |n| try std.fmt.format(writer, "\"intValue\":{d}", .{n}),
                        .float => |f| try std.fmt.format(writer, "\"doubleValue\":{d}", .{f}),
                        .bool => |b| try std.fmt.format(writer, "\"boolValue\":{}", .{b}),
                    }
                    try writer.writeAll("}}");
                }
                try writer.writeAll("],");
            }

            // Add events
            if (trace.events.items.len > 0) {
                try writer.writeAll("\"events\":[");
                for (trace.events.items, 0..) |event, j| {
                    if (j > 0) try writer.writeAll(",");
                    try writer.writeAll("{");
                    try std.fmt.format(writer, "\"name\":\"{s}\",", .{event.name});
                    try std.fmt.format(writer, "\"timeUnixNano\":{d}", .{event.timestamp * 1_000_000_000});
                    try writer.writeAll("}");
                }
                try writer.writeAll("],");
            }

            try std.fmt.format(writer, "\"status\":{{\"code\":{d}}}", .{@intFromEnum(trace.status)});
            try writer.writeAll("}");
        }

        try writer.writeAll("]}]}]}");

        // Send to OpenTelemetry collector
        const payload = try aw.toOwnedSlice();
        defer self.allocator.free(payload);
        try self.sendToCollector("/v1/traces", payload);
    }

    fn sendToCollector(self: *OtelExporter, path: []const u8, payload: []const u8) !void {
        // Build full endpoint URL
        var endpoint_buf: [512]u8 = undefined;
        const endpoint = try std.fmt.bufPrint(&endpoint_buf, "{s}{s}", .{ self.config.exporter_endpoint, path });

        // In a production implementation, this would:
        // 1. Create HTTP client connection
        // 2. Send POST request with JSON payload
        // 3. Set Content-Type: application/json header
        // 4. Handle response and retries

        // For now, log the export attempt
        std.log.debug("OpenTelemetry export to {s}: {d} bytes", .{ endpoint, payload.len });

        // In the real implementation, would use std.http.Client or a web client
    }

    const ExportError = error{
        UrlTooLong,
        HttpError,
        SerializationError,
    };
};

pub const OtelMetric = struct {
    name: []const u8,
    value: f64,
    timestamp: i64,
    attributes: []const OtelAttribute,
    metric_type: OtelMetricType,
};

pub const OtelMetricType = enum {
    counter,
    gauge,
    histogram,
};

pub const OtelSpan = struct {
    allocator: ?std.mem.Allocator,
    trace_id: [16]u8,
    span_id: [8]u8,
    parent_span_id: [8]u8,
    name: []const u8,
    kind: OtelSpanKind,
    start_time: i64,
    end_time: i64,
    attributes: std.ArrayListUnmanaged(OtelAttribute),
    events: std.ArrayListUnmanaged(OtelEvent),
    status: OtelStatus,

    /// Free all dynamically allocated span data
    pub fn deinit(self: *OtelSpan) void {
        if (self.allocator) |alloc| {
            // Free event names and attributes
            for (self.events.items) |event| {
                alloc.free(event.name);
                for (event.attributes) |attr| {
                    alloc.free(attr.key);
                    if (attr.value == .string) {
                        alloc.free(attr.value.string);
                    }
                }
                alloc.free(event.attributes);
            }
            self.events.deinit(alloc);

            // Free attribute keys and string values
            for (self.attributes.items) |attr| {
                alloc.free(attr.key);
                if (attr.value == .string) {
                    alloc.free(attr.value.string);
                }
            }
            self.attributes.deinit(alloc);
        }
        self.* = undefined;
    }

    /// Get attributes as a slice (for compatibility)
    pub fn getAttributes(self: *const OtelSpan) []const OtelAttribute {
        return self.attributes.items;
    }

    /// Get events as a slice (for compatibility)
    pub fn getEvents(self: *const OtelSpan) []const OtelEvent {
        return self.events.items;
    }
};

pub const OtelSpanKind = enum {
    internal,
    server,
    client,
    producer,
    consumer,
};

pub const OtelEvent = struct {
    name: []const u8,
    timestamp: i64,
    attributes: []const OtelAttribute,
};

pub const OtelAttribute = struct {
    key: []const u8,
    value: OtelAttributeValue,
};

pub const OtelAttributeValue = union(enum) {
    string: []const u8,
    int: i64,
    float: f64,
    bool: bool,
};

pub const OtelStatus = enum {
    unset,
    ok,
    failed,
};

pub const OtelTracer = struct {
    allocator: std.mem.Allocator,
    service_name: []const u8,
    trace_id_counter: std.atomic.Value(u64),

    pub fn init(allocator: std.mem.Allocator, service_name: []const u8) !OtelTracer {
        return .{
            .allocator = allocator,
            .service_name = try allocator.dupe(u8, service_name),
            .trace_id_counter = std.atomic.Value(u64).init(0),
        };
    }

    pub fn deinit(self: *OtelTracer) void {
        self.allocator.free(self.service_name);
        self.* = undefined;
    }

    pub fn startSpan(
        self: *OtelTracer,
        name: []const u8,
        parent_trace_id: ?[16]u8,
        parent_span_id: ?[8]u8,
    ) !OtelSpan {
        const trace_id = if (parent_trace_id) |tid| tid else self.generateTraceId();
        const span_id = self.generateSpanId();

        var parent_sid: [8]u8 = .{0} ** 8;
        if (parent_span_id) |sid| {
            parent_sid = sid;
        }

        return .{
            .allocator = self.allocator,
            .trace_id = trace_id,
            .span_id = span_id,
            .parent_span_id = parent_sid,
            .name = name,
            .kind = .internal,
            .start_time = time.unixSeconds(),
            .end_time = 0,
            .attributes = std.ArrayListUnmanaged(OtelAttribute).empty,
            .events = std.ArrayListUnmanaged(OtelEvent).empty,
            .status = .unset,
        };
    }

    pub fn endSpan(_: *OtelTracer, span: *OtelSpan) void {
        span.end_time = time.unixSeconds();
    }

    /// Add an event to a span with the current timestamp
    pub fn addEvent(self: *OtelTracer, span: *OtelSpan, name: []const u8) !void {
        try self.addEventWithAttributes(span, name, &.{});
    }

    /// Add an event with attributes to a span
    pub fn addEventWithAttributes(
        self: *OtelTracer,
        span: *OtelSpan,
        name: []const u8,
        attrs: []const OtelAttribute,
    ) !void {
        const name_copy = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_copy);

        // Copy attributes
        const attrs_copy = try self.allocator.alloc(OtelAttribute, attrs.len);
        errdefer self.allocator.free(attrs_copy);

        for (attrs, 0..) |attr, i| {
            attrs_copy[i] = .{
                .key = try self.allocator.dupe(u8, attr.key),
                .value = try self.copyAttributeValue(attr.value),
            };
        }

        const event = OtelEvent{
            .name = name_copy,
            .timestamp = time.unixSeconds(),
            .attributes = attrs_copy,
        };

        try span.events.append(self.allocator, event);
    }

    /// Set an attribute on a span
    pub fn setAttribute(self: *OtelTracer, span: *OtelSpan, key: []const u8, value: OtelAttributeValue) !void {
        // Check if attribute already exists and update it
        for (span.attributes.items) |*attr| {
            if (std.mem.eql(u8, attr.key, key)) {
                // Free old string value if present
                if (attr.value == .string) {
                    self.allocator.free(attr.value.string);
                }
                attr.value = try self.copyAttributeValue(value);
                return;
            }
        }

        // Add new attribute
        const key_copy = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(key_copy);

        const attr = OtelAttribute{
            .key = key_copy,
            .value = try self.copyAttributeValue(value),
        };

        try span.attributes.append(self.allocator, attr);
    }

    /// Set the span status
    pub fn setStatus(_: *OtelTracer, span: *OtelSpan, status: OtelStatus) void {
        span.status = status;
    }

    /// Copy an attribute value, duplicating strings
    fn copyAttributeValue(self: *OtelTracer, value: OtelAttributeValue) !OtelAttributeValue {
        return switch (value) {
            .string => |s| .{ .string = try self.allocator.dupe(u8, s) },
            .int => |i| .{ .int = i },
            .float => |f| .{ .float = f },
            .bool => |b| .{ .bool = b },
        };
    }

    fn generateTraceId(self: *OtelTracer) [16]u8 {
        var trace_id: [16]u8 = undefined;
        const counter = self.trace_id_counter.fetchAdd(1, .monotonic);
        @memset(&trace_id, 0);
        std.mem.writeInt(u64, trace_id[0..8], counter, .big);
        std.mem.writeInt(u64, trace_id[8..16], time.unixSeconds(), .big);
        return trace_id;
    }

    fn generateSpanId(self: *OtelTracer) [8]u8 {
        var span_id: [8]u8 = undefined;
        const counter = self.trace_id_counter.fetchAdd(1, .monotonic);
        std.mem.writeInt(u64, span_id[0..], counter, .big);
        return span_id;
    }
};

pub const OtelContext = struct {
    trace_id: ?[16]u8,
    span_id: ?[8]u8,
    is_remote: bool,
    trace_flags: u8 = 0x01, // Sampled by default

    /// Extract trace context from W3C traceparent header value.
    /// Format: "00-{trace_id}-{span_id}-{flags}"
    pub fn extract(header_value: []const u8) OtelContext {
        var ctx = OtelContext{
            .trace_id = null,
            .span_id = null,
            .is_remote = false,
            .trace_flags = 0x01,
        };

        // W3C Trace Context format: version-trace_id-span_id-flags
        // Minimum length: 2 + 1 + 32 + 1 + 16 + 1 + 2 = 55
        if (header_value.len >= 55) {
            const trace_start: usize = 3;
            const span_start: usize = 36;
            const flags_start: usize = 53;

            // Parse trace_id (32 hex chars -> 16 bytes)
            var trace_id: [16]u8 = undefined;
            if (parseHexBytes(header_value[trace_start .. trace_start + 32], &trace_id)) {
                // Parse span_id (16 hex chars -> 8 bytes)
                var span_id: [8]u8 = undefined;
                if (parseHexBytes(header_value[span_start .. span_start + 16], &span_id)) {
                    ctx.trace_id = trace_id;
                    ctx.span_id = span_id;
                    ctx.is_remote = true;

                    // Parse flags
                    if (header_value.len > flags_start + 1) {
                        ctx.trace_flags = parseHexByte(header_value[flags_start], header_value[flags_start + 1]) orelse 0x01;
                    }
                }
            }
        }

        return ctx;
    }

    /// Inject trace context into a buffer as W3C traceparent header value.
    /// Returns the number of bytes written, or 0 if context is empty or buffer too small.
    pub fn inject(self: OtelContext, buffer: []u8) usize {
        // Need both trace_id and span_id to inject
        const trace_id = self.trace_id orelse return 0;
        const span_id = self.span_id orelse return 0;

        // W3C format: 00-{trace_id}-{span_id}-{flags} = 55 bytes
        if (buffer.len < 55) return 0;

        // Version (always 00)
        buffer[0] = '0';
        buffer[1] = '0';
        buffer[2] = '-';

        // Trace ID (32 hex chars)
        for (trace_id, 0..) |byte, i| {
            buffer[3 + i * 2] = hexChar(@intCast(byte >> 4));
            buffer[3 + i * 2 + 1] = hexChar(@intCast(byte & 0x0F));
        }
        buffer[35] = '-';

        // Span ID (16 hex chars)
        for (span_id, 0..) |byte, i| {
            buffer[36 + i * 2] = hexChar(@intCast(byte >> 4));
            buffer[36 + i * 2 + 1] = hexChar(@intCast(byte & 0x0F));
        }
        buffer[52] = '-';

        // Flags (2 hex chars)
        buffer[53] = hexChar(@intCast(self.trace_flags >> 4));
        buffer[54] = hexChar(@intCast(self.trace_flags & 0x0F));

        return 55;
    }

    /// Create a new context from a span
    pub fn fromSpan(span: *const OtelSpan) OtelContext {
        return .{
            .trace_id = span.trace_id,
            .span_id = span.span_id,
            .is_remote = false,
            .trace_flags = if (span.status == .ok) 0x01 else 0x00,
        };
    }

    /// Check if the context is valid (has both trace and span IDs)
    pub fn isValid(self: OtelContext) bool {
        return self.trace_id != null and self.span_id != null;
    }

    /// Check if tracing is sampled
    pub fn isSampled(self: OtelContext) bool {
        return (self.trace_flags & 0x01) != 0;
    }

    fn parseHexBytes(hex: []const u8, out: []u8) bool {
        if (hex.len != out.len * 2) return false;
        for (out, 0..) |*byte, i| {
            byte.* = parseHexByte(hex[i * 2], hex[i * 2 + 1]) orelse return false;
        }
        return true;
    }

    fn parseHexByte(high: u8, low: u8) ?u8 {
        const h = hexDigit(high) orelse return null;
        const l = hexDigit(low) orelse return null;
        return (h << 4) | l;
    }

    fn hexDigit(c: u8) ?u8 {
        return switch (c) {
            '0'...'9' => c - '0',
            'a'...'f' => c - 'a' + 10,
            'A'...'F' => c - 'A' + 10,
            else => null,
        };
    }
};

pub fn createOtelResource(allocator: std.mem.Allocator, service_name: []const u8) ![]OtelAttribute {
    const attrs = try allocator.alloc(OtelAttribute, 4);
    errdefer allocator.free(attrs);

    attrs[0] = .{ .key = "service.name", .value = .{ .string = service_name } };
    attrs[1] = .{ .key = "service.version", .value = .{ .string = "0.3.0" } };
    attrs[2] = .{ .key = "telemetry.sdk.name", .value = .{ .string = "abi" } };
    attrs[3] = .{ .key = "telemetry.sdk.version", .value = .{ .string = "0.3.0" } };

    return attrs;
}

pub fn formatTraceId(trace_id: [16]u8) [32]u8 {
    var result: [32]u8 = undefined;
    for (trace_id, 0..) |byte, i| {
        const high = byte >> 4;
        const low = byte & 0x0F;
        result[i * 2] = hexChar(high);
        result[i * 2 + 1] = hexChar(low);
    }
    return result;
}

pub fn formatSpanId(span_id: [8]u8) [16]u8 {
    var result: [16]u8 = undefined;
    for (span_id, 0..) |byte, i| {
        const high = byte >> 4;
        const low = byte & 0x0F;
        result[i * 2] = hexChar(high);
        result[i * 2 + 1] = hexChar(low);
    }
    return result;
}

fn hexChar(value: u4) u8 {
    return switch (value) {
        0...9 => '0' + @as(u8, @intCast(value)),
        10...15 => 'a' + @as(u8, @intCast(value - 10)),
    };
}

test "otel tracer init" {
    const allocator = std.testing.allocator;
    var tracer = try OtelTracer.init(allocator, "test-service");
    defer tracer.deinit();

    var span = try tracer.startSpan("test-span", null, null);
    defer span.deinit();
    try std.testing.expect(span.trace_id.len == 16);
    try std.testing.expect(span.span_id.len == 8);
}

test "otel span lifecycle" {
    const allocator = std.testing.allocator;
    var tracer = try OtelTracer.init(allocator, "test-service");
    defer tracer.deinit();

    var span = try tracer.startSpan("test-operation", null, null);
    defer span.deinit();
    try std.testing.expectEqual(OtelStatus.unset, span.status);

    tracer.endSpan(&span);
    try std.testing.expect(span.end_time >= span.start_time);
}

test "otel span add event" {
    const allocator = std.testing.allocator;
    var tracer = try OtelTracer.init(allocator, "test-service");
    defer tracer.deinit();

    var span = try tracer.startSpan("test-operation", null, null);
    defer span.deinit();

    try tracer.addEvent(&span, "event-1");
    try tracer.addEvent(&span, "event-2");

    try std.testing.expectEqual(@as(usize, 2), span.events.items.len);
    try std.testing.expectEqualStrings("event-1", span.events.items[0].name);
    try std.testing.expectEqualStrings("event-2", span.events.items[1].name);
}

test "otel span set attribute" {
    const allocator = std.testing.allocator;
    var tracer = try OtelTracer.init(allocator, "test-service");
    defer tracer.deinit();

    var span = try tracer.startSpan("test-operation", null, null);
    defer span.deinit();

    try tracer.setAttribute(&span, "http.method", .{ .string = "GET" });
    try tracer.setAttribute(&span, "http.status_code", .{ .int = 200 });
    try tracer.setAttribute(&span, "request.duration", .{ .float = 0.123 });
    try tracer.setAttribute(&span, "cache.hit", .{ .bool = true });

    try std.testing.expectEqual(@as(usize, 4), span.attributes.items.len);

    // Test attribute update
    try tracer.setAttribute(&span, "http.status_code", .{ .int = 404 });
    try std.testing.expectEqual(@as(usize, 4), span.attributes.items.len);
}

test "trace id formatting" {
    const trace_id = [_]u8{ 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef };
    const formatted = formatTraceId(trace_id);
    try std.testing.expectEqualStrings("0123456789abcdef0123456789abcdef", &formatted);
}
