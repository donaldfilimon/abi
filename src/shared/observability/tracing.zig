//! Distributed tracing with span propagation across async tasks and network boundaries.
const std = @import("std");

pub const TraceId = [16]u8;
pub const SpanId = [8]u8;

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
            .start_time = std.time.timestamp(),
            .attributes = attrs,
            .events = events,
            .links = links,
            .allocator = allocator,
        };
    }

    pub fn end(self: *Span) void {
        self.end_time = std.time.timestamp();
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
            .timestamp = std.time.timestamp(),
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

    pub fn setStatus(self: *Span, status: SpanStatus, message: ?[]const u8) void {
        self.status = status;
        self.error_message = message;
    }

    pub fn recordError(self: *Span, message: []const u8) !void {
        self.status = .error_status;
        const msg_copy = try self.allocator.dupe(u8, message);
        self.error_message = msg_copy;
        try self.addEvent("exception");
    }

    pub fn getDuration(self: Span) i64 {
        return self.end_time - self.start_time;
    }

    fn generateTraceId() TraceId {
        var trace_id: TraceId = undefined;
        std.crypto.random.bytes(&trace_id);
        trace_id[0] = 0;
        return trace_id;
    }

    fn generateSpanId() SpanId {
        var span_id: SpanId = undefined;
        std.crypto.random.bytes(&span_id);
        return span_id;
    }
};

pub const Tracer = struct {
    allocator: std.mem.Allocator,
    service_name: []const u8,
    tracer_version: []const u8,
    schema_url: []const u8,

    pub fn init(allocator: std.mem.Allocator, service_name: []const u8) !Tracer {
        return .{
            .allocator = allocator,
            .service_name = try allocator.dupe(u8, service_name),
            .tracer_version = try allocator.dupe(u8, "0.3.0"),
            .schema_url = try allocator.dupe(u8, "https://opentelemetry.io/schemas/1.11.0"),
        };
    }

    pub fn deinit(self: *Tracer) void {
        self.allocator.free(self.service_name);
        self.allocator.free(self.tracer_version);
        self.allocator.free(self.schema_url);
        self.* = undefined;
    }

    pub fn startSpan(
        self: *Tracer,
        name: []const u8,
        parent_context: ?*const TraceContext,
        kind: SpanKind,
    ) !Span {
        const trace_id = if (parent_context) |ctx| ctx.trace_id else null;
        const parent_span_id = if (parent_context) |ctx| ctx.span_id else null;
        return Span.start(self.allocator, name, trace_id, parent_span_id, kind);
    }
};

pub const TraceContext = struct {
    trace_id: TraceId,
    span_id: SpanId,
    is_remote: bool,
    trace_flags: u8 = 0x01, // sampled by default

    /// Extract trace context from W3C traceparent header value.
    /// Format: "00-{trace_id}-{span_id}-{flags}"
    pub fn extract(header_value: []const u8) TraceContext {
        var ctx: TraceContext = undefined;
        ctx.trace_flags = 0x01;
        ctx.is_remote = false;

        // W3C Trace Context format: version-trace_id-span_id-flags
        // Example: 00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01
        if (header_value.len >= 55) { // Minimum valid length
            // Skip version (2 chars) and dash (1 char)
            const trace_start: usize = 3;
            const span_start: usize = 36; // 3 + 32 + 1
            const flags_start: usize = 53; // 36 + 16 + 1

            // Parse trace_id (32 hex chars -> 16 bytes)
            if (parseHexBytes(header_value[trace_start .. trace_start + 32], &ctx.trace_id)) {
                // Parse span_id (16 hex chars -> 8 bytes)
                if (parseHexBytes(header_value[span_start .. span_start + 16], &ctx.span_id)) {
                    // Parse flags (2 hex chars -> 1 byte)
                    if (header_value.len > flags_start + 1) {
                        ctx.trace_flags = parseHexByte(header_value[flags_start], header_value[flags_start + 1]) orelse 0x01;
                    }
                    ctx.is_remote = true;
                    return ctx;
                }
            }
        }

        // Fall back to generating new context
        std.crypto.random.bytes(&ctx.trace_id);
        std.crypto.random.bytes(&ctx.span_id);
        ctx.trace_id[0] = 0;
        return ctx;
    }

    /// Inject trace context into a buffer as W3C traceparent header value.
    /// Returns the number of bytes written.
    pub fn inject(self: TraceContext, buffer: []u8) usize {
        // W3C Trace Context format: version-trace_id-span_id-flags
        // Total length: 2 + 1 + 32 + 1 + 16 + 1 + 2 = 55 bytes
        if (buffer.len < 55) return 0;

        // Version (always 00)
        buffer[0] = '0';
        buffer[1] = '0';
        buffer[2] = '-';

        // Trace ID (32 hex chars)
        for (self.trace_id, 0..) |byte, i| {
            buffer[3 + i * 2] = hexChar(byte >> 4);
            buffer[3 + i * 2 + 1] = hexChar(byte & 0x0F);
        }
        buffer[35] = '-';

        // Span ID (16 hex chars)
        for (self.span_id, 0..) |byte, i| {
            buffer[36 + i * 2] = hexChar(byte >> 4);
            buffer[36 + i * 2 + 1] = hexChar(byte & 0x0F);
        }
        buffer[52] = '-';

        // Flags (2 hex chars)
        buffer[53] = hexChar(self.trace_flags >> 4);
        buffer[54] = hexChar(self.trace_flags & 0x0F);

        return 55;
    }

    /// Create a child context from this context
    pub fn createChild(self: TraceContext) TraceContext {
        var child = self;
        std.crypto.random.bytes(&child.span_id);
        child.is_remote = false;
        return child;
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

pub const PropagationFormat = enum {
    w3c_trace_context,
    b3,
    jaeger,
    aws_xray,
};

pub const Propagator = struct {
    format: PropagationFormat,

    pub fn init(format: PropagationFormat) Propagator {
        return .{ .format = format };
    }

    /// Extract trace context from header value based on propagation format
    pub fn extract(self: Propagator, header_value: []const u8) ?TraceContext {
        return switch (self.format) {
            .w3c_trace_context => extractW3C(header_value),
            .b3 => extractB3(header_value),
            .jaeger => extractJaeger(header_value),
            .aws_xray => extractAwsXray(header_value),
        };
    }

    /// Inject trace context into buffer based on propagation format.
    /// Returns number of bytes written.
    pub fn inject(self: Propagator, ctx: TraceContext, buffer: []u8) usize {
        return switch (self.format) {
            .w3c_trace_context => ctx.inject(buffer),
            .b3 => injectB3(ctx, buffer),
            .jaeger => injectJaeger(ctx, buffer),
            .aws_xray => injectAwsXray(ctx, buffer),
        };
    }

    /// W3C Trace Context extraction
    fn extractW3C(header_value: []const u8) ?TraceContext {
        const ctx = TraceContext.extract(header_value);
        if (ctx.is_remote) return ctx;
        return null;
    }

    /// B3 single header format: {trace_id}-{span_id}-{sampled}-{parent_span_id}
    fn extractB3(header_value: []const u8) ?TraceContext {
        if (header_value.len < 32) return null;

        var ctx: TraceContext = undefined;
        ctx.is_remote = true;
        ctx.trace_flags = 0x01;

        // B3 format can have 16 or 32 char trace IDs
        var pos: usize = 0;
        var dash_pos = std.mem.indexOf(u8, header_value, "-") orelse return null;

        // Parse trace_id (16 or 32 hex chars)
        const trace_id_len = dash_pos;
        if (trace_id_len == 32) {
            if (!TraceContext.parseHexBytes(header_value[0..32], &ctx.trace_id)) return null;
        } else if (trace_id_len == 16) {
            @memset(&ctx.trace_id, 0);
            if (!TraceContext.parseHexBytes(header_value[0..16], ctx.trace_id[8..16])) return null;
        } else {
            return null;
        }

        pos = dash_pos + 1;
        if (pos + 16 > header_value.len) return null;

        // Parse span_id (16 hex chars)
        if (!TraceContext.parseHexBytes(header_value[pos .. pos + 16], &ctx.span_id)) return null;

        pos += 16;
        if (pos < header_value.len and header_value[pos] == '-') {
            pos += 1;
            // Parse sampled flag
            if (pos < header_value.len) {
                ctx.trace_flags = if (header_value[pos] == '1' or header_value[pos] == 'd') 0x01 else 0x00;
            }
        }

        return ctx;
    }

    /// Jaeger format: {trace_id}:{span_id}:{parent_span_id}:{flags}
    fn extractJaeger(header_value: []const u8) ?TraceContext {
        if (header_value.len < 35) return null;

        var ctx: TraceContext = undefined;
        ctx.is_remote = true;

        var parts: [4][]const u8 = undefined;
        var part_count: usize = 0;
        var start: usize = 0;

        for (header_value, 0..) |c, i| {
            if (c == ':') {
                if (part_count < 4) {
                    parts[part_count] = header_value[start..i];
                    part_count += 1;
                }
                start = i + 1;
            }
        }
        if (part_count < 4 and start < header_value.len) {
            parts[part_count] = header_value[start..];
            part_count += 1;
        }

        if (part_count < 4) return null;

        // Parse trace_id
        if (parts[0].len == 32) {
            if (!TraceContext.parseHexBytes(parts[0], &ctx.trace_id)) return null;
        } else if (parts[0].len == 16) {
            @memset(&ctx.trace_id, 0);
            if (!TraceContext.parseHexBytes(parts[0], ctx.trace_id[8..16])) return null;
        } else {
            return null;
        }

        // Parse span_id
        if (parts[1].len != 16) return null;
        if (!TraceContext.parseHexBytes(parts[1], &ctx.span_id)) return null;

        // Parse flags
        if (parts[3].len >= 1) {
            ctx.trace_flags = TraceContext.parseHexByte(
                if (parts[3].len > 1) parts[3][0] else '0',
                parts[3][parts[3].len - 1],
            ) orelse 0x01;
        } else {
            ctx.trace_flags = 0x01;
        }

        return ctx;
    }

    /// AWS X-Ray format: Root={trace_id};Parent={span_id};Sampled={0|1}
    fn extractAwsXray(header_value: []const u8) ?TraceContext {
        var ctx: TraceContext = undefined;
        ctx.is_remote = true;
        ctx.trace_flags = 0x01;
        @memset(&ctx.trace_id, 0);
        @memset(&ctx.span_id, 0);

        var has_root = false;
        var has_parent = false;

        var start: usize = 0;
        var i: usize = 0;
        while (i <= header_value.len) : (i += 1) {
            const at_end = i == header_value.len;
            const c = if (at_end) ';' else header_value[i];

            if (c == ';' or at_end) {
                const part = header_value[start..i];
                if (std.mem.startsWith(u8, part, "Root=")) {
                    // X-Ray trace ID format: 1-xxxxxxxx-xxxxxxxxxxxxxxxxxxxxxxxx
                    const root = part[5..];
                    if (root.len >= 35 and root[0] == '1' and root[1] == '-' and root[10] == '-') {
                        // Parse epoch (8 hex) + random (24 hex) into trace_id
                        if (TraceContext.parseHexBytes(root[2..10], ctx.trace_id[0..4])) {
                            if (TraceContext.parseHexBytes(root[11..35], ctx.trace_id[4..16])) {
                                has_root = true;
                            }
                        }
                    }
                } else if (std.mem.startsWith(u8, part, "Parent=")) {
                    const parent = part[7..];
                    if (parent.len == 16) {
                        if (TraceContext.parseHexBytes(parent, &ctx.span_id)) {
                            has_parent = true;
                        }
                    }
                } else if (std.mem.startsWith(u8, part, "Sampled=")) {
                    if (part.len > 8) {
                        ctx.trace_flags = if (part[8] == '1') 0x01 else 0x00;
                    }
                }
                start = i + 1;
            }
        }

        if (has_root and has_parent) return ctx;
        return null;
    }

    /// Inject B3 single header format
    fn injectB3(ctx: TraceContext, buffer: []u8) usize {
        // Format: {trace_id}-{span_id}-{sampled}
        // Length: 32 + 1 + 16 + 1 + 1 = 51
        if (buffer.len < 51) return 0;

        // Trace ID
        for (ctx.trace_id, 0..) |byte, i| {
            buffer[i * 2] = hexChar(byte >> 4);
            buffer[i * 2 + 1] = hexChar(byte & 0x0F);
        }
        buffer[32] = '-';

        // Span ID
        for (ctx.span_id, 0..) |byte, i| {
            buffer[33 + i * 2] = hexChar(byte >> 4);
            buffer[33 + i * 2 + 1] = hexChar(byte & 0x0F);
        }
        buffer[49] = '-';

        // Sampled
        buffer[50] = if (ctx.trace_flags & 0x01 != 0) '1' else '0';

        return 51;
    }

    /// Inject Jaeger format
    fn injectJaeger(ctx: TraceContext, buffer: []u8) usize {
        // Format: {trace_id}:{span_id}:{parent_span_id}:{flags}
        // Length: 32 + 1 + 16 + 1 + 1 + 1 + 2 = 54
        if (buffer.len < 54) return 0;

        // Trace ID
        for (ctx.trace_id, 0..) |byte, i| {
            buffer[i * 2] = hexChar(byte >> 4);
            buffer[i * 2 + 1] = hexChar(byte & 0x0F);
        }
        buffer[32] = ':';

        // Span ID
        for (ctx.span_id, 0..) |byte, i| {
            buffer[33 + i * 2] = hexChar(byte >> 4);
            buffer[33 + i * 2 + 1] = hexChar(byte & 0x0F);
        }
        buffer[49] = ':';

        // Parent span ID (empty/zero)
        buffer[50] = '0';
        buffer[51] = ':';

        // Flags
        buffer[52] = hexChar(ctx.trace_flags >> 4);
        buffer[53] = hexChar(ctx.trace_flags & 0x0F);

        return 54;
    }

    /// Inject AWS X-Ray format
    fn injectAwsXray(ctx: TraceContext, buffer: []u8) usize {
        // Format: Root=1-xxxxxxxx-xxxxxxxxxxxxxxxxxxxxxxxx;Parent=xxxxxxxxxxxxxxxx;Sampled=x
        // Length: 5 + 35 + 8 + 16 + 9 + 1 = 74
        if (buffer.len < 74) return 0;

        var pos: usize = 0;

        // Root=1-
        @memcpy(buffer[pos .. pos + 7], "Root=1-");
        pos += 7;

        // Epoch (first 4 bytes as hex)
        for (ctx.trace_id[0..4], 0..) |byte, i| {
            buffer[pos + i * 2] = hexChar(byte >> 4);
            buffer[pos + i * 2 + 1] = hexChar(byte & 0x0F);
        }
        pos += 8;
        buffer[pos] = '-';
        pos += 1;

        // Random (remaining 12 bytes as hex)
        for (ctx.trace_id[4..16], 0..) |byte, i| {
            buffer[pos + i * 2] = hexChar(byte >> 4);
            buffer[pos + i * 2 + 1] = hexChar(byte & 0x0F);
        }
        pos += 24;

        // ;Parent=
        @memcpy(buffer[pos .. pos + 8], ";Parent=");
        pos += 8;

        // Span ID
        for (ctx.span_id, 0..) |byte, i| {
            buffer[pos + i * 2] = hexChar(byte >> 4);
            buffer[pos + i * 2 + 1] = hexChar(byte & 0x0F);
        }
        pos += 16;

        // ;Sampled=x
        @memcpy(buffer[pos .. pos + 9], ";Sampled=");
        pos += 9;
        buffer[pos] = if (ctx.trace_flags & 0x01 != 0) '1' else '0';
        pos += 1;

        return pos;
    }
};

pub const TraceSampler = struct {
    sampler_type: SamplerType,
    param: f64,
    trace_id_counter: std.atomic.Value(u64),

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
            .trace_id_counter = std.atomic.Value(u64).init(0),
        };
    }

    pub fn shouldSample(self: *TraceSampler, trace_id: TraceId) bool {
        return switch (self.sampler_type) {
            .always_on => true,
            .always_off => false,
            .trace_id_ratio => {
                const first_byte = trace_id[0];
                const threshold = @as(u8, @intFromFloat(self.param * 256.0));
                return first_byte < threshold;
            },
            .parent_based => false,
        };
    }
};

pub fn formatTraceId(trace_id: TraceId) [32]u8 {
    var result: [32]u8 = undefined;
    for (trace_id, 0..) |byte, i| {
        result[i * 2] = hexChar(byte >> 4);
        result[i * 2 + 1] = hexChar(byte & 0x0F);
    }
    return result;
}

pub fn formatSpanId(span_id: SpanId) [16]u8 {
    var result: [16]u8 = undefined;
    for (span_id, 0..) |byte, i| {
        result[i * 2] = hexChar(byte >> 4);
        result[i * 2 + 1] = hexChar(byte & 0x0F);
    }
    return result;
}

fn hexChar(value: u8) u8 {
    return switch (value & 0x0F) {
        0...9 => '0' + value,
        10...15 => 'a' + value - 10,
    };
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
        var json_buffer = std.ArrayList(u8).init(self.allocator);
        defer json_buffer.deinit();

        const writer = json_buffer.writer();
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

        // In a production implementation, this would send to the OTLP endpoint via HTTP.
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

test "span lifecycle" {
    const allocator = std.testing.allocator;
    var span = try Span.start(allocator, "test-operation", null, null, .internal);
    defer span.deinit();

    try span.setAttribute("key", .{ .string = "value" });
    try span.addEvent("event1");

    span.end();
    try std.testing.expect(span.end_time > span.start_time);
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
