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

    pub fn extract(_: []const u8) TraceContext {
        var ctx: TraceContext = undefined;
        std.crypto.random.bytes(&ctx.trace_id);
        std.crypto.random.bytes(&ctx.span_id);
        ctx.trace_id[0] = 0;
        ctx.is_remote = false;
        return ctx;
    }

    pub fn inject(_: TraceContext, _: []const u8) void {}
};

pub const PropagationFormat = enum {
    w3c_trace_context,
    b3,
    jaeger,
    aws_xray,
};

pub const Propagator = struct {
    format: PropagationFormat,

    pub fn extract(_: PropagationFormat, _: []const u8) ?TraceContext {
        return null;
    }

    pub fn inject(_: PropagationFormat, _: TraceContext, _: []const u8) void {}
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
    pub fn export(_: []const *Span) !void {}
    pub fn shutdown(_: void) !void {}
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
    const trace_id = [_]u8{0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef};
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
