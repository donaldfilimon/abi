//! OpenTelemetry integration for distributed tracing and metrics.
const std = @import("std");
const observability = @import("../../shared/observability/mod.zig");

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
            std.time.sleep(std.time.ns_per_ms * self.config.export_interval_ms);
            if (!self.running.load(.acquire)) break;
        }
    }

    pub fn exportMetrics(self: *OtelExporter, metrics: []const OtelMetric) !void {
        _ = self;
        _ = metrics;
    }

    pub fn exportTraces(self: *OtelExporter, traces: []const OtelSpan) !void {
        _ = self;
        _ = traces;
    }
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
    trace_id: [16]u8,
    span_id: [8]u8,
    parent_span_id: [8]u8,
    name: []const u8,
    kind: OtelSpanKind,
    start_time: i64,
    end_time: i64,
    attributes: []const OtelAttribute,
    events: []const OtelEvent,
    status: OtelStatus,
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
    error,
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
        const trace_id = self.generateTraceId();
        const span_id = self.generateSpanId();

        var parent_tid: [8]u8 = undefined;
        var parent_sid: [8]u8 = undefined;

        if (parent_trace_id) |tid| {
            @memcpy(parent_sid[0..8], tid[0..8]);
        }
        if (parent_span_id) |sid| {
            parent_sid = sid;
        }

        return .{
            .trace_id = trace_id,
            .span_id = span_id,
            .parent_span_id = parent_sid,
            .name = name,
            .kind = .internal,
            .start_time = std.time.timestamp(),
            .end_time = 0,
            .attributes = &.{},
            .events = &.{},
            .status = .unset,
        };
    }

    pub fn endSpan(self: *OtelTracer, span: *OtelSpan) void {
        span.end_time = std.time.timestamp();
    }

    pub fn addEvent(self: *OtelTracer, span: *OtelSpan, name: []const u8) !void {
        _ = self;
        _ = span;
        _ = name;
    }

    pub fn setAttribute(self: *OtelTracer, span: *OtelSpan, key: []const u8, value: OtelAttributeValue) !void {
        _ = self;
        _ = span;
        _ = key;
        _ = value;
    }

    fn generateTraceId(self: *OtelTracer) [16]u8 {
        var trace_id: [16]u8 = undefined;
        const counter = self.trace_id_counter.fetchAdd(1, .monotonic);
        @memset(&trace_id, 0);
        std.mem.writeInt(u64, trace_id[0..8], counter, .big);
        std.mem.writeInt(u64, trace_id[8..16], std.time.timestamp(), .big);
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

    pub fn extract(_: []const u8) OtelContext {
        return .{
            .trace_id = null,
            .span_id = null,
            .is_remote = false,
        };
    }

    pub fn inject(_: OtelContext, _: []const u8) void {}
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

    const span = try tracer.startSpan("test-span", null, null);
    try std.testing.expect(span.trace_id.len == 16);
    try std.testing.expect(span.span_id.len == 8);
}

test "otel span lifecycle" {
    const allocator = std.testing.allocator;
    var tracer = try OtelTracer.init(allocator, "test-service");
    defer tracer.deinit();

    var span = try tracer.startSpan("test-operation", null, null);
    try std.testing.expectEqual(OtelStatus.unset, span.status);

    tracer.endSpan(&span);
    try std.testing.expect(span.end_time > span.start_time);
}

test "trace id formatting" {
    const trace_id = [_]u8{0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef};
    const formatted = formatTraceId(trace_id);
    try std.testing.expectEqualStrings("0123456789abcdef0123456789abcdef", &formatted);
}
