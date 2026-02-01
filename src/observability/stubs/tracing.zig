const std = @import("std");
const types = @import("types.zig");

// Tracing types (API-compatible stubs)
pub const Tracer = struct {
    pub fn init(_: std.mem.Allocator, _: []const u8) types.Error!Tracer {
        return error.ObservabilityDisabled;
    }
    pub fn deinit(_: *Tracer) void {}
    pub fn startSpan(_: *Tracer, _: []const u8, _: ?*const TraceContext, _: types.SpanKind) types.Error!Span {
        return error.ObservabilityDisabled;
    }
};

pub const Span = struct {
    pub fn start(_: std.mem.Allocator, _: []const u8, _: ?types.TraceId, _: ?types.SpanId, _: types.SpanKind) types.Error!Span {
        return error.ObservabilityDisabled;
    }
    pub fn end(_: *Span) void {}
    pub fn deinit(_: *Span) void {}
    pub fn setAttribute(_: *Span, _: []const u8, _: types.AttributeValue) types.Error!void {
        return error.ObservabilityDisabled;
    }
    pub fn addEvent(_: *Span, _: []const u8) types.Error!void {
        return error.ObservabilityDisabled;
    }
};

pub const TraceContext = struct {
    trace_id: types.TraceId = [_]u8{0} ** 16,
    span_id: types.SpanId = [_]u8{0} ** 8,
    is_remote: bool = false,
    trace_flags: u8 = 0x01,

    pub fn extract(_: []const u8) TraceContext {
        return .{};
    }
    pub fn inject(_: TraceContext, _: []u8) usize {
        return 0;
    }
};

pub const Propagator = struct {
    pub fn init(_: types.PropagationFormat) Propagator {
        return .{};
    }
};

pub const TraceSampler = struct {
    pub fn init(_: types.SamplerType, _: f64) TraceSampler {
        return .{};
    }
    pub fn shouldSample(_: *TraceSampler, _: types.TraceId) bool {
        return false;
    }
};

pub const SpanProcessor = struct {
    pub fn init(_: std.mem.Allocator) SpanProcessor {
        return .{};
    }
    pub fn deinit(_: *SpanProcessor) void {}
};

pub const SpanExporter = struct {
    pub fn init(_: std.mem.Allocator) SpanExporter {
        return .{};
    }
    pub fn deinit(_: *SpanExporter) void {}
};

// OpenTelemetry (stubs)
pub const otel = struct {};
pub const OtelExporter = struct {};
pub const OtelTracer = struct {};
pub const OtelSpan = struct {};
pub const OtelContext = struct {};
pub const OtelMetric = struct {};
pub const OtelAttribute = struct {};
pub const OtelAttributeValue = types.AttributeValue;
pub const OtelEvent = struct {};

pub fn formatTraceId(_: [16]u8) [32]u8 {
    return [_]u8{0} ** 32;
}
pub fn formatSpanId(_: [8]u8) [16]u8 {
    return [_]u8{0} ** 16;
}
pub fn createOtelResource(_: std.mem.Allocator, _: []const u8) types.Error!void {
    return error.ObservabilityDisabled;
}
