//! Distributed tracing with span propagation across async tasks and network boundaries.
//!
//! This module is split into submodules for maintainability:
//! - tracing_span.zig: Core span types and lifecycle management
//! - tracing_context.zig: Trace context propagation and sampling
//! - tracing_export.zig: Span processing and OTLP export

const std = @import("std");

// Import submodules
pub const tracing_span = @import("tracing_span.zig");
pub const tracing_context = @import("tracing_context.zig");
pub const tracing_export = @import("tracing_export.zig");

// Re-export core types from tracing_span
pub const TraceId = tracing_span.TraceId;
pub const SpanId = tracing_span.SpanId;
pub const SpanKind = tracing_span.SpanKind;
pub const SpanStatus = tracing_span.SpanStatus;
pub const AttributeValue = tracing_span.AttributeValue;
pub const SpanAttribute = tracing_span.SpanAttribute;
pub const SpanEvent = tracing_span.SpanEvent;
pub const SpanLink = tracing_span.SpanLink;
pub const Span = tracing_span.Span;

// Re-export context types from tracing_context
pub const Tracer = tracing_context.Tracer;
pub const TraceContext = tracing_context.TraceContext;
pub const PropagationFormat = tracing_context.PropagationFormat;
pub const Propagator = tracing_context.Propagator;
pub const TraceSampler = tracing_context.TraceSampler;

// Re-export export types from tracing_export
pub const formatTraceId = tracing_export.formatTraceId;
pub const formatSpanId = tracing_export.formatSpanId;
pub const SpanProcessor = tracing_export.SpanProcessor;
pub const SpanExporter = tracing_export.SpanExporter;

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
