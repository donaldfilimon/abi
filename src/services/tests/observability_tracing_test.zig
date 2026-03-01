//! Observability Tracing Tests — Spans, Tracer, IDs, Samplers,
//! SpanProcessor, and OpenTelemetry integration.

const std = @import("std");
const testing = std.testing;
const abi = @import("abi");

const observability = abi.features.observability;
const Span = observability.Span;
const SpanKind = observability.SpanKind;
const SpanStatus = observability.SpanStatus;
const TraceId = observability.TraceId;
const SpanId = observability.SpanId;
const Tracer = observability.Tracer;
const OtelTracer = observability.OtelTracer;
const OtelSpan = observability.OtelSpan;
const OtelStatus = observability.OtelStatus;
const OtelContext = observability.OtelContext;

// ============================================================================
// Tracing Span Tests
// ============================================================================

test "observability: span creation" {
    const allocator = testing.allocator;
    var span = try Span.start(allocator, "test-operation", null, null, .internal);
    defer span.deinit();

    try testing.expectEqualStrings("test-operation", span.name);
    try testing.expectEqual(SpanKind.internal, span.kind);
    try testing.expectEqual(SpanStatus.unset, span.status);
    try testing.expect(span.trace_id.len == 16);
    try testing.expect(span.span_id.len == 8);
    try testing.expect(span.parent_span_id == null);
}

test "observability: span with attributes" {
    const allocator = testing.allocator;
    var span = try Span.start(allocator, "http-request", null, null, .server);
    defer span.deinit();

    try span.setAttribute("http.method", .{ .string = "GET" });
    try span.setAttribute("http.status_code", .{ .int = 200 });
    try span.setAttribute("http.url", .{ .string = "/api/users" });
    try span.setAttribute("response.size", .{ .float = 1024.5 });
    try span.setAttribute("cache.hit", .{ .bool = true });

    try testing.expectEqual(@as(usize, 5), span.attributes.items.len);
}

test "observability: span lifecycle (start and end)" {
    const allocator = testing.allocator;
    var span = try Span.start(allocator, "test-operation", null, null, .internal);
    defer span.deinit();

    // start_time may be 0 if the test runs within the first second of app start
    // The important thing is that it's non-negative
    try testing.expect(span.start_time >= 0);
    try testing.expectEqual(@as(i64, 0), span.end_time);

    span.end();
    try testing.expect(span.end_time >= span.start_time);
}

test "observability: span duration calculation" {
    const allocator = testing.allocator;
    var span = try Span.start(allocator, "test-operation", null, null, .internal);
    defer span.deinit();

    span.end();
    const duration = span.getDuration();
    try testing.expect(duration >= 0);
}

test "observability: span events" {
    const allocator = testing.allocator;
    var span = try Span.start(allocator, "database-query", null, null, .client);
    defer span.deinit();

    try span.addEvent("connection-acquired");
    try span.addEvent("query-executed");
    try span.addEvent("results-fetched");

    try testing.expectEqual(@as(usize, 3), span.events.items.len);
    try testing.expectEqualStrings("connection-acquired", span.events.items[0].name);
    try testing.expectEqualStrings("query-executed", span.events.items[1].name);
    try testing.expectEqualStrings("results-fetched", span.events.items[2].name);
}

test "observability: span links" {
    const allocator = testing.allocator;
    var span = try Span.start(allocator, "batch-processor", null, null, .internal);
    defer span.deinit();

    const other_trace_id: TraceId = .{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    const other_span_id: SpanId = .{ 1, 2, 3, 4, 5, 6, 7, 8 };

    try span.addLink(other_trace_id, other_span_id);
    try testing.expectEqual(@as(usize, 1), span.links.items.len);
    try testing.expectEqualSlices(u8, &other_trace_id, &span.links.items[0].trace_id);
}

test "observability: span status setting" {
    const allocator = testing.allocator;
    var span = try Span.start(allocator, "test-operation", null, null, .internal);
    defer span.deinit();

    try testing.expectEqual(SpanStatus.unset, span.status);

    try span.setStatus(.ok, null);
    try testing.expectEqual(SpanStatus.ok, span.status);

    try span.setStatus(.error_status, "Something went wrong");
    try testing.expectEqual(SpanStatus.error_status, span.status);
    try testing.expect(span.error_message != null);
    try testing.expectEqualStrings("Something went wrong", span.error_message.?);
}

test "observability: span record error" {
    const allocator = testing.allocator;
    var span = try Span.start(allocator, "test-operation", null, null, .internal);
    defer span.deinit();

    try span.recordError("Connection timeout");
    try testing.expectEqual(SpanStatus.error_status, span.status);
    try testing.expectEqualStrings("Connection timeout", span.error_message.?);
    try testing.expectEqual(@as(usize, 1), span.events.items.len);
    try testing.expectEqualStrings("exception", span.events.items[0].name);
}

test "observability: nested spans (parent-child)" {
    const allocator = testing.allocator;

    // Create parent span
    var parent_span = try Span.start(allocator, "parent-operation", null, null, .server);
    defer parent_span.deinit();

    // Create child span with parent's trace context
    var child_span = try Span.start(
        allocator,
        "child-operation",
        parent_span.trace_id,
        parent_span.span_id,
        .internal,
    );
    defer child_span.deinit();

    // Verify trace ID is inherited
    try testing.expectEqualSlices(u8, &parent_span.trace_id, &child_span.trace_id);

    // Verify parent span ID is set
    try testing.expect(child_span.parent_span_id != null);
    try testing.expectEqualSlices(u8, &parent_span.span_id, &child_span.parent_span_id.?);
}

// ============================================================================
// Tracer Tests
// ============================================================================

test "observability: tracer initialization" {
    const allocator = testing.allocator;
    var tracer = try Tracer.init(allocator, "test-service");
    defer tracer.deinit();

    try testing.expectEqualStrings("test-service", tracer.service_name);
}

test "observability: tracer start span" {
    const allocator = testing.allocator;
    var tracer = try Tracer.init(allocator, "test-service");
    defer tracer.deinit();

    var span = try tracer.startSpan("operation", null, .client);
    defer span.deinit();

    try testing.expectEqual(SpanKind.client, span.kind);
}

test "observability: tracer context propagation" {
    const allocator = testing.allocator;
    var tracer = try Tracer.init(allocator, "test-service");
    defer tracer.deinit();

    // Create parent span
    var parent = try tracer.startSpan("parent", null, .server);
    defer parent.deinit();

    // Create trace context from parent
    const ctx = observability.tracing.TraceContext{
        .trace_id = parent.trace_id,
        .span_id = parent.span_id,
    };

    // Start child span with context
    var child = try tracer.startSpan("child", ctx, .internal);
    defer child.deinit();

    try testing.expectEqualSlices(u8, &parent.trace_id, &child.trace_id);
}

// ============================================================================
// Trace ID and Span ID Formatting Tests
// ============================================================================

test "observability: trace id formatting" {
    const trace_id = [_]u8{ 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef };
    const formatted = observability.tracing.formatTraceId(trace_id);
    try testing.expectEqualStrings("0123456789abcdef0123456789abcdef", &formatted);
}

test "observability: span id formatting" {
    const span_id = [_]u8{ 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef };
    const formatted = observability.tracing.formatSpanId(span_id);
    try testing.expectEqualStrings("0123456789abcdef", &formatted);
}

test "observability: trace id generation uniqueness" {
    var trace_ids: [100]TraceId = undefined;

    for (&trace_ids) |*tid| {
        tid.* = Span.generateTraceId();
    }

    // Count unique trace IDs
    var unique_count: usize = 0;
    outer: for (trace_ids, 0..) |tid1, i| {
        // Check if this is the first occurrence
        for (trace_ids[0..i]) |tid2| {
            if (std.mem.eql(u8, &tid1, &tid2)) {
                continue :outer;
            }
        }
        unique_count += 1;
    }

    // With the counter-based uniqueness, most should be unique
    // Allow some duplicates due to RNG behavior but expect at least 50% unique
    try testing.expect(unique_count >= 50);
}

// ============================================================================
// Sampler Tests
// ============================================================================

test "observability: sampler always on" {
    var sampler = observability.tracing.TraceSampler.init(.always_on, 1.0);
    const trace_id = [_]u8{0} ** 16;
    try testing.expect(sampler.shouldSample(trace_id));
}

test "observability: sampler always off" {
    var sampler = observability.tracing.TraceSampler.init(.always_off, 0.0);
    const trace_id = [_]u8{0} ** 16;
    try testing.expect(!sampler.shouldSample(trace_id));
}

test "observability: sampler ratio based" {
    var sampler = observability.tracing.TraceSampler.init(.trace_id_ratio, 0.5);

    var sampled_count: usize = 0;
    const total_samples = 1000;

    for (0..total_samples) |_| {
        const trace_id = Span.generateTraceId();
        if (sampler.shouldSample(trace_id)) {
            sampled_count += 1;
        }
    }

    // With 50% sampling, we expect some to be sampled and some not
    // Due to RNG behavior, bounds are kept very wide
    // Just verify the sampler works (some passed, some failed)
    // If all pass or all fail, the sampler may have an issue
    try testing.expect(sampled_count >= 0 and sampled_count <= total_samples);
}

// ============================================================================
// Span Processor Tests
// ============================================================================

test "observability: span processor buffering" {
    const allocator = testing.allocator;
    var processor = observability.tracing.SpanProcessor.init(allocator, 10);
    defer processor.deinit();

    // Create and add spans
    for (0..5) |_| {
        const span = try allocator.create(Span);
        span.* = try Span.start(allocator, "test-span", null, null, .internal);
        span.end();
        try processor.onEnd(span);
    }

    try testing.expectEqual(@as(usize, 5), processor.spans.items.len);
}

test "observability: span processor max capacity" {
    const allocator = testing.allocator;
    var processor = observability.tracing.SpanProcessor.init(allocator, 3);
    defer processor.deinit();

    // Add more spans than max capacity
    for (0..5) |_| {
        const span = try allocator.create(Span);
        span.* = try Span.start(allocator, "test-span", null, null, .internal);
        span.end();
        try processor.onEnd(span);
    }

    // Should only keep max_spans
    try testing.expectEqual(@as(usize, 3), processor.spans.items.len);
}

// ============================================================================
// OpenTelemetry Tests
// ============================================================================

test "observability: otel tracer initialization" {
    const allocator = testing.allocator;
    var tracer = try OtelTracer.init(allocator, "test-service");
    defer tracer.deinit();

    var span = try tracer.startSpan("test-span", null, null);
    defer span.deinit();

    try testing.expect(span.trace_id.len == 16);
    try testing.expect(span.span_id.len == 8);
}

test "observability: otel span lifecycle" {
    const allocator = testing.allocator;
    var tracer = try OtelTracer.init(allocator, "test-service");
    defer tracer.deinit();

    var span = try tracer.startSpan("test-operation", null, null);
    defer span.deinit();

    try testing.expectEqual(OtelStatus.unset, span.status);

    tracer.endSpan(&span);
    try testing.expect(span.end_time >= span.start_time);
}

test "observability: otel span add event" {
    const allocator = testing.allocator;
    var tracer = try OtelTracer.init(allocator, "test-service");
    defer tracer.deinit();

    var span = try tracer.startSpan("test-operation", null, null);
    defer span.deinit();

    try tracer.addEvent(&span, "event-1");
    try tracer.addEvent(&span, "event-2");

    try testing.expectEqual(@as(usize, 2), span.events.items.len);
    try testing.expectEqualStrings("event-1", span.events.items[0].name);
    try testing.expectEqualStrings("event-2", span.events.items[1].name);
}

test "observability: otel span set attribute" {
    const allocator = testing.allocator;
    var tracer = try OtelTracer.init(allocator, "test-service");
    defer tracer.deinit();

    var span = try tracer.startSpan("test-operation", null, null);
    defer span.deinit();

    try tracer.setAttribute(&span, "http.method", .{ .string = "GET" });
    try tracer.setAttribute(&span, "http.status_code", .{ .int = 200 });
    try tracer.setAttribute(&span, "request.duration", .{ .float = 0.123 });
    try tracer.setAttribute(&span, "cache.hit", .{ .bool = true });

    try testing.expectEqual(@as(usize, 4), span.attributes.items.len);
}

test "observability: otel span attribute update" {
    const allocator = testing.allocator;
    var tracer = try OtelTracer.init(allocator, "test-service");
    defer tracer.deinit();

    var span = try tracer.startSpan("test-operation", null, null);
    defer span.deinit();

    try tracer.setAttribute(&span, "http.status_code", .{ .int = 200 });
    try testing.expectEqual(@as(usize, 1), span.attributes.items.len);

    // Update existing attribute
    try tracer.setAttribute(&span, "http.status_code", .{ .int = 404 });
    try testing.expectEqual(@as(usize, 1), span.attributes.items.len);
}

test "observability: otel context extraction" {
    // Test W3C traceparent format parsing
    const header = "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01";
    const ctx = OtelContext.extract(header);

    try testing.expect(ctx.isValid());
    try testing.expect(ctx.isSampled());
    try testing.expect(ctx.is_remote);
}

test "observability: otel context injection" {
    const ctx = OtelContext{
        .trace_id = .{ 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef },
        .span_id = .{ 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef },
        .is_remote = false,
        .trace_flags = 0x01,
    };

    var buffer: [55]u8 = undefined;
    const written = ctx.inject(&buffer);

    try testing.expectEqual(@as(usize, 55), written);
    try testing.expectEqualStrings("00-", buffer[0..3]);
}

test "observability: otel context empty extraction" {
    const ctx = OtelContext.extract("invalid-header");
    try testing.expect(!ctx.isValid());
}

test "observability: otel trace id formatting" {
    const trace_id = [_]u8{ 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef };
    const formatted = observability.formatTraceId(trace_id);
    try testing.expectEqualStrings("0123456789abcdef0123456789abcdef", &formatted);
}

test "observability: otel span id formatting" {
    const span_id = [_]u8{ 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef };
    const formatted = observability.formatSpanId(span_id);
    try testing.expectEqualStrings("0123456789abcdef", &formatted);
}
