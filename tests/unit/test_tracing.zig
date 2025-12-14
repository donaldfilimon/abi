//! Tests for the distributed tracing module

const std = @import("std");
const tracing = @import("abi").monitoring.tracing;

test "Distributed tracing basic functionality" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Initialize tracer
    const config = tracing.Tracer.TracerConfig{
        .max_active_spans = 10,
        .max_finished_spans = 100,
    };

    const tracer = try tracing.Tracer.init(allocator, config);
    defer tracer.deinit();

    // Create and manage spans
    const span1 = try tracer.startSpan("test_operation", .internal, null);
    defer tracer.endSpan(span1);

    // Add attributes
    try span1.setAttribute(allocator, "operation", "test");
    try span1.setAttribute(allocator, "user_id", "12345");

    // Add events
    try span1.addEvent(allocator, "starting");
    std.time.sleep(1000000); // 1ms
    try span1.addEvent(allocator, "processing");
    std.time.sleep(1000000); // 1ms
    try span1.addEvent(allocator, "completed");

    // End span
    span1.end();

    // Verify span properties
    try testing.expectEqualStrings("test_operation", span1.name);
    try testing.expectEqual(tracing.SpanKind.internal, span1.kind);
    try testing.expect(span1.end_time != null);
    try testing.expect(span1.duration() != null);
    try testing.expect(span1.duration().? > 0);

    // Verify attributes
    try testing.expectEqual(@as(usize, 2), span1.attributes.count());
    try testing.expect(std.mem.eql(u8, "test", span1.attributes.get("operation").?));
    try testing.expect(std.mem.eql(u8, "12345", span1.attributes.get("user_id").?));

    // Verify events
    try testing.expectEqual(@as(usize, 3), span1.events.items.len);
    try testing.expect(std.mem.eql(u8, "starting", span1.events.items[0].name));
    try testing.expect(std.mem.eql(u8, "processing", span1.events.items[1].name));
    try testing.expect(std.mem.eql(u8, "completed", span1.events.items[2].name));
}

test "Trace context propagation" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Create initial context
    const context1 = tracing.TraceContext.init();

    // Create child context
    const context2 = context1.child();

    // Verify they have the same trace ID but different span IDs
    try testing.expectEqual(context1.trace_id.high, context2.trace_id.high);
    try testing.expectEqual(context1.trace_id.low, context2.trace_id.low);
    try testing.expect(context1.span_id != context2.span_id);

    // Test serialization
    const serialized = try context1.toString(allocator);
    defer allocator.free(serialized);

    const deserialized = try tracing.TraceContext.fromString(serialized);

    try testing.expectEqual(context1.trace_id.high, deserialized.trace_id.high);
    try testing.expectEqual(context1.trace_id.low, deserialized.trace_id.low);
    try testing.expectEqual(context1.span_id, deserialized.span_id);
    try testing.expectEqual(context1.sampled, deserialized.sampled);
}

test "Tracer span management" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const config = tracing.Tracer.TracerConfig{
        .max_active_spans = 5,
        .max_finished_spans = 10,
    };

    const tracer = try tracing.Tracer.init(allocator, config);
    defer tracer.deinit();

    // Start multiple spans
    const span1 = try tracer.startSpan("span1", .internal, null);
    const span2 = try tracer.startSpan("span2", .internal, null);
    const span3 = try tracer.startSpan("span3", .internal, null);

    try testing.expectEqual(@as(usize, 3), tracer.active_spans.count());

    // End spans
    tracer.endSpan(span1);
    tracer.endSpan(span2);

    try testing.expectEqual(@as(usize, 1), tracer.active_spans.count());
    try testing.expectEqual(@as(usize, 2), tracer.finished_spans.items.len);

    // End last span
    tracer.endSpan(span3);

    try testing.expectEqual(@as(usize, 0), tracer.active_spans.count());
    try testing.expectEqual(@as(usize, 3), tracer.finished_spans.items.len);

    // Clean up finished spans
    for (tracer.finished_spans.items) |span| {
        span.deinit(allocator);
    }
    tracer.finished_spans.clearRetainingCapacity();
}

test "Sampling strategies" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const config = tracing.Tracer.TracerConfig{};
    const tracer = try tracing.Tracer.init(allocator, config);
    defer tracer.deinit();

    // Test always sampler
    tracer.sampler = .{ .always = {} };
    try testing.expect(tracer.sampler.shouldSample());

    // Test never sampler
    tracer.sampler = .{ .never = {} };
    try testing.expect(!tracer.sampler.shouldSample());

    // Test probability sampler (this is non-deterministic, but we can test the structure)
    tracer.sampler = .{ .probability = 0.5 };
    const result = tracer.sampler.shouldSample();
    try testing.expect(@TypeOf(result) == bool);

    // Test rate limiting sampler
    tracer.sampler = .{ .rate_limiting = .{ .max_traces_per_second = 10 } };
    try testing.expect(tracer.sampler.shouldSample()); // First call should succeed
}

test "Global tracer integration" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Initialize global tracer
    const config = tracing.Tracer.TracerConfig{
        .max_active_spans = 10,
        .max_finished_spans = 50,
    };

    try tracing.initGlobalTracer(allocator, config);
    defer tracing.deinitGlobalTracer();

    // Get global tracer
    const global_tracer = tracing.getGlobalTracer();
    try testing.expect(global_tracer != null);

    // Use global helper functions
    const span = try tracing.startSpan("global_test", .internal, null);
    defer tracing.endSpan(span);

    try span.setAttribute(allocator, "test", "global");

    // Verify span was created
    try testing.expectEqualStrings("global_test", span.name);
    try testing.expect(std.mem.eql(u8, "global", span.attributes.get("test").?));
}

test "JSON export functionality" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const config = tracing.Tracer.TracerConfig{
        .max_finished_spans = 10,
    };

    const tracer = try tracing.Tracer.init(allocator, config);
    defer tracer.deinit();

    // Create and finish some spans
    const span1 = try tracer.startSpan("operation1", .internal, null);
    span1.end();
    tracer.endSpan(span1);

    const span2 = try tracer.startSpan("operation2", .internal, null);
    span2.end();
    tracer.endSpan(span2);

    // Export to JSON
    const json = try tracer.exportToJson(allocator);
    defer allocator.free(json);

    // Basic JSON structure validation
    try testing.expect(json.len > 0);
    try testing.expect(std.mem.indexOf(u8, json, "operation1") != null);
    try testing.expect(std.mem.indexOf(u8, json, "operation2") != null);
    try testing.expect(std.mem.indexOf(u8, json, "duration_ns") != null);

    // Clean up spans
    for (tracer.finished_spans.items) |span| {
        span.deinit(allocator);
    }
    tracer.finished_spans.clearRetainingCapacity();
}

test "Performance impact measurement" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const config = tracing.Tracer.TracerConfig{
        .max_active_spans = 1000,
        .max_finished_spans = 10000,
    };

    const tracer = try tracing.Tracer.init(allocator, config);
    defer tracer.deinit();

    // Measure time to create many spans
    const start_time = std.time.nanoTimestamp;

    var spans: [100]*tracing.Span = undefined;
    for (0..100) |i| {
        const span_name = try std.fmt.allocPrint(allocator, "span_{d}", .{i});
        defer allocator.free(span_name);

        spans[i] = try tracer.startSpan(span_name, .internal, null);
        spans[i].end();
        tracer.endSpan(spans[i]);
    }

    const end_time = std.time.nanoTimestamp;
    const total_time = end_time - start_time;

    // Should be reasonably fast (< 1ms per span typically)
    const avg_time_per_span = total_time / 100;
    try testing.expect(avg_time_per_span < 1000000); // Less than 1ms per span

    // Clean up
    for (tracer.finished_spans.items) |span| {
        span.deinit(allocator);
    }
    tracer.finished_spans.clearRetainingCapacity();
}

test "Error handling and edge cases" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const config = tracing.Tracer.TracerConfig{
        .max_active_spans = 2, // Small limit for testing
        .max_finished_spans = 1,
    };

    const tracer = try tracing.Tracer.init(allocator, config);
    defer tracer.deinit();

    // Test span not found
    const non_existent_span = tracer.getSpan(99999);
    try testing.expect(non_existent_span == null);

    // Test global tracer when not initialized
    tracing.deinitGlobalTracer(); // Ensure it's deinitialized
    const global_before_init = tracing.getGlobalTracer();
    try testing.expect(global_before_init == null);

    // Test starting span without global tracer
    const result = tracing.startSpan("test", .internal, null);
    try testing.expectError(tracing.TracingError.SpanNotFound, result);
}
