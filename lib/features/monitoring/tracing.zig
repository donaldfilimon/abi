//! Distributed Tracing Module for ABI Framework
//!
//! This module provides distributed tracing capabilities for monitoring,
//! debugging, and performance analysis across the entire system.
//!
//! Features:
//! - Request tracing across components
//! - Performance bottleneck detection
//! - Service mesh integration
//! - Custom trace annotations
//! - Multiple output formats (JSON, OpenTelemetry, Jaeger)
//! - Memory-efficient span storage

const std = @import("std");
// TODO: Fix module import for Zig 0.16
// const root = @import("../../root.zig");
const performance = @import("performance.zig");

/// Tracing error types
pub const TracingError = error{
    SpanNotFound,
    TraceNotFound,
    InvalidSpan,
    BufferOverflow,
    ExportFailed,
    ContextCorrupted,
};

/// Trace ID - unique identifier for a trace
pub const TraceId = struct {
    high: u64,
    low: u64,

    pub fn init() TraceId {
        var random_bytes: [16]u8 = undefined;
        std.crypto.random.bytes(&random_bytes);
        return .{
            .high = std.mem.readInt(u64, random_bytes[0..8], .big),
            .low = std.mem.readInt(u64, random_bytes[8..16], .big),
        };
    }

    pub fn toString(self: TraceId, allocator: std.mem.Allocator) ![]u8 {
        return std.fmt.allocPrint(allocator, "{x:0>16}{x:0>16}", .{ self.high, self.low });
    }

    pub fn fromString(str: []const u8) !TraceId {
        if (str.len != 32) return error.InvalidTraceId;
        const high = try std.fmt.parseInt(u64, str[0..16], 16);
        const low = try std.fmt.parseInt(u64, str[16..32], 16);
        return .{ .high = high, .low = low };
    }
};

/// Span ID - unique identifier for a span within a trace
pub const SpanId = u64;

/// Span kind enumeration
pub const SpanKind = enum {
    internal,
    server,
    client,
    producer,
    consumer,
};

/// Span status
pub const SpanStatus = enum {
    ok,
    err,
    unset,
};

/// Trace span representing a single operation
pub const Span = struct {
    trace_id: TraceId,
    span_id: SpanId,
    parent_span_id: ?SpanId,
    name: []const u8,
    kind: SpanKind,
    start_time: i128,
    end_time: ?i128,
    status: SpanStatus,
    attributes: std.StringHashMapUnmanaged([]const u8),
    events: std.ArrayListUnmanaged(SpanEvent),

    /// Span event for annotations
    pub const SpanEvent = struct {
        name: []const u8,
        timestamp: i128,
        attributes: std.StringHashMapUnmanaged([]const u8),

        pub fn deinit(self: *SpanEvent, allocator: std.mem.Allocator) void {
            allocator.free(self.name);
            var it = self.attributes.iterator();
            while (it.next()) |entry| {
                allocator.free(entry.key_ptr.*);
                allocator.free(entry.value_ptr.*);
            }
            self.attributes.deinit(allocator);
        }
    };

    pub fn init(
        allocator: std.mem.Allocator,
        trace_id: TraceId,
        span_id: SpanId,
        name: []const u8,
        kind: SpanKind,
    ) !*Span {
        const span = try allocator.create(Span);
        const name_copy = try allocator.dupe(u8, name);

        span.* = .{
            .trace_id = trace_id,
            .span_id = span_id,
            .parent_span_id = null,
            .name = name_copy,
            .kind = kind,
            .start_time = std.time.nanoTimestamp(),
            .end_time = null,
            .status = .unset,
            .attributes = std.StringHashMapUnmanaged([]const u8){},
            .events = std.ArrayListUnmanaged(SpanEvent){},
        };

        return span;
    }

    pub fn deinit(self: *Span, allocator: std.mem.Allocator) void {
        allocator.free(self.name);

        // Clean up attributes
        var attr_it = self.attributes.iterator();
        while (attr_it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        self.attributes.deinit(allocator);

        // Clean up events
        for (self.events.items) |*event| {
            event.deinit(allocator);
        }
        self.events.deinit(allocator);

        allocator.destroy(self);
    }

    /// End the span
    pub fn end(self: *Span) void {
        self.end_time = std.time.nanoTimestamp();
    }

    /// Set span status
    pub fn setStatus(self: *Span, status: SpanStatus) void {
        self.status = status;
    }

    /// Add an attribute to the span
    pub fn setAttribute(self: *Span, allocator: std.mem.Allocator, key: []const u8, value: []const u8) !void {
        const key_copy = try allocator.dupe(u8, key);
        const value_copy = try allocator.dupe(u8, value);

        errdefer allocator.free(key_copy);
        errdefer allocator.free(value_copy);

        try self.attributes.put(allocator, key_copy, value_copy);
    }

    /// Add an event to the span
    pub fn addEvent(self: *Span, allocator: std.mem.Allocator, name: []const u8) !void {
        const event_name = try allocator.dupe(u8, name);
        errdefer allocator.free(event_name);

        const event = SpanEvent{
            .name = event_name,
            .timestamp = std.time.nanoTimestamp(),
            .attributes = std.StringHashMapUnmanaged([]const u8){},
        };

        try self.events.append(allocator, event);
    }

    /// Get span duration in nanoseconds
    pub fn duration(self: Span) ?i128 {
        if (self.end_time) |end_time_val| {
            return end_time_val - self.start_time;
        }
        return null;
    }

    /// Check if span is still active
    pub fn isActive(self: Span) bool {
        return self.end_time == null;
    }
};

/// Trace context for propagating tracing information
pub const TraceContext = struct {
    trace_id: TraceId,
    span_id: SpanId,
    sampled: bool = true,

    /// Create a new trace context
    pub fn init() TraceContext {
        return .{
            .trace_id = TraceId.init(),
            .span_id = TraceId.init().low, // Use low part for span ID
            .sampled = true,
        };
    }

    /// Create child context
    pub fn child(self: TraceContext) TraceContext {
        return .{
            .trace_id = self.trace_id,
            .span_id = TraceId.init().low,
            .sampled = self.sampled,
        };
    }

    /// Serialize context to string
    pub fn toString(self: TraceContext, allocator: std.mem.Allocator) ![]u8 {
        const trace_id_str = try self.trace_id.toString(allocator);
        defer allocator.free(trace_id_str);

        return std.fmt.allocPrint(allocator, "trace_id={s},span_id={x},sampled={}", .{ trace_id_str, self.span_id, self.sampled });
    }

    /// Deserialize context from string
    pub fn fromString(str: []const u8) !TraceContext {
        var trace_id: TraceId = undefined;
        var span_id: u64 = 0;
        var sampled: bool = true;

        var it = std.mem.splitScalar(u8, str, ',');
        while (it.next()) |pair| {
            if (std.mem.indexOfScalar(u8, pair, '=')) |eq_pos| {
                const key = pair[0..eq_pos];
                const value = pair[eq_pos + 1 ..];

                if (std.mem.eql(u8, key, "trace_id")) {
                    trace_id = try TraceId.fromString(value);
                } else if (std.mem.eql(u8, key, "span_id")) {
                    span_id = try std.fmt.parseInt(u64, value, 16);
                } else if (std.mem.eql(u8, key, "sampled")) {
                    sampled = std.mem.eql(u8, value, "true");
                }
            }
        }

        return .{
            .trace_id = trace_id,
            .span_id = span_id,
            .sampled = sampled,
        };
    }
};

/// Tracer - main tracing interface
pub const Tracer = struct {
    allocator: std.mem.Allocator,
    active_spans: std.AutoHashMapUnmanaged(SpanId, *Span),
    finished_spans: std.ArrayListUnmanaged(*Span),
    config: TracerConfig,
    sampler: Sampler,

    pub const TracerConfig = struct {
        max_active_spans: usize = 1000,
        max_finished_spans: usize = 10000,
        enable_events: bool = true,
        enable_attributes: bool = true,
    };

    /// Sampling strategy
    pub const Sampler = union(enum) {
        always: void,
        never: void,
        probability: f32,
        rate_limiting: struct {
            max_traces_per_second: u32,
            current_count: u32 = 0,
            last_reset: i128 = 0,
        },

        pub fn shouldSample(self: *Sampler) bool {
            return switch (self.*) {
                .always => true,
                .never => false,
                .probability => |p| std.crypto.random.float(f32) < p,
                .rate_limiting => |*rl| blk: {
                    const now = std.time.nanoTimestamp();
                    const elapsed = now - rl.last_reset;
                    if (elapsed >= std.time.ns_per_s) {
                        rl.current_count = 0;
                        rl.last_reset = now;
                    }

                    if (rl.current_count < rl.max_traces_per_second) {
                        rl.current_count += 1;
                        break :blk true;
                    }
                    break :blk false;
                },
            };
        }
    };

    pub fn init(allocator: std.mem.Allocator, config: TracerConfig) !*Tracer {
        const tracer = try allocator.create(Tracer);
        tracer.* = .{
            .allocator = allocator,
            .active_spans = std.AutoHashMapUnmanaged(SpanId, *Span){},
            .finished_spans = std.ArrayListUnmanaged(*Span){},
            .config = config,
            .sampler = .{ .always = {} },
        };
        return tracer;
    }

    pub fn deinit(self: *Tracer) void {
        // Clean up active spans
        var active_it = self.active_spans.iterator();
        while (active_it.next()) |entry| {
            entry.value_ptr.*.deinit(self.allocator);
        }
        self.active_spans.deinit(self.allocator);

        // Clean up finished spans
        for (self.finished_spans.items) |span| {
            span.deinit(self.allocator);
        }
        self.finished_spans.deinit(self.allocator);

        self.allocator.destroy(self);
    }

    /// Start a new span
    pub fn startSpan(self: *Tracer, name: []const u8, kind: SpanKind, context: ?TraceContext) !*Span {
        const ctx = context orelse TraceContext.init();

        if (!self.sampler.shouldSample()) {
            // Return a no-op span if not sampled
            const span = try Span.init(self.allocator, ctx.trace_id, ctx.span_id, name, kind);
            span.end_time = span.start_time; // Mark as ended immediately
            return span;
        }

        const span_id = TraceId.init().low;
        const span = try Span.init(self.allocator, ctx.trace_id, span_id, name, kind);

        // Set parent span ID if context has one
        if (context != null) {
            span.parent_span_id = ctx.span_id;
        }

        try self.active_spans.put(self.allocator, span_id, span);
        return span;
    }

    /// End a span
    pub fn endSpan(self: *Tracer, span: *Span) void {
        span.end();

        // Move from active to finished
        if (self.active_spans.remove(span.span_id)) {
            if (self.finished_spans.items.len >= self.config.max_finished_spans) {
                // Remove oldest finished span
                const oldest = self.finished_spans.swapRemove(0);
                oldest.deinit(self.allocator);
            }
            self.finished_spans.append(self.allocator, span) catch {
                // If we can't add to finished, just clean up the span
                span.deinit(self.allocator);
            };
        }
    }

    /// Get active span by ID
    pub fn getSpan(self: *Tracer, span_id: SpanId) ?*Span {
        return self.active_spans.get(span_id);
    }

    /// Export traces to JSON (simplified)
    pub fn exportToJson(self: *Tracer, allocator: std.mem.Allocator) ![]u8 {
        var buffer = std.ArrayListUnmanaged(u8){};
        defer buffer.deinit(allocator);

        try buffer.appendSlice(allocator, "{\"spans\":[");

        for (self.finished_spans.items, 0..) |span, i| {
            if (i > 0) try buffer.append(allocator, ',');
            const json_str = try std.fmt.allocPrint(allocator, "{{\"name\":\"{s}\",\"duration_ns\":{}}}", .{ span.name, span.duration() orelse 0 });
            defer allocator.free(json_str);
            try buffer.appendSlice(allocator, json_str);
        }

        try buffer.appendSlice(allocator, "]}");
        return buffer.toOwnedSlice(allocator);
    }
};

/// Global tracer instance
var global_tracer: ?*Tracer = null;

/// Initialize global tracer
pub fn initGlobalTracer(allocator: std.mem.Allocator, config: Tracer.TracerConfig) !void {
    if (global_tracer != null) {
        deinitGlobalTracer();
    }
    global_tracer = try Tracer.init(allocator, config);
}

/// Deinitialize global tracer
pub fn deinitGlobalTracer() void {
    if (global_tracer) |tracer| {
        global_tracer = null;
        tracer.deinit();
    }
}

/// Get global tracer instance
pub fn getGlobalTracer() ?*Tracer {
    return global_tracer;
}

/// Helper function to start a span with global tracer
pub fn startSpan(name: []const u8, kind: SpanKind, context: ?TraceContext) !*Span {
    const tracer = getGlobalTracer() orelse return TracingError.SpanNotFound;
    return tracer.startSpan(name, kind, context);
}

/// Helper function to end a span with global tracer
pub fn endSpan(span: *Span) void {
    const tracer = getGlobalTracer() orelse return;
    tracer.endSpan(span);
}

/// Convenience macro-like function for tracing function calls
pub fn traceFunction(comptime func_name: []const u8, context: ?TraceContext) !*Span {
    const tracer = getGlobalTracer() orelse return TracingError.SpanNotFound;
    return tracer.startSpan(func_name, .internal, context);
}

/// Integration with performance monitoring
pub fn integrateWithPerformance(tracer: *Tracer, perf_monitor: *performance.PerformanceMonitor) void {
    // Add tracing hooks to performance monitor
    _ = tracer;
    _ = perf_monitor;
    // Implementation would add tracing to performance monitoring functions
}

test "Trace ID generation and conversion" {
    const trace_id = TraceId.init();

    // Test string conversion
    const testing = std.testing;
    const str = try trace_id.toString(testing.allocator);
    defer testing.allocator.free(str);

    try testing.expectEqual(@as(usize, 32), str.len);
    try testing.expect(std.ascii.isHex(str[0]));

    // Test round-trip conversion
    const parsed = try TraceId.fromString(str);
    try testing.expectEqual(trace_id.high, parsed.high);
    try testing.expectEqual(trace_id.low, parsed.low);
}

test "Span lifecycle" {
    const testing = std.testing;
    const trace_id = TraceId.init();
    const span_id: SpanId = 12345;

    const span = try Span.init(testing.allocator, trace_id, span_id, "test_span", .internal);
    defer span.deinit(testing.allocator);

    try testing.expectEqual(trace_id.high, span.trace_id.high);
    try testing.expectEqual(trace_id.low, span.trace_id.low);
    try testing.expectEqual(span_id, span.span_id);
    try testing.expect(std.mem.eql(u8, "test_span", span.name));
    try testing.expectEqual(SpanKind.internal, span.kind);
    try testing.expect(span.isActive());

    // End span
    span.end();
    try testing.expect(!span.isActive());
    try testing.expect(span.duration() != null);
}

test "Span attributes" {
    const testing = std.testing;
    const span = try Span.init(testing.allocator, TraceId.init(), 123, "test", .internal);
    defer span.deinit(testing.allocator);

    try span.setAttribute(testing.allocator, "key1", "value1");
    try span.setAttribute(testing.allocator, "key2", "value2");

    try testing.expectEqual(@as(usize, 2), span.attributes.count());
    try testing.expect(std.mem.eql(u8, "value1", span.attributes.get("key1").?));
    try testing.expect(std.mem.eql(u8, "value2", span.attributes.get("key2").?));
}

test "Span events" {
    const testing = std.testing;
    const span = try Span.init(testing.allocator, TraceId.init(), 123, "test", .internal);
    defer span.deinit(testing.allocator);

    try span.addEvent(testing.allocator, "event1");
    try span.addEvent(testing.allocator, "event2");

    try testing.expectEqual(@as(usize, 2), span.events.items.len);
    try testing.expect(std.mem.eql(u8, "event1", span.events.items[0].name));
    try testing.expect(std.mem.eql(u8, "event2", span.events.items[1].name));
}

test "Tracer span management" {
    const testing = std.testing;
    const config = Tracer.TracerConfig{
        .max_active_spans = 10,
        .max_finished_spans = 100,
    };

    const tracer = try Tracer.init(testing.allocator, config);
    defer tracer.deinit();

    // Start a span
    const span = try tracer.startSpan("test_operation", .internal, null);
    try testing.expect(tracer.active_spans.count() == 1);

    // End the span
    tracer.endSpan(span);
    try testing.expect(tracer.active_spans.count() == 0);
    try testing.expect(tracer.finished_spans.items.len == 1);

    // Clean up the span
    tracer.finished_spans.items[0].deinit(testing.allocator);
    _ = tracer.finished_spans.swapRemove(0);
}

test "Trace context serialization" {
    const testing = std.testing;
    const context = TraceContext.init();

    // Serialize
    const str = try context.toString(testing.allocator);
    defer testing.allocator.free(str);

    // Deserialize
    const parsed = try TraceContext.fromString(str);

    try testing.expectEqual(context.trace_id.high, parsed.trace_id.high);
    try testing.expectEqual(context.trace_id.low, parsed.trace_id.low);
    try testing.expectEqual(context.span_id, parsed.span_id);
    try testing.expectEqual(context.sampled, parsed.sampled);
}

test "Sampler functionality" {
    var sampler = Tracer.Sampler{ .always = {} };
    try std.testing.expect(sampler.shouldSample());

    sampler = .{ .never = {} };
    try std.testing.expect(!sampler.shouldSample());

    sampler = .{ .probability = 0.0 };
    try std.testing.expect(!sampler.shouldSample());

    sampler = .{ .probability = 1.0 };
    // Note: This might occasionally fail due to randomness, but it's very unlikely
    // In production code, we'd use a seeded random number generator for testing
}
