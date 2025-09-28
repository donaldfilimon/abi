//! Telemetry and structured logging utilities for ABI.
//!
//! Provides trace identifiers, structured JSON logs, and runtime metrics
//! counters that can be shared between the CLI, TUI, and service layers.

const std = @import("std");

pub const Allocator = std.mem.Allocator;

pub const LogLevel = enum { debug, info, warn, error };

pub const CallStatus = enum { success, throttled, error };

pub const TraceId = struct {
    bytes: [16]u8,

    pub fn random() TraceId {
        var id = TraceId{ .bytes = undefined };
        std.crypto.random.bytes(&id.bytes);
        return id;
    }

    pub fn fromBytes(bytes: [16]u8) TraceId {
        return .{ .bytes = bytes };
    }

    pub fn format(self: TraceId, buffer: []u8) []const u8 {
        std.debug.assert(buffer.len >= 32);
        return std.fmt.bufPrint(buffer, "{s}", .{std.fmt.fmtSliceHexLower(&self.bytes)}) catch unreachable;
    }
};

pub const StructuredEvent = struct {
    timestamp_ns: i128,
    level: LogLevel,
    trace_id: TraceId,
    message: []const u8,
    persona: ?[]const u8 = null,
    backend: ?[]const u8 = null,
    model: ?[]const u8 = null,
    latency_ns: ?u64 = null,
    prompt_tokens: ?usize = null,
    completion_tokens: ?usize = null,
    retries: ?u8 = null,
    backoff_ms: ?u32 = null,
    status: ?CallStatus = null,
};

pub const StructuredLogger = struct {
    console: std.io.AnyWriter,
    file: ?std.fs.File = null,

    pub fn init(console: std.io.AnyWriter) StructuredLogger {
        return .{ .console = console, .file = null };
    }

    pub fn initWithFile(console: std.io.AnyWriter, file: std.fs.File) StructuredLogger {
        return .{ .console = console, .file = file };
    }

    pub fn deinit(self: *StructuredLogger) void {
        if (self.file) |*f| {
            f.close();
            self.file = null;
        }
    }

    pub fn logEvent(self: *StructuredLogger, allocator: Allocator, event: StructuredEvent) !void {
        var buffer = std.ArrayList(u8).init(allocator);
        defer buffer.deinit();

        var writer = buffer.writer();
        try writer.writeByte('{');
        var first = true;
        try writeNumberField(writer, &first, "timestamp", event.timestamp_ns);
        try writeStringField(writer, &first, "level", levelToString(event.level));
        var trace_buf: [32]u8 = undefined;
        const trace_hex = event.trace_id.format(&trace_buf);
        try writeStringField(writer, &first, "trace_id", trace_hex);
        try writeStringField(writer, &first, "message", event.message);
        try writeOptionalStringField(writer, &first, "persona", event.persona);
        try writeOptionalStringField(writer, &first, "backend", event.backend);
        try writeOptionalStringField(writer, &first, "model", event.model);
        try writeOptionalNumberField(writer, &first, "latency_ns", u64, event.latency_ns);
        try writeOptionalNumberField(writer, &first, "prompt_tokens", usize, event.prompt_tokens);
        try writeOptionalNumberField(writer, &first, "completion_tokens", usize, event.completion_tokens);
        try writeOptionalNumberField(writer, &first, "retries", u8, event.retries);
        try writeOptionalNumberField(writer, &first, "backoff_ms", u32, event.backoff_ms);
        if (event.status) |status| {
            try writeStringField(writer, &first, "status", callStatusToString(status));
        }
        try writer.writeByte('}');

        try self.console.writeAll(buffer.items);
        try self.console.writeByte('\n');

        if (self.file) |*file| {
            var file_writer = file.writer();
            try file_writer.writeAll(buffer.items);
            try file_writer.writeByte('\n');
            try file_writer.flush();
        }
    }
};

fn levelToString(level: LogLevel) []const u8 {
    return switch (level) {
        .debug => "debug",
        .info => "info",
        .warn => "warn",
        .error => "error",
    };
}

fn callStatusToString(status: CallStatus) []const u8 {
    return switch (status) {
        .success => "success",
        .throttled => "throttled",
        .error => "error",
    };
}

fn writeComma(writer: anytype, first: *bool) !void {
    if (first.*) {
        first.* = false;
    } else {
        try writer.writeByte(',');
    }
}

fn writeStringField(writer: anytype, first: *bool, key: []const u8, value: []const u8) !void {
    try writeComma(writer, first);
    try writer.writeByte('"');
    try writer.writeAll(key);
    try writer.writeAll("":"");
    try std.json.escapeString(value, .{}, writer);
    try writer.writeByte('"');
}

fn writeNumberField(writer: anytype, first: *bool, key: []const u8, value: anytype) !void {
    try writeComma(writer, first);
    try writer.writeByte('"');
    try writer.writeAll(key);
    try writer.writeAll("":");
    try writer.print("{}", .{value});
}

fn writeOptionalStringField(writer: anytype, first: *bool, key: []const u8, value: ?[]const u8) !void {
    if (value) |val| {
        try writeStringField(writer, first, key, val);
    }
}

fn writeOptionalNumberField(writer: anytype, first: *bool, key: []const u8, comptime T: type, value: ?T) !void {
    if (value) |val| {
        try writeNumberField(writer, first, key, val);
    }
}

pub const TelemetrySink = struct {
    allocator: Allocator,
    total_calls: usize = 0,
    total_errors: usize = 0,
    latency_samples: std.ArrayListUnmanaged(u64) = .{},
    persona_usage: std.StringHashMap(usize),
    error_counts: std.StringHashMap(usize),

    pub fn init(allocator: Allocator) TelemetrySink {
        return .{
            .allocator = allocator,
            .persona_usage = std.StringHashMap(usize).init(allocator),
            .error_counts = std.StringHashMap(usize).init(allocator),
        };
    }

    pub fn deinit(self: *TelemetrySink) void {
        self.latency_samples.deinit(self.allocator);
        var it = self.persona_usage.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.persona_usage.deinit();
        it = self.error_counts.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.error_counts.deinit();
    }

    pub fn record(self: *TelemetrySink, persona: []const u8, latency_ns: u64, status: CallStatus, error_tag: ?[]const u8) !void {
        self.total_calls += 1;
        try self.latency_samples.append(self.allocator, latency_ns);

        if (status != .success) {
            self.total_errors += 1;
        }

        try incrementStringCount(self.allocator, &self.persona_usage, persona);
        if (error_tag) |tag| {
            try incrementStringCount(self.allocator, &self.error_counts, tag);
        }
    }

    pub fn snapshot(self: *TelemetrySink, allocator: Allocator) !TelemetrySnapshot {
        var persona_counts = std.ArrayList(PersonaCount).init(allocator);
        errdefer persona_counts.deinit();

        var persona_it = self.persona_usage.iterator();
        while (persona_it.next()) |entry| {
            const name = try allocator.dupe(u8, entry.key_ptr.*);
            errdefer allocator.free(name);
            try persona_counts.append(.{ .name = name, .calls = entry.value_ptr.* });
        }

        const total_latency = totalLatency(self.latency_samples.items);
        const avg_latency: u64 = if (self.latency_samples.items.len == 0)
            0
        else
            @intCast(total_latency / self.latency_samples.items.len);

        const percentiles = try computePercentiles(allocator, self.latency_samples.items);

        return TelemetrySnapshot{
            .total_calls = self.total_calls,
            .total_errors = self.total_errors,
            .avg_latency_ns = avg_latency,
            .persona_counts = try persona_counts.toOwnedSlice(),
            .p50_latency_ns = percentiles.p50,
            .p95_latency_ns = percentiles.p95,
            .p99_latency_ns = percentiles.p99,
        };
    }
};

fn incrementStringCount(allocator: Allocator, map: *std.StringHashMap(usize), key: []const u8) !void {
    if (map.getPtr(key)) |existing| {
        existing.* += 1;
        return;
    }

    const owned = try allocator.dupe(u8, key);
    errdefer allocator.free(owned);
    try map.put(owned, 1);
}

fn totalLatency(samples: []const u64) u128 {
    var sum: u128 = 0;
    for (samples) |value| {
        sum += value;
    }
    return sum;
}

const PercentileResult = struct {
    p50: u64,
    p95: u64,
    p99: u64,
};

fn computePercentiles(allocator: Allocator, samples: []const u64) !PercentileResult {
    if (samples.len == 0) {
        return PercentileResult{ .p50 = 0, .p95 = 0, .p99 = 0 };
    }

    var sorted = try allocator.dupe(u64, samples);
    defer allocator.free(sorted);
    std.sort.sort(u64, sorted, {}, std.sort.asc(u64));

    const idx = struct {
        fn percentile(len: usize, p: f64) usize {
            const pos = (p / 100.0) * @as(f64, @floatFromInt(len - 1));
            return @intFromFloat(std.math.round(pos));
        }
    };

    const len = sorted.len;
    return PercentileResult{
        .p50 = sorted[idx.percentile(len, 50.0)],
        .p95 = sorted[idx.percentile(len, 95.0)],
        .p99 = sorted[idx.percentile(len, 99.0)],
    };
}

pub const PersonaCount = struct {
    name: []const u8,
    calls: usize,
};

pub const TelemetrySnapshot = struct {
    total_calls: usize,
    total_errors: usize,
    avg_latency_ns: u64,
    persona_counts: []PersonaCount,
    p50_latency_ns: u64,
    p95_latency_ns: u64,
    p99_latency_ns: u64,

    pub fn errorRate(self: TelemetrySnapshot) f64 {
        if (self.total_calls == 0) return 0;
        return @as(f64, @floatFromInt(self.total_errors)) / @as(f64, @floatFromInt(self.total_calls));
    }

    pub fn deinit(self: *TelemetrySnapshot, allocator: Allocator) void {
        for (self.persona_counts) |entry| {
            allocator.free(entry.name);
        }
        allocator.free(self.persona_counts);
        self.persona_counts = &.{};
    }
};

fn expectSnapshot(snapshot: TelemetrySnapshot, expected_calls: usize, expected_errors: usize) !void {
    const testing = std.testing;
    try testing.expectEqual(expected_calls, snapshot.total_calls);
    try testing.expectEqual(expected_errors, snapshot.total_errors);
}

test "structured logger writes JSON line" {
    const testing = std.testing;
    var buffer = std.ArrayList(u8).init(testing.allocator);
    defer buffer.deinit();
    var logger = StructuredLogger.init(buffer.writer().any());
    defer logger.deinit();
    const zero_bytes = [_]u8{0} ** 16;
    const event = StructuredEvent{
        .timestamp_ns = 1234,
        .level = .info,
        .trace_id = TraceId.fromBytes(zero_bytes),
        .message = "hello",
        .persona = "creative",
        .status = .success,
    };
    try logger.logEvent(testing.allocator, event);
    try testing.expect(buffer.items.len > 0);
}

test "telemetry sink aggregates metrics" {
    const testing = std.testing;
    var sink = TelemetrySink.init(testing.allocator);
    defer sink.deinit();

    try sink.record("creative", 100, .success, null);
    try sink.record("creative", 200, .error, "timeout");
    try sink.record("analytical", 300, .throttled, "rate_limit");

    var snapshot = try sink.snapshot(testing.allocator);
    defer snapshot.deinit(testing.allocator);

    try expectSnapshot(snapshot, 3, 2);
    try testing.expect(snapshot.avg_latency_ns > 0);
    try testing.expect(snapshot.p95_latency_ns >= snapshot.p50_latency_ns);
}
