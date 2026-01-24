//! Prometheus Text Format Export
//!
//! Writes metrics in Prometheus exposition format.

const std = @import("std");
const primitives = @import("primitives.zig");

pub const MetricWriter = struct {
    output: std.ArrayList(u8),

    pub fn init(allocator: std.mem.Allocator) MetricWriter {
        return .{ .output = std.ArrayList(u8).init(allocator) };
    }

    pub fn deinit(self: *MetricWriter) void {
        self.output.deinit();
    }

    pub fn writeCounter(self: *MetricWriter, name: []const u8, help: []const u8, value: u64, labels: ?[]const u8) !void {
        const writer = self.output.writer();
        try writer.print("# HELP {s} {s}\n", .{ name, help });
        try writer.print("# TYPE {s} counter\n", .{name});
        if (labels) |l| {
            try writer.print("{s}{{{s}}} {d}\n\n", .{ name, l, value });
        } else {
            try writer.print("{s} {d}\n\n", .{ name, value });
        }
    }

    pub fn writeGauge(self: *MetricWriter, name: []const u8, help: []const u8, value: anytype, labels: ?[]const u8) !void {
        const writer = self.output.writer();
        try writer.print("# HELP {s} {s}\n", .{ name, help });
        try writer.print("# TYPE {s} gauge\n", .{name});
        if (labels) |l| {
            try writer.print("{s}{{{s}}} {d}\n\n", .{ name, l, value });
        } else {
            try writer.print("{s} {d}\n\n", .{ name, value });
        }
    }

    pub fn writeHistogram(
        self: *MetricWriter,
        name: []const u8,
        help: []const u8,
        comptime bucket_count: usize,
        histogram: *primitives.Histogram(bucket_count),
        labels: ?[]const u8,
    ) !void {
        histogram.mutex.lock();
        defer histogram.mutex.unlock();

        const writer = self.output.writer();
        try writer.print("# HELP {s} {s}\n", .{ name, help });
        try writer.print("# TYPE {s} histogram\n", .{name});

        const label_prefix = if (labels) |l| l else "";
        const comma = if (labels != null) "," else "";

        var cumulative: u64 = 0;
        for (histogram.buckets, 0..) |bucket, i| {
            cumulative += bucket;
            try writer.print("{s}_bucket{{{s}{s}le=\"{d}\"}} {d}\n", .{
                name, label_prefix, comma, histogram.bucket_bounds[i], cumulative,
            });
        }
        try writer.print("{s}_bucket{{{s}{s}le=\"+Inf\"}} {d}\n", .{
            name, label_prefix, comma, histogram.count,
        });
        try writer.print("{s}_sum{{{s}}} {d}\n", .{ name, label_prefix, histogram.sum });
        try writer.print("{s}_count{{{s}}} {d}\n\n", .{ name, label_prefix, histogram.count });
    }

    pub fn finish(self: *MetricWriter) ![]u8 {
        return try self.output.toOwnedSlice();
    }

    pub fn clear(self: *MetricWriter) void {
        self.output.clearRetainingCapacity();
    }
};

// ============================================================================
// Tests
// ============================================================================

test "prometheus counter export" {
    const allocator = std.testing.allocator;
    var writer = MetricWriter.init(allocator);
    defer writer.deinit();

    try writer.writeCounter("requests_total", "Total requests", 42, null);
    const output = try writer.finish();
    defer allocator.free(output);

    try std.testing.expect(std.mem.indexOf(u8, output, "requests_total 42") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "# TYPE requests_total counter") != null);
}

test "prometheus gauge export with labels" {
    const allocator = std.testing.allocator;
    var writer = MetricWriter.init(allocator);
    defer writer.deinit();

    try writer.writeGauge("temperature", "Current temperature", @as(i64, 25), "location=\"server1\"");
    const output = try writer.finish();
    defer allocator.free(output);

    try std.testing.expect(std.mem.indexOf(u8, output, "temperature{location=\"server1\"} 25") != null);
}

test "prometheus histogram export" {
    const allocator = std.testing.allocator;
    var writer = MetricWriter.init(allocator);
    defer writer.deinit();

    var hist = primitives.LatencyHistogram.initDefault();
    hist.observe(10);
    hist.observe(50);
    hist.observe(100);

    try writer.writeHistogram("request_duration_ms", "Request duration", 14, &hist, null);
    const output = try writer.finish();
    defer allocator.free(output);

    try std.testing.expect(std.mem.indexOf(u8, output, "request_duration_ms_bucket") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "request_duration_ms_sum") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "request_duration_ms_count") != null);
}
