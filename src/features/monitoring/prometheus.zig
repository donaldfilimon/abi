//! Prometheus metrics exporter for observability.
const std = @import("std");
const observability = @import("../../shared/observability/mod.zig");

pub const PrometheusConfig = struct {
    enabled: bool = true,
    path: []const u8 = "/metrics",
    port: u16 = 9090,
    namespace: []const u8 = "abi",
    include_timestamp: bool = false,
};

pub const PrometheusExporter = struct {
    allocator: std.mem.Allocator,
    config: PrometheusConfig,
    metrics: *observability.MetricsCollector,
    running: std.atomic.Value(bool),
    server_thread: ?std.Thread = null,

    pub fn init(allocator: std.mem.Allocator, config: PrometheusConfig, metrics: *observability.MetricsCollector) !PrometheusExporter {
        return .{
            .allocator = allocator,
            .config = config,
            .metrics = metrics,
            .running = std.atomic.Value(bool).init(false),
            .server_thread = null,
        };
    }

    pub fn deinit(self: *PrometheusExporter) void {
        self.stop();
        self.* = undefined;
    }

    pub fn start(self: *PrometheusExporter) !void {
        if (self.running.load(.acquire)) return;
        self.running.store(true, .release);
        self.server_thread = try std.Thread.spawn(.{}, runServer, .{self});
    }

    pub fn stop(self: *PrometheusExporter) void {
        if (!self.running.load(.acquire)) return;
        self.running.store(false, .release);
        if (self.server_thread) |t| {
            t.join();
            self.server_thread = null;
        }
    }

    pub fn generateMetrics(self: *PrometheusExporter, allocator: std.mem.Allocator) ![]const u8 {
        var buffer = std.ArrayListUnmanaged(u8).empty;
        errdefer buffer.deinit(allocator);

        try buffer.appendSlice(allocator, "# HELP abi_build_info Build information\n");
        try buffer.appendSlice(allocator, "# TYPE abi_build_info gauge\n");
        try buffer.appendSlice(allocator, "abi_build_info{version=\"0.3.0\",commit=\"unknown\"} 1\n\n");

        try self.formatCounters(allocator, &buffer);
        try self.formatHistograms(allocator, &buffer);

        return buffer.toOwnedSlice(allocator);
    }

    fn formatCounters(self: *PrometheusExporter, allocator: std.mem.Allocator, buffer: *std.ArrayListUnmanaged(u8)) !void {
        const counters = self.metrics.getCounters();
        for (counters) |counter| {
            const help = try std.fmt.allocPrint(allocator, "# HELP {}_{} {}\n", .{
                self.config.namespace,
                counter.name,
                counter.name,
            });
            defer allocator.free(help);
            try buffer.appendSlice(allocator, help);

            const type_line = try std.fmt.allocPrint(allocator, "# TYPE {}_{} counter\n", .{
                self.config.namespace,
                counter.name,
            });
            defer allocator.free(type_line);
            try buffer.appendSlice(allocator, type_line);

            const value = try std.fmt.allocPrint(allocator, "{}_{{{s}}} {d}\n", .{
                self.config.namespace,
                counter.name,
                counter.labels,
                counter.value,
            });
            defer allocator.free(value);
            try buffer.appendSlice(allocator, value);
        }
    }

    fn formatHistograms(self: *PrometheusExporter, allocator: std.mem.Allocator, buffer: *std.ArrayListUnmanaged(u8)) !void {
        const histograms = self.metrics.getHistograms();
        for (histograms) |hist| {
            const help = try std.fmt.allocPrint(allocator, "# HELP {}_{} {}\n", .{
                self.config.namespace,
                hist.name,
                hist.name,
            });
            defer allocator.free(help);
            try buffer.appendSlice(allocator, help);

            const type_line = try std.fmt.allocPrint(allocator, "# TYPE {}_{} histogram\n", .{
                self.config.namespace,
                hist.name,
            });
            defer allocator.free(type_line);
            try buffer.appendSlice(allocator, type_line);

            const sum = try std.fmt.allocPrint(allocator, "{}_{}_sum{{{s}}} {d}\n", .{
                self.config.namespace,
                hist.name,
                hist.labels,
                hist.sum,
            });
            defer allocator.free(sum);
            try buffer.appendSlice(allocator, sum);

            const count = try std.fmt.allocPrint(allocator, "{}_{}_count{{{s}}} {d}\n", .{
                self.config.namespace,
                hist.name,
                hist.labels,
                hist.count,
            });
            defer allocator.free(count);
            try buffer.appendSlice(allocator, count);

            for (hist.buckets, 0..) |bound, i| {
                const bucket_line = try std.fmt.allocPrint(allocator, "{}_{}_bucket{{{s},le=\"{d}\"}} {d}\n", .{
                    self.config.namespace,
                    hist.name,
                    hist.labels,
                    bound,
                    hist.bucket_counts[i],
                });
                defer allocator.free(bucket_line);
                try buffer.appendSlice(allocator, bucket_line);
            }

            try buffer.appendSlice(allocator, "\n");
        }
    }

    fn runServer(self: *PrometheusExporter) void {
        _ = self;
    }
};

pub const PrometheusFormatter = struct {
    allocator: std.mem.Allocator,
    namespace: []const u8,

    pub fn init(allocator: std.mem.Allocator, namespace: []const u8) PrometheusFormatter {
        return .{
            .allocator = allocator,
            .namespace = namespace,
        };
    }

    pub fn formatCounter(
        self: *PrometheusFormatter,
        name: []const u8,
        value: u64,
        labels: []const u8,
    ) ![]const u8 {
        return std.fmt.allocPrint(self.allocator, "{}_{}{{{s}}} {d}\n", .{
            self.namespace,
            name,
            if (labels.len > 0) labels else "",
            value,
        });
    }

    pub fn formatHistogram(
        self: *PrometheusFormatter,
        name: []const u8,
        sum: u64,
        count: u64,
        buckets: []const u64,
        bounds: []const u64,
        labels: []const u8,
    ) ![]const u8 {
        var result = std.ArrayListUnmanaged(u8).empty;
        errdefer result.deinit(self.allocator);

        const sum_line = try std.fmt.allocPrint(self.allocator, "{}_{}_sum{{{s}}} {d}\n", .{
            self.namespace,
            name,
            if (labels.len > 0) labels else "",
            sum,
        });
        defer self.allocator.free(sum_line);
        try result.appendSlice(self.allocator, sum_line);

        const count_line = try std.fmt.allocPrint(self.allocator, "{}_{}_count{{{s}}} {d}\n", .{
            self.namespace,
            name,
            if (labels.len > 0) labels else "",
            count,
        });
        defer self.allocator.free(count_line);
        try result.appendSlice(self.allocator, count_line);

        for (buckets, bounds) |bucket_count, bound| {
            const line = try std.fmt.allocPrint(self.allocator, "{}_{}_bucket{{{s},le=\"{d}\"}} {d}\n", .{
                self.namespace,
                name,
                if (labels.len > 0) labels else "",
                bound,
                bucket_count,
            });
            defer self.allocator.free(line);
            try result.appendSlice(self.allocator, line);
        }

        return result.toOwnedSlice(self.allocator);
    }
};

pub fn generateMetricsOutput(
    allocator: std.mem.Allocator,
    namespace: []const u8,
    counters: []const struct { name: []const u8, value: u64 },
    histograms: []const struct { name: []const u8, sum: u64, count: u64, buckets: []const u64, bounds: []const u64 },
) ![]const u8 {
    var output = std.ArrayListUnmanaged(u8).empty;
    errdefer output.deinit(allocator);

    try output.appendSlice(allocator, "# Prometheus metrics exported by ABI framework\n\n");

    for (counters) |counter| {
        const line = try std.fmt.allocPrint(allocator, "{}_{} {d}\n", .{
            namespace,
            counter.name,
            counter.value,
        });
        defer allocator.free(line);
        try output.appendSlice(allocator, line);
    }

    for (histograms) |hist| {
        const sum_line = try std.fmt.allocPrint(allocator, "{}_{}_sum {d}\n", .{
            namespace,
            hist.name,
            hist.sum,
        });
        defer allocator.free(sum_line);
        try output.appendSlice(allocator, sum_line);

        const count_line = try std.fmt.allocPrint(allocator, "{}_{}_count {d}\n", .{
            namespace,
            hist.name,
            hist.count,
        });
        defer allocator.free(count_line);
        try output.appendSlice(allocator, count_line);

        for (hist.buckets, hist.bounds) |bucket_count, bound| {
            const bucket_line = try std.fmt.allocPrint(allocator, "{}_{}_bucket{{le=\"{d}\"}} {d}\n", .{
                namespace,
                hist.name,
                bound,
                bucket_count,
            });
            defer allocator.free(bucket_line);
            try output.appendSlice(allocator, bucket_line);
        }

        try output.appendSlice(allocator, "\n");
    }

    return output.toOwnedSlice(allocator);
}

test "prometheus formatter counter" {
    const allocator = std.testing.allocator;
    var formatter = PrometheusFormatter.init(allocator, "abi");

    const output = try formatter.formatCounter("requests_total", 42, "method=\"get\"");
    defer allocator.free(output);

    try std.testing.expect(std.mem.startsWith(u8, output, "abi_requests_total"));
    try std.testing.expect(std.mem.containsAtLeast(u8, output, 1, "42"));
}

test "generate metrics output" {
    const allocator = std.testing.allocator;

    const counters = [_]struct { name: []const u8, value: u64 }{
        .{ .name = "requests_total", .value = 100 },
        .{ .name = "errors_total", .value = 5 },
    };

    const histograms = [_]struct { name: []const u8, sum: u64, count: u64, buckets: []const u64, bounds: []const u64 }{
        .{
            .name = "latency_ms",
            .sum = 5000,
            .count = 100,
            .buckets = &.{ 50, 80, 95, 100 },
            .bounds = &.{ 10, 50, 100, 500 },
        },
    };

    const output = try generateMetricsOutput(allocator, "abi", &counters, &histograms);
    defer allocator.free(output);

    try std.testing.expect(std.mem.containsAtLeast(u8, output, 1, "abi_requests_total"));
    try std.testing.expect(std.mem.containsAtLeast(u8, output, 1, "abi_latency_ms_sum"));
}
