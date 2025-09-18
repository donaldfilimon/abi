//! Prometheus Metrics Export for WDBX
//!
//! Provides comprehensive metrics collection and export in Prometheus format including:
//! - Database operation metrics
//! - HTTP API metrics
//! - System resource metrics
//! - Custom business metrics
//! - Performance counters

const std = @import("std");

/// Prometheus metric types
pub const MetricType = enum {
    counter,
    gauge,
    histogram,
    summary,
};

/// Individual metric definition
pub const Metric = struct {
    name: []const u8,
    help: []const u8,
    type: MetricType,
    value: f64,
    labels: std.StringHashMap([]const u8),

    pub fn init(allocator: std.mem.Allocator, name: []const u8, help: []const u8, metric_type: MetricType) !Metric {
        return Metric{
            .name = try allocator.dupe(u8, name),
            .help = try allocator.dupe(u8, help),
            .type = metric_type,
            .value = 0.0,
            .labels = std.StringHashMap([]const u8).init(allocator),
        };
    }

    pub fn deinit(self: *Metric, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.help);

        var iter = self.labels.iterator();
        while (iter.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        self.labels.deinit();
    }

    pub fn addLabel(self: *Metric, allocator: std.mem.Allocator, key: []const u8, value: []const u8) !void {
        try self.labels.put(try allocator.dupe(u8, key), try allocator.dupe(u8, value));
    }

    pub fn setValue(self: *Metric, value: f64) void {
        self.value = value;
    }

    pub fn increment(self: *Metric) void {
        self.value += 1.0;
    }

    pub fn add(self: *Metric, value: f64) void {
        self.value += value;
    }
};

/// Metrics collector and registry
pub const MetricsCollector = struct {
    allocator: std.mem.Allocator,
    metrics: std.StringHashMap(Metric),

    // Built-in metrics
    db_operations_total: *Metric,
    db_operation_duration_seconds: *Metric,
    db_vectors_stored: *Metric,
    db_searches_total: *Metric,
    db_compression_ratio: *Metric,

    http_requests_total: *Metric,
    http_request_duration_seconds: *Metric,
    http_response_size_bytes: *Metric,

    system_cpu_usage_percent: *Metric,
    system_memory_usage_bytes: *Metric,
    system_memory_available_bytes: *Metric,
    system_disk_usage_bytes: *Metric,

    process_cpu_seconds_total: *Metric,
    process_memory_bytes: *Metric,
    process_threads: *Metric,
    process_start_time_seconds: *Metric,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        self.* = .{
            .allocator = allocator,
            .metrics = std.StringHashMap(Metric).init(allocator),
            .db_operations_total = undefined,
            .db_operation_duration_seconds = undefined,
            .db_vectors_stored = undefined,
            .db_searches_total = undefined,
            .db_compression_ratio = undefined,
            .http_requests_total = undefined,
            .http_request_duration_seconds = undefined,
            .http_response_size_bytes = undefined,
            .system_cpu_usage_percent = undefined,
            .system_memory_usage_bytes = undefined,
            .system_memory_available_bytes = undefined,
            .system_disk_usage_bytes = undefined,
            .process_cpu_seconds_total = undefined,
            .process_memory_bytes = undefined,
            .process_threads = undefined,
            .process_start_time_seconds = undefined,
        };

        // Initialize built-in metrics
        try self.initBuiltinMetrics();

        return self;
    }

    pub fn deinit(self: *Self) void {
        var iter = self.metrics.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
        }
        self.metrics.deinit();
        self.allocator.destroy(self);
    }

    fn initBuiltinMetrics(self: *Self) !void {
        // Database metrics
        self.db_operations_total = try self.registerMetric("wdbx_db_operations_total", "Total number of database operations", .counter);
        self.db_operation_duration_seconds = try self.registerMetric("wdbx_db_operation_duration_seconds", "Duration of database operations", .histogram);
        self.db_vectors_stored = try self.registerMetric("wdbx_db_vectors_stored", "Number of vectors currently stored", .gauge);
        self.db_searches_total = try self.registerMetric("wdbx_db_searches_total", "Total number of search operations", .counter);
        self.db_compression_ratio = try self.registerMetric("wdbx_db_compression_ratio", "Database compression ratio", .gauge);

        // HTTP metrics
        self.http_requests_total = try self.registerMetric("wdbx_http_requests_total", "Total number of HTTP requests", .counter);
        self.http_request_duration_seconds = try self.registerMetric("wdbx_http_request_duration_seconds", "Duration of HTTP requests", .histogram);
        self.http_response_size_bytes = try self.registerMetric("wdbx_http_response_size_bytes", "Size of HTTP responses", .histogram);

        // System metrics
        self.system_cpu_usage_percent = try self.registerMetric("wdbx_system_cpu_usage_percent", "System CPU usage percentage", .gauge);
        self.system_memory_usage_bytes = try self.registerMetric("wdbx_system_memory_usage_bytes", "System memory usage in bytes", .gauge);
        self.system_memory_available_bytes = try self.registerMetric("wdbx_system_memory_available_bytes", "Available system memory in bytes", .gauge);
        self.system_disk_usage_bytes = try self.registerMetric("wdbx_system_disk_usage_bytes", "Disk usage in bytes", .gauge);

        // Process metrics
        self.process_cpu_seconds_total = try self.registerMetric("wdbx_process_cpu_seconds_total", "Total CPU time consumed by process", .counter);
        self.process_memory_bytes = try self.registerMetric("wdbx_process_memory_bytes", "Process memory usage in bytes", .gauge);
        self.process_threads = try self.registerMetric("wdbx_process_threads", "Number of process threads", .gauge);
        self.process_start_time_seconds = try self.registerMetric("wdbx_process_start_time_seconds", "Process start time in Unix seconds", .gauge);

        // Set initial values
        self.process_start_time_seconds.setValue(@as(f64, @floatFromInt(std.time.timestamp())));
    }

    pub fn registerMetric(self: *Self, name: []const u8, help: []const u8, metric_type: MetricType) !*Metric {
        var metric = try Metric.init(self.allocator, name, help, metric_type);
        const result = try self.metrics.getOrPut(metric.name);
        if (result.found_existing) {
            metric.deinit(self.allocator);
            return result.value_ptr;
        } else {
            result.value_ptr.* = metric;
            return result.value_ptr;
        }
    }

    pub fn getMetric(self: *Self, name: []const u8) ?*Metric {
        return self.metrics.getPtr(name);
    }

    // Database metric updates
    pub fn recordDatabaseOperation(self: *Self, operation: []const u8, duration_ns: u64) !void {
        self.db_operations_total.increment();
        try self.db_operations_total.addLabel(self.allocator, "operation", operation);

        const duration_seconds = @as(f64, @floatFromInt(duration_ns)) / 1_000_000_000.0;
        self.db_operation_duration_seconds.setValue(duration_seconds);
        try self.db_operation_duration_seconds.addLabel(self.allocator, "operation", operation);
    }

    pub fn updateDatabaseStats(self: *Self, vectors_stored: u64, compression_ratio: f64) void {
        self.db_vectors_stored.setValue(@as(f64, @floatFromInt(vectors_stored)));
        self.db_compression_ratio.setValue(compression_ratio);
    }

    pub fn recordSearch(self: *Self, duration_ns: u64, results_count: usize) !void {
        self.db_searches_total.increment();

        const duration_seconds = @as(f64, @floatFromInt(duration_ns)) / 1_000_000_000.0;
        self.db_operation_duration_seconds.setValue(duration_seconds);
        try self.db_operation_duration_seconds.addLabel(self.allocator, "operation", "search");
        try self.db_operation_duration_seconds.addLabel(self.allocator, "results", try std.fmt.allocPrint(self.allocator, "{d}", .{results_count}));
    }

    // HTTP metric updates
    pub fn recordHttpRequest(self: *Self, method: []const u8, path: []const u8, status_code: u16, duration_ns: u64, response_size: usize) !void {
        self.http_requests_total.increment();
        try self.http_requests_total.addLabel(self.allocator, "method", method);
        try self.http_requests_total.addLabel(self.allocator, "path", path);
        try self.http_requests_total.addLabel(self.allocator, "status", try std.fmt.allocPrint(self.allocator, "{d}", .{status_code}));

        const duration_seconds = @as(f64, @floatFromInt(duration_ns)) / 1_000_000_000.0;
        self.http_request_duration_seconds.setValue(duration_seconds);
        try self.http_request_duration_seconds.addLabel(self.allocator, "method", method);
        try self.http_request_duration_seconds.addLabel(self.allocator, "path", path);

        self.http_response_size_bytes.setValue(@as(f64, @floatFromInt(response_size)));
        try self.http_response_size_bytes.addLabel(self.allocator, "path", path);
    }

    // System metric updates
    pub fn updateSystemMetrics(self: *Self, cpu_percent: f64, memory_used: u64, memory_available: u64, disk_used: u64) void {
        self.system_cpu_usage_percent.setValue(cpu_percent);
        self.system_memory_usage_bytes.setValue(@as(f64, @floatFromInt(memory_used)));
        self.system_memory_available_bytes.setValue(@as(f64, @floatFromInt(memory_available)));
        self.system_disk_usage_bytes.setValue(@as(f64, @floatFromInt(disk_used)));
    }

    // Process metric updates
    pub fn updateProcessMetrics(self: *Self, cpu_seconds: f64, memory_bytes: u64, thread_count: u32) void {
        self.process_cpu_seconds_total.setValue(cpu_seconds);
        self.process_memory_bytes.setValue(@as(f64, @floatFromInt(memory_bytes)));
        self.process_threads.setValue(@as(f64, @floatFromInt(thread_count)));
    }

    /// Export metrics in Prometheus format
    pub fn exportPrometheusFormat(self: *Self, allocator: std.mem.Allocator) ![]const u8 {
        var output = std.ArrayList(u8).init(allocator);
        defer output.deinit();

        var iter = self.metrics.iterator();
        while (iter.next()) |entry| {
            const metric = entry.value_ptr;

            // Write metric header
            try output.writer().print("# HELP {s} {s}\n", .{ metric.name, metric.help });
            try output.writer().print("# TYPE {s} {s}\n", .{ metric.name, @tagName(metric.type) });

            // Write metric value with labels
            try output.writer().print("{s}", .{metric.name});

            if (metric.labels.count() > 0) {
                try output.writer().print("{{");
                var label_iter = metric.labels.iterator();
                var first = true;
                while (label_iter.next()) |label| {
                    if (!first) try output.writer().print(",");
                    try output.writer().print("{s}=\"{s}\"", .{ label.key_ptr.*, label.value_ptr.* });
                    first = false;
                }
                try output.writer().print("}}");
            }

            try output.writer().print(" {d}\n\n", .{metric.value});
        }

        return try output.toOwnedSlice();
    }
};

/// Prometheus HTTP server for metrics export
pub const PrometheusServer = struct {
    allocator: std.mem.Allocator,
    metrics_collector: *MetricsCollector,
    server: ?std.net.Server,
    host: []const u8,
    port: u16,
    path: []const u8,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, metrics_collector: *MetricsCollector, host: []const u8, port: u16, path: []const u8) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        self.* = .{
            .allocator = allocator,
            .metrics_collector = metrics_collector,
            .server = null,
            .host = try allocator.dupe(u8, host),
            .port = port,
            .path = try allocator.dupe(u8, path),
        };

        return self;
    }

    pub fn deinit(self: *Self) void {
        if (self.server) |*server| {
            server.deinit();
        }
        self.allocator.free(self.host);
        self.allocator.free(self.path);
        self.allocator.destroy(self);
    }

    pub fn start(self: *Self) !void {
        const address = try std.net.Address.parseIp(self.host, self.port);
        self.server = try address.listen(.{ .reuse_address = true });

        std.debug.print("Prometheus metrics server started on {s}:{} (path: {s})\n", .{ self.host, self.port, self.path });

        while (true) {
            const connection = self.server.?.accept() catch |err| {
                std.debug.print("Failed to accept connection: {any}\n", .{err});
                continue;
            };

            self.handleConnection(connection) catch |err| {
                std.debug.print("Connection handling error: {any}\n", .{err});
            };
        }
    }

    fn handleConnection(self: *Self, connection: std.net.Server.Connection) !void {
        defer connection.stream.close();

        var buffer: [4096]u8 = undefined;
        const bytes_read = try connection.stream.read(&buffer);
        if (bytes_read == 0) return;

        const request = buffer[0..bytes_read];

        // Simple HTTP request parsing
        var lines = std.mem.splitScalar(u8, request, '\n');
        const first_line = lines.next() orelse return;
        var parts = std.mem.splitScalar(u8, first_line, ' ');
        _ = parts.next(); // method
        const path = parts.next() orelse return;

        if (std.mem.eql(u8, path, self.path)) {
            try self.serveMetrics(connection);
        } else {
            try self.serveNotFound(connection);
        }
    }

    fn serveMetrics(self: *Self, connection: std.net.Server.Connection) !void {
        const metrics_text = try self.metrics_collector.exportPrometheusFormat(self.allocator);
        defer self.allocator.free(metrics_text);

        const response = try std.fmt.allocPrint(self.allocator, "HTTP/1.1 200 OK\r\n" ++
            "Content-Type: text/plain; version=0.0.4; charset=utf-8\r\n" ++
            "Content-Length: {d}\r\n" ++
            "\r\n" ++
            "{s}", .{ metrics_text.len, metrics_text });
        defer self.allocator.free(response);

        _ = try connection.stream.write(response);
    }

    fn serveNotFound(self: *Self, connection: std.net.Server.Connection) !void {
        const body = try std.fmt.allocPrint(self.allocator, "Metrics available at {s}", .{self.path});
        defer self.allocator.free(body);

        const response = try std.fmt.allocPrint(self.allocator, "HTTP/1.1 404 Not Found\r\n" ++
            "Content-Type: text/plain\r\n" ++
            "Content-Length: {d}\r\n" ++
            "\r\n" ++
            "{s}", .{ body.len, body });
        defer self.allocator.free(response);

        _ = try connection.stream.write(response);
    }
};
