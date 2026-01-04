//! StatsD metrics exporter for observability.
//!
//! Provides functionality to send metrics to a StatsD server
//! including counters, gauges, timers, and histograms.

const std = @import("std");

pub const StatsDError = error{
    ConnectionFailed,
    SendFailed,
    InvalidMetricName,
};

pub const StatsDConfig = struct {
    host: []const u8 = "localhost",
    port: u16 = 8125,
    prefix: []const u8 = "",
    sample_rate: f64 = 1.0,
    max_packet_size: usize = 1400,
    buffer_size: usize = 65536,
    flush_interval_ms: u32 = 1000,
};

pub const MetricValue = union(enum) {
    counter: f64,
    gauge: f64,
    timer: f64,
    set: []const u8,
};

pub const StatsDClient = struct {
    allocator: std.mem.Allocator,
    config: StatsDConfig,
    socket: ?std.net.Stream = null,
    buffer: std.ArrayListUnmanaged(u8),
    connected: bool,

    pub fn init(allocator: std.mem.Allocator, config: StatsDConfig) !StatsDClient {
        const host_copy = try allocator.dupe(u8, config.host);
        errdefer allocator.free(host_copy);

        const prefix_copy = try allocator.dupe(u8, config.prefix);
        errdefer allocator.free(prefix_copy);

        return .{
            .allocator = allocator,
            .config = config,
            .socket = null,
            .buffer = std.ArrayListUnmanaged(u8).init(allocator),
            .connected = false,
        };
    }

    pub fn deinit(self: *StatsDClient) void {
        self.disconnect();
        self.buffer.deinit(self.allocator);
        self.allocator.free(self.config.host);
        self.allocator.free(self.config.prefix);
        self.* = undefined;
    }

    pub fn connect(self: *StatsDClient) !void {
        if (self.connected) return;

        const address = try std.net.Address.parseIp4(self.config.host, self.config.port);
        self.socket = std.net.tcpConnectToAddress(address) catch {
            return StatsDError.ConnectionFailed;
        };
        self.connected = true;
    }

    pub fn disconnect(self: *StatsDClient) void {
        if (self.socket) |*socket| {
            socket.close();
            self.socket = null;
        }
        self.connected = false;
    }

    pub fn isConnected(self: *const StatsDClient) bool {
        return self.connected;
    }

    pub fn increment(self: *StatsDClient, name: []const u8, value: f64, tags: []const []const u8) !void {
        try self.send(name, value, "c", tags);
    }

    pub fn decrement(self: *StatsDClient, name: []const u8, value: f64, tags: []const []const u8) !void {
        try self.send(name, -value, "c", tags);
    }

    pub fn gauge(self: *StatsDClient, name: []const u8, value: f64, tags: []const []const u8) !void {
        try self.send(name, value, "g", tags);
    }

    pub fn timing(self: *StatsDClient, name: []const u8, value: f64, tags: []const []const u8) !void {
        try self.send(name, value, "ms", tags);
    }

    pub fn histogram(self: *StatsDClient, name: []const u8, value: f64, tags: []const []const u8) !void {
        try self.send(name, value, "h", tags);
    }

    pub fn set(self: *StatsDClient, name: []const u8, value: []const u8, tags: []const []const u8) !void {
        try self.sendSet(name, value, tags);
    }

    fn send(self: *StatsDClient, name: []const u8, value: f64, type_: []const u8, tags: []const []const u8) !void {
        const prefixed = if (self.config.prefix.len > 0)
            std.fmt.allocPrint(self.allocator, "{s}.{s}", .{ self.config.prefix, name }) catch return
        else
            name;

        if (prefixed.len != name.len) {
            defer self.allocator.free(prefixed);
        }

        const metric_line = try std.fmt.allocPrint(self.allocator, "{s}:{d}|{s}", .{
            prefixed,
            value,
            type_,
        });
        defer self.allocator.free(metric_line);

        try self.sendPacket(metric_line, tags);
    }

    fn sendSet(self: *StatsDClient, name: []const u8, value: []const u8, tags: []const []const u8) !void {
        const prefixed = if (self.config.prefix.len > 0)
            std.fmt.allocPrint(self.allocator, "{s}.{s}", .{ self.config.prefix, name }) catch return
        else
            name;

        if (prefixed.len != name.len) {
            defer self.allocator.free(prefixed);
        }

        const metric_line = try std.fmt.allocPrint(self.allocator, "{s}:{s}|s", .{
            prefixed,
            value,
        });
        defer self.allocator.free(metric_line);

        try self.sendPacket(metric_line, tags);
    }

    fn sendPacket(self: *StatsDClient, metric: []const u8, tags: []const []const u8) !void {
        var line = std.ArrayListUnmanaged(u8).empty;
        defer line.deinit(self.allocator);

        try line.appendSlice(self.allocator, metric);

        for (tags) |tag| {
            try line.appendSlice(self.allocator, "|@");
            try line.appendSlice(self.allocator, tag);
        }

        try line.append('\n');

        if (self.socket) |*socket| {
            _ = socket.write(line.items) catch {
                self.connected = false;
                return StatsDError.SendFailed;
            };
        } else {
            try self.buffer.appendSlice(self.allocator, line.items);
        }
    }

    pub fn flush(self: *StatsDClient) !void {
        if (!self.connected or self.buffer.items.len == 0) return;

        if (self.socket) |*socket| {
            _ = socket.write(self.buffer.items) catch {
                self.connected = false;
                return StatsDError.SendFailed;
            };
            self.buffer.clearRetainingCapacity();
        }
    }

    pub fn bufferSize(self: *const StatsDClient) usize {
        return self.buffer.items.len;
    }

    pub fn needsFlush(self: *const StatsDClient) bool {
        return self.buffer.items.len >= self.config.max_packet_size;
    }
};

pub fn createClient(allocator: std.mem.Allocator, host: []const u8, port: u16) !StatsDClient {
    return StatsDClient.init(allocator, .{
        .host = host,
        .port = port,
    });
}

pub fn createPrefixedClient(allocator: std.mem.Allocator, prefix: []const u8) !StatsDClient {
    return StatsDClient.init(allocator, .{
        .prefix = prefix,
    });
}

test "statsd client initialization" {
    const allocator = std.testing.allocator;
    var client = try StatsDClient.init(allocator, .{});
    defer client.deinit();

    try std.testing.expect(!client.isConnected());
    try std.testing.expectEqual(@as(usize, 0), client.bufferSize());
}

test "statsd client buffer management" {
    const allocator = std.testing.allocator;
    var client = try StatsDClient.init(allocator, .{});
    defer client.deinit();

    try client.gauge("test_metric", 42.0, &.{});
    try std.testing.expect(client.bufferSize() > 0);
}

test "statsd metric formatting" {
    const allocator = std.testing.allocator;
    var client = try StatsDClient.init(allocator, .{
        .prefix = "abi",
    });
    defer client.deinit();

    try client.increment("requests", 1, &.{"method:get"});
    try std.testing.expect(client.bufferSize() > 0);
}

test "statsd gauge formatting" {
    const allocator = std.testing.allocator;
    var client = try StatsDClient.init(allocator, .{
        .prefix = "app",
    });
    defer client.deinit();

    try client.gauge("memory_usage", 1024.5, &.{});
    try std.testing.expect(client.bufferSize() > 0);
}

test "statsd timing formatting" {
    const allocator = std.testing.allocator;
    var client = try StatsDClient.init(allocator, .{});
    defer client.deinit();

    try client.timing("latency", 150.5, &.{});
    try std.testing.expect(client.bufferSize() > 0);
}
