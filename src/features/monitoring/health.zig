//! Health Monitoring for WDBX
//!
//! Comprehensive health checking and system diagnostics including:
//! - Database health checks
//! - HTTP server health monitoring
//! - System resource health tracking
//! - Automated health reporting
//! - Health status aggregation

const std = @import("std");

/// Health status levels
pub const HealthStatus = enum {
    healthy,
    warning,
    critical,
    unknown,

    pub fn toString(self: HealthStatus) []const u8 {
        return switch (self) {
            .healthy => "healthy",
            .warning => "warning",
            .critical => "critical",
            .unknown => "unknown",
        };
    }
};

/// Individual health check result
pub const HealthCheck = struct {
    name: []const u8,
    status: HealthStatus,
    message: []const u8,
    timestamp: i64,
    response_time_ms: u64,
    metadata: std.StringHashMap([]const u8),

    pub fn init(allocator: std.mem.Allocator, name: []const u8, status: HealthStatus, message: []const u8, response_time_ms: u64) !HealthCheck {
        return HealthCheck{
            .name = try allocator.dupe(u8, name),
            .status = status,
            .message = try allocator.dupe(u8, message),
            .timestamp = std.time.timestamp(),
            .response_time_ms = response_time_ms,
            .metadata = std.StringHashMap([]const u8).init(allocator),
        };
    }

    pub fn deinit(self: *HealthCheck, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.message);

        var iter = self.metadata.iterator();
        while (iter.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        self.metadata.deinit();
    }

    pub fn addMetadata(self: *HealthCheck, allocator: std.mem.Allocator, key: []const u8, value: []const u8) !void {
        try self.metadata.put(try allocator.dupe(u8, key), try allocator.dupe(u8, value));
    }
};

/// Health checker configuration
pub const HealthConfig = struct {
    check_interval_ms: u32 = 10000, // 10 seconds
    timeout_ms: u32 = 5000, // 5 seconds per check
    max_failures: u32 = 3, // Failures before marking as critical
    enable_database_checks: bool = true,
    enable_http_checks: bool = true,
    enable_system_checks: bool = true,
    enable_memory_checks: bool = true,
    enable_disk_checks: bool = true,
    memory_warning_threshold: f64 = 80.0, // 80% memory usage
    memory_critical_threshold: f64 = 95.0, // 95% memory usage
    disk_warning_threshold: f64 = 85.0, // 85% disk usage
    disk_critical_threshold: f64 = 95.0, // 95% disk usage
    cpu_warning_threshold: f64 = 90.0, // 90% CPU usage
    cpu_critical_threshold: f64 = 98.0, // 98% CPU usage
};

/// Overall system health status
pub const SystemHealth = struct {
    overall_status: HealthStatus,
    checks: std.ArrayList(HealthCheck),
    last_updated: i64,
    uptime_seconds: u64,

    // Aggregated metrics
    total_checks: u32,
    healthy_checks: u32,
    warning_checks: u32,
    critical_checks: u32,
    unknown_checks: u32,

    pub fn init(allocator: std.mem.Allocator) SystemHealth {
        return SystemHealth{
            .overall_status = .unknown,
            .checks = std.ArrayList(HealthCheck){},
            .last_updated = std.time.timestamp(),
            .uptime_seconds = 0,
            .total_checks = 0,
            .healthy_checks = 0,
            .warning_checks = 0,
            .critical_checks = 0,
            .unknown_checks = 0,
        };
    }

    pub fn deinit(self: *SystemHealth) void {
        for (self.checks.items) |*check| {
            check.deinit(self.checks.allocator);
        }
        self.checks.deinit();
    }

    pub fn updateOverallStatus(self: *SystemHealth) void {
        if (self.critical_checks > 0) {
            self.overall_status = .critical;
        } else if (self.warning_checks > 0) {
            self.overall_status = .warning;
        } else if (self.healthy_checks > 0) {
            self.overall_status = .healthy;
        } else {
            self.overall_status = .unknown;
        }

        self.last_updated = std.time.timestamp();
    }

    pub fn addCheck(self: *SystemHealth, check: HealthCheck) !void {
        try self.checks.append(check);

        // Update counters
        self.total_checks += 1;
        switch (check.status) {
            .healthy => self.healthy_checks += 1,
            .warning => self.warning_checks += 1,
            .critical => self.critical_checks += 1,
            .unknown => self.unknown_checks += 1,
        }

        self.updateOverallStatus();
    }
};

/// Comprehensive health checker
pub const HealthChecker = struct {
    allocator: std.mem.Allocator,
    config: HealthConfig,
    system_health: SystemHealth,
    start_time: i64,
    running: std.atomic.Value(bool),
    thread: ?std.Thread,
    failure_counts: std.StringHashMap(u32),

    // Health check callbacks
    health_change_callback: ?*const fn (SystemHealth) void,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, config: HealthConfig) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        self.* = .{
            .allocator = allocator,
            .config = config,
            .system_health = SystemHealth.init(allocator),
            .start_time = std.time.timestamp(),
            .running = std.atomic.Value(bool).init(false),
            .thread = null,
            .failure_counts = std.StringHashMap(u32).init(allocator),
            .health_change_callback = null,
        };

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.stop();
        self.system_health.deinit();

        // Clean up failure counts
        var iter = self.failure_counts.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.failure_counts.deinit();

        self.allocator.destroy(self);
    }

    pub fn start(self: *Self) !void {
        if (self.running.load(.monotonic)) return; // Already running

        self.running.store(true, .monotonic);
        self.thread = try std.Thread.spawn(.{}, healthCheckLoop, .{self});

        std.debug.print("Health checker started (interval: {}ms)\n", .{self.config.check_interval_ms});
    }

    pub fn stop(self: *Self) void {
        self.running.store(false, .monotonic);

        if (self.thread) |thread| {
            thread.join();
            self.thread = null;
        }

        std.debug.print("Health checker stopped\n");
    }

    pub fn setHealthChangeCallback(self: *Self, callback: *const fn (SystemHealth) void) void {
        self.health_change_callback = callback;
    }

    pub fn getCurrentHealth(self: *Self) *const SystemHealth {
        self.system_health.uptime_seconds = @as(u64, @intCast(std.time.timestamp() - self.start_time));
        return &self.system_health;
    }

    fn healthCheckLoop(self: *Self) void {
        while (self.running.load(.monotonic)) {
            self.performHealthChecks() catch |err| {
                std.debug.print("Health check error: {any}\n", .{err});
            };

            // Sleep until next check
            std.time.sleep(self.config.check_interval_ms * std.time.ns_per_ms);
        }
    }

    fn performHealthChecks(self: *Self) !void {
        // Clear previous checks
        for (self.system_health.checks.items) |*check| {
            check.deinit(self.allocator);
        }
        self.system_health.checks.clearRetainingCapacity();

        // Reset counters
        self.system_health.total_checks = 0;
        self.system_health.healthy_checks = 0;
        self.system_health.warning_checks = 0;
        self.system_health.critical_checks = 0;
        self.system_health.unknown_checks = 0;

        // Database health checks
        if (self.config.enable_database_checks) {
            try self.checkDatabaseHealth();
        }

        // HTTP server health checks
        if (self.config.enable_http_checks) {
            try self.checkHttpHealth();
        }

        // System resource checks
        if (self.config.enable_system_checks) {
            try self.checkSystemHealth();
        }

        // Memory checks
        if (self.config.enable_memory_checks) {
            try self.checkMemoryHealth();
        }

        // Disk checks
        if (self.config.enable_disk_checks) {
            try self.checkDiskHealth();
        }

        // Update overall status
        self.system_health.updateOverallStatus();

        // Trigger callback if status changed
        if (self.health_change_callback) |callback| {
            callback(self.system_health);
        }
    }

    fn checkDatabaseHealth(self: *Self) !void {
        const start_time = std.time.milliTimestamp();

        // Simulate database health check
        const is_healthy = true; // Would actually check database connectivity
        const response_time = @as(u64, @intCast(std.time.milliTimestamp() - start_time));

        const status: HealthStatus = if (is_healthy) .healthy else .critical;
        const message = if (is_healthy) "Database is responsive" else "Database connection failed";

        var check = try HealthCheck.init(self.allocator, "database", status, message, response_time);
        try check.addMetadata(self.allocator, "type", "connectivity");
        try check.addMetadata(self.allocator, "timeout_ms", try std.fmt.allocPrint(self.allocator, "{d}", .{self.config.timeout_ms}));

        try self.system_health.addCheck(check);
    }

    fn checkHttpHealth(self: *Self) !void {
        const start_time = std.time.milliTimestamp();

        // Simulate HTTP server health check
        const is_responsive = true; // Would actually make HTTP request
        const response_time = @as(u64, @intCast(std.time.milliTimestamp() - start_time));

        const status: HealthStatus = if (is_responsive) .healthy else .critical;
        const message = if (is_responsive) "HTTP server is responsive" else "HTTP server not responding";

        var check = try HealthCheck.init(self.allocator, "http_server", status, message, response_time);
        try check.addMetadata(self.allocator, "type", "endpoint");
        try check.addMetadata(self.allocator, "expected_response_time_ms", "< 100");

        try self.system_health.addCheck(check);
    }

    fn checkSystemHealth(self: *Self) !void {
        const start_time = std.time.milliTimestamp();

        // Simulate system health check - would check CPU, load average, etc.
        const cpu_usage = 45.5; // Would get actual CPU usage
        const response_time = @as(u64, @intCast(std.time.milliTimestamp() - start_time));

        const status: HealthStatus = if (cpu_usage < self.config.cpu_warning_threshold)
            .healthy
        else if (cpu_usage < self.config.cpu_critical_threshold)
            .warning
        else
            .critical;

        const message = try std.fmt.allocPrint(self.allocator, "CPU usage: {d:.1}%", .{cpu_usage});
        defer self.allocator.free(message);

        var check = try HealthCheck.init(self.allocator, "system_cpu", status, message, response_time);
        try check.addMetadata(self.allocator, "cpu_usage", try std.fmt.allocPrint(self.allocator, "{d:.1}", .{cpu_usage}));
        try check.addMetadata(self.allocator, "warning_threshold", try std.fmt.allocPrint(self.allocator, "{d:.1}", .{self.config.cpu_warning_threshold}));

        try self.system_health.addCheck(check);
    }

    fn checkMemoryHealth(self: *Self) !void {
        const start_time = std.time.milliTimestamp();

        // Simulate memory health check
        const memory_usage = 72.3; // Would get actual memory usage percentage
        const response_time = @as(u64, @intCast(std.time.milliTimestamp() - start_time));

        const status: HealthStatus = if (memory_usage < self.config.memory_warning_threshold)
            .healthy
        else if (memory_usage < self.config.memory_critical_threshold)
            .warning
        else
            .critical;

        const message = try std.fmt.allocPrint(self.allocator, "Memory usage: {d:.1}%", .{memory_usage});
        defer self.allocator.free(message);

        var check = try HealthCheck.init(self.allocator, "system_memory", status, message, response_time);
        try check.addMetadata(self.allocator, "memory_usage", try std.fmt.allocPrint(self.allocator, "{d:.1}", .{memory_usage}));
        try check.addMetadata(self.allocator, "warning_threshold", try std.fmt.allocPrint(self.allocator, "{d:.1}", .{self.config.memory_warning_threshold}));

        try self.system_health.addCheck(check);
    }

    fn checkDiskHealth(self: *Self) !void {
        const start_time = std.time.milliTimestamp();

        // Simulate disk health check
        const disk_usage = 58.7; // Would get actual disk usage percentage
        const response_time = @as(u64, @intCast(std.time.milliTimestamp() - start_time));

        const status: HealthStatus = if (disk_usage < self.config.disk_warning_threshold)
            .healthy
        else if (disk_usage < self.config.disk_critical_threshold)
            .warning
        else
            .critical;

        const message = try std.fmt.allocPrint(self.allocator, "Disk usage: {d:.1}%", .{disk_usage});
        defer self.allocator.free(message);

        var check = try HealthCheck.init(self.allocator, "system_disk", status, message, response_time);
        try check.addMetadata(self.allocator, "disk_usage", try std.fmt.allocPrint(self.allocator, "{d:.1}", .{disk_usage}));
        try check.addMetadata(self.allocator, "warning_threshold", try std.fmt.allocPrint(self.allocator, "{d:.1}", .{self.config.disk_warning_threshold}));

        try self.system_health.addCheck(check);
    }

    /// Export health status as JSON
    pub fn exportHealthStatus(self: *Self, allocator: std.mem.Allocator) ![]const u8 {
        const health = self.getCurrentHealth();

        // Build checks JSON array
        var checks_json = try std.ArrayList(u8).initCapacity(allocator, 0);
        defer checks_json.deinit(allocator);

        try checks_json.appendSlice(allocator, "[");
        for (health.checks.items, 0..) |check, i| {
            if (i > 0) try checks_json.appendSlice(allocator, ",");

            // Build metadata JSON
            var metadata_json = try std.ArrayList(u8).initCapacity(allocator, 0);
            defer metadata_json.deinit(allocator);

            try metadata_json.appendSlice(allocator, "{");
            var iter = check.metadata.iterator();
            var first = true;
            while (iter.next()) |entry| {
                if (!first) try metadata_json.appendSlice(allocator, ",");
                const meta_item = try std.fmt.allocPrint(allocator, "\"{s}\":\"{s}\"", .{ entry.key_ptr.*, entry.value_ptr.* });
                defer allocator.free(meta_item);
                try metadata_json.appendSlice(allocator, meta_item);
                first = false;
            }
            try metadata_json.appendSlice(allocator, "}");

            const check_item = try std.fmt.allocPrint(allocator, "{{\"name\":\"{s}\",\"status\":\"{s}\",\"message\":\"{s}\",\"timestamp\":{d},\"response_time_ms\":{d},\"metadata\":{s}}}", .{ check.name, check.status.toString(), check.message, check.timestamp, check.response_time_ms, metadata_json.items });
            defer allocator.free(check_item);
            try checks_json.appendSlice(allocator, check_item);
        }
        try checks_json.appendSlice(allocator, "]");

        return try std.fmt.allocPrint(allocator,
            \\{{"health":{{"overall_status":"{s}","last_updated":{d},"uptime_seconds":{d},"summary":{{"total_checks":{d},"healthy":{d},"warning":{d},"critical":{d},"unknown":{d}}},"checks":{s}}}}}
        , .{ health.overall_status.toString(), health.last_updated, health.uptime_seconds, health.total_checks, health.healthy_checks, health.warning_checks, health.critical_checks, health.unknown_checks, checks_json.items });
    }
};
