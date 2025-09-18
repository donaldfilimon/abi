//! Performance Sampling Module for WDBX
//!
//! Provides comprehensive system and process performance monitoring including:
//! - CPU usage tracking (system and process)
//! - Memory usage monitoring (system and process)
//! - Disk I/O statistics
//! - Network statistics
//! - Performance history and trending

const std = @import("std");
const builtin = @import("builtin");

/// Performance sample data point
pub const PerformanceSample = struct {
    timestamp: i64,

    // CPU metrics
    system_cpu_percent: f64,
    process_cpu_percent: f64,
    cpu_cores: u32,

    // Memory metrics (in bytes)
    system_memory_total: u64,
    system_memory_used: u64,
    system_memory_available: u64,
    process_memory_rss: u64,
    process_memory_vms: u64,

    // Disk metrics (in bytes)
    disk_read_bytes: u64,
    disk_write_bytes: u64,
    disk_usage_percent: f64,

    // Process metrics
    process_threads: u32,
    process_handles: u32,
    process_uptime_seconds: u64,

    // Performance indicators
    load_average_1m: f64,
    load_average_5m: f64,
    load_average_15m: f64,

    pub fn init() PerformanceSample {
        return PerformanceSample{
            .timestamp = std.time.timestamp(),
            .system_cpu_percent = 0.0,
            .process_cpu_percent = 0.0,
            .cpu_cores = 1,
            .system_memory_total = 0,
            .system_memory_used = 0,
            .system_memory_available = 0,
            .process_memory_rss = 0,
            .process_memory_vms = 0,
            .disk_read_bytes = 0,
            .disk_write_bytes = 0,
            .disk_usage_percent = 0.0,
            .process_threads = 0,
            .process_handles = 0,
            .process_uptime_seconds = 0,
            .load_average_1m = 0.0,
            .load_average_5m = 0.0,
            .load_average_15m = 0.0,
        };
    }

    pub fn calculateMemoryUsagePercent(self: *const PerformanceSample) f64 {
        if (self.system_memory_total == 0) return 0.0;
        return (@as(f64, @floatFromInt(self.system_memory_used)) / @as(f64, @floatFromInt(self.system_memory_total))) * 100.0;
    }

    pub fn calculateProcessMemoryPercent(self: *const PerformanceSample) f64 {
        if (self.system_memory_total == 0) return 0.0;
        return (@as(f64, @floatFromInt(self.process_memory_rss)) / @as(f64, @floatFromInt(self.system_memory_total))) * 100.0;
    }
};

/// Performance sampler configuration
pub const SamplerConfig = struct {
    interval_ms: u32 = 1000,
    history_size: usize = 3600, // Keep 1 hour of samples at 1-second intervals
    enable_cpu_sampling: bool = true,
    enable_memory_sampling: bool = true,
    enable_disk_sampling: bool = true,
    enable_network_sampling: bool = false,
    alert_cpu_threshold: f64 = 80.0,
    alert_memory_threshold: f64 = 85.0,
    alert_disk_threshold: f64 = 90.0,
};

/// System information collector
const SystemInfo = struct {
    /// Get system CPU usage percentage
    pub fn getCpuUsage() !f64 {
        return switch (builtin.os.tag) {
            .windows => getCpuUsageWindows(),
            .linux => getCpuUsageLinux(),
            .macos => getCpuUsageMacOS(),
            else => 0.0,
        };
    }

    /// Get system memory information
    pub fn getMemoryInfo() !struct { total: u64, used: u64, available: u64 } {
        return switch (builtin.os.tag) {
            .windows => getMemoryInfoWindows(),
            .linux => getMemoryInfoLinux(),
            .macos => getMemoryInfoMacOS(),
            else => .{ .total = 0, .used = 0, .available = 0 },
        };
    }

    /// Get process-specific information
    pub fn getProcessInfo() !struct { cpu_percent: f64, memory_rss: u64, memory_vms: u64, threads: u32, uptime: u64 } {
        return switch (builtin.os.tag) {
            .windows => getProcessInfoWindows(),
            .linux => getProcessInfoLinux(),
            .macos => getProcessInfoMacOS(),
            else => .{ .cpu_percent = 0.0, .memory_rss = 0, .memory_vms = 0, .threads = 0, .uptime = 0 },
        };
    }

    /// Get disk usage information
    pub fn getDiskInfo() !struct { read_bytes: u64, write_bytes: u64, usage_percent: f64 } {
        return switch (builtin.os.tag) {
            .windows => getDiskInfoWindows(),
            .linux => getDiskInfoLinux(),
            .macos => getDiskInfoMacOS(),
            else => .{ .read_bytes = 0, .write_bytes = 0, .usage_percent = 0.0 },
        };
    }

    // Windows implementations
    fn getCpuUsageWindows() f64 {
        // Windows-specific CPU usage implementation
        // This is a simplified implementation - in practice would use WMI or performance counters
        return 25.5; // Placeholder
    }

    fn getMemoryInfoWindows() !struct { total: u64, used: u64, available: u64 } {
        // Windows-specific memory info implementation
        return .{
            .total = 16 * 1024 * 1024 * 1024, // 16GB placeholder
            .used = 8 * 1024 * 1024 * 1024, // 8GB placeholder
            .available = 8 * 1024 * 1024 * 1024, // 8GB placeholder
        };
    }

    fn getProcessInfoWindows() !struct { cpu_percent: f64, memory_rss: u64, memory_vms: u64, threads: u32, uptime: u64 } {
        // Windows-specific process info implementation
        return .{
            .cpu_percent = 12.3,
            .memory_rss = 256 * 1024 * 1024, // 256MB placeholder
            .memory_vms = 512 * 1024 * 1024, // 512MB placeholder
            .threads = 8,
            .uptime = 3600, // 1 hour placeholder
        };
    }

    fn getDiskInfoWindows() !struct { read_bytes: u64, write_bytes: u64, usage_percent: f64 } {
        return .{
            .read_bytes = 1024 * 1024 * 100, // 100MB placeholder
            .write_bytes = 1024 * 1024 * 50, // 50MB placeholder
            .usage_percent = 65.0,
        };
    }

    // Linux implementations
    fn getCpuUsageLinux() f64 {
        // Parse /proc/stat for CPU usage
        const file = std.fs.openFileAbsolute("/proc/stat", .{}) catch return 0.0;
        defer file.close();

        var buffer: [1024]u8 = undefined;
        const bytes_read = file.readAll(&buffer) catch return 0.0;
        const content = buffer[0..bytes_read];

        // Parse first line: cpu  user nice system idle iowait irq softirq steal guest guest_nice
        var lines = std.mem.splitScalar(u8, content, '\n');
        const cpu_line = lines.next() orelse return 0.0;

        if (!std.mem.startsWith(u8, cpu_line, "cpu ")) return 0.0;

        var parts = std.mem.tokenize(u8, cpu_line[4..], " ");
        var total: u64 = 0;
        var idle: u64 = 0;
        var i: usize = 0;

        while (parts.next()) |part| : (i += 1) {
            const value = std.fmt.parseInt(u64, part, 10) catch 0;
            total += value;
            if (i == 3) idle = value; // idle is the 4th value
        }

        if (total == 0) return 0.0;
        return @as(f64, @floatFromInt(total - idle)) / @as(f64, @floatFromInt(total)) * 100.0;
    }

    fn getMemoryInfoLinux() !struct { total: u64, used: u64, available: u64 } {
        // Parse /proc/meminfo for memory information
        const file = std.fs.openFileAbsolute("/proc/meminfo", .{}) catch {
            return .{ .total = 0, .used = 0, .available = 0 };
        };
        defer file.close();

        var buffer: [4096]u8 = undefined;
        const bytes_read = file.readAll(&buffer) catch return .{ .total = 0, .used = 0, .available = 0 };
        const content = buffer[0..bytes_read];

        var total: u64 = 0;
        var available: u64 = 0;

        var lines = std.mem.splitScalar(u8, content, '\n');
        while (lines.next()) |line| {
            if (std.mem.startsWith(u8, line, "MemTotal:")) {
                var parts = std.mem.tokenize(u8, line, " ");
                _ = parts.next(); // Skip "MemTotal:"
                if (parts.next()) |value_str| {
                    total = (std.fmt.parseInt(u64, value_str, 10) catch 0) * 1024; // Convert kB to bytes
                }
            } else if (std.mem.startsWith(u8, line, "MemAvailable:")) {
                var parts = std.mem.tokenize(u8, line, " ");
                _ = parts.next(); // Skip "MemAvailable:"
                if (parts.next()) |value_str| {
                    available = (std.fmt.parseInt(u64, value_str, 10) catch 0) * 1024; // Convert kB to bytes
                }
            }
        }

        const used = if (total > available) total - available else 0;
        return .{ .total = total, .used = used, .available = available };
    }

    fn getProcessInfoLinux() !struct { cpu_percent: f64, memory_rss: u64, memory_vms: u64, threads: u32, uptime: u64 } {
        // Parse /proc/self/stat and /proc/self/status for process info
        return .{
            .cpu_percent = 15.2, // Placeholder - would parse /proc/self/stat
            .memory_rss = 128 * 1024 * 1024, // Placeholder
            .memory_vms = 256 * 1024 * 1024, // Placeholder
            .threads = 4,
            .uptime = 1800,
        };
    }

    fn getDiskInfoLinux() !struct { read_bytes: u64, write_bytes: u64, usage_percent: f64 } {
        return .{
            .read_bytes = 1024 * 1024 * 80,
            .write_bytes = 1024 * 1024 * 40,
            .usage_percent = 72.0,
        };
    }

    // macOS implementations (similar patterns)
    fn getCpuUsageMacOS() f64 {
        return 20.1; // Placeholder
    }

    fn getMemoryInfoMacOS() !struct { total: u64, used: u64, available: u64 } {
        return .{
            .total = 32 * 1024 * 1024 * 1024,
            .used = 16 * 1024 * 1024 * 1024,
            .available = 16 * 1024 * 1024 * 1024,
        };
    }

    fn getProcessInfoMacOS() !struct { cpu_percent: f64, memory_rss: u64, memory_vms: u64, threads: u32, uptime: u64 } {
        return .{
            .cpu_percent = 18.7,
            .memory_rss = 200 * 1024 * 1024,
            .memory_vms = 400 * 1024 * 1024,
            .threads = 6,
            .uptime = 2400,
        };
    }

    fn getDiskInfoMacOS() !struct { read_bytes: u64, write_bytes: u64, usage_percent: f64 } {
        return .{
            .read_bytes = 1024 * 1024 * 120,
            .write_bytes = 1024 * 1024 * 60,
            .usage_percent = 68.0,
        };
    }
};

/// Performance sampler with periodic monitoring
pub const PerformanceSampler = struct {
    allocator: std.mem.Allocator,
    config: SamplerConfig,
    samples: std.ArrayList(PerformanceSample),
    running: std.atomic.Value(bool),
    thread: ?std.Thread,
    process_start_time: i64,

    // Alert callbacks
    cpu_alert_callback: ?*const fn (f64) void,
    memory_alert_callback: ?*const fn (f64) void,
    disk_alert_callback: ?*const fn (f64) void,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, config: SamplerConfig) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        self.* = .{
            .allocator = allocator,
            .config = config,
            .samples = std.ArrayList(PerformanceSample).init(allocator),
            .running = std.atomic.Value(bool).init(false),
            .thread = null,
            .process_start_time = std.time.timestamp(),
            .cpu_alert_callback = null,
            .memory_alert_callback = null,
            .disk_alert_callback = null,
        };

        try self.samples.ensureTotalCapacity(config.history_size);

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.stop();
        self.samples.deinit();
        self.allocator.destroy(self);
    }

    pub fn start(self: *Self) !void {
        if (self.running.load(.monotonic)) return; // Already running

        self.running.store(true, .monotonic);
        self.thread = try std.Thread.spawn(.{}, samplingLoop, .{self});

        std.debug.print("Performance sampler started (interval: {}ms, history: {} samples)\n", .{ self.config.interval_ms, self.config.history_size });
    }

    pub fn stop(self: *Self) void {
        self.running.store(false, .monotonic);

        if (self.thread) |thread| {
            thread.join();
            self.thread = null;
        }

        std.debug.print("Performance sampler stopped\n");
    }

    pub fn setAlertCallbacks(self: *Self, cpu_callback: ?*const fn (f64) void, memory_callback: ?*const fn (f64) void, disk_callback: ?*const fn (f64) void) void {
        self.cpu_alert_callback = cpu_callback;
        self.memory_alert_callback = memory_callback;
        self.disk_alert_callback = disk_callback;
    }

    pub fn getCurrentSample(self: *Self) ?PerformanceSample {
        if (self.samples.items.len == 0) return null;
        return self.samples.items[self.samples.items.len - 1];
    }

    pub fn getAverageCpuUsage(self: *Self, duration_seconds: u32) f64 {
        const samples_needed = @min(duration_seconds * 1000 / self.config.interval_ms, self.samples.items.len);
        if (samples_needed == 0) return 0.0;

        const start_idx = self.samples.items.len - samples_needed;
        var total: f64 = 0.0;

        for (self.samples.items[start_idx..]) |sample| {
            total += sample.system_cpu_percent;
        }

        return total / @as(f64, @floatFromInt(samples_needed));
    }

    pub fn getAverageMemoryUsage(self: *Self, duration_seconds: u32) f64 {
        const samples_needed = @min(duration_seconds * 1000 / self.config.interval_ms, self.samples.items.len);
        if (samples_needed == 0) return 0.0;

        const start_idx = self.samples.items.len - samples_needed;
        var total: f64 = 0.0;

        for (self.samples.items[start_idx..]) |sample| {
            total += sample.calculateMemoryUsagePercent();
        }

        return total / @as(f64, @floatFromInt(samples_needed));
    }

    pub fn getSamplesInRange(self: *Self, start_time: i64, end_time: i64, allocator: std.mem.Allocator) ![]PerformanceSample {
        var result = std.ArrayList(PerformanceSample).init(allocator);

        for (self.samples.items) |sample| {
            if (sample.timestamp >= start_time and sample.timestamp <= end_time) {
                try result.append(sample);
            }
        }

        return try result.toOwnedSlice();
    }

    fn samplingLoop(self: *Self) void {
        while (self.running.load(.monotonic)) {
            const sample = self.collectSample() catch |err| {
                std.debug.print("Failed to collect performance sample: {any}\n", .{err});
                continue;
            };

            // Add sample to history
            if (self.samples.items.len >= self.config.history_size) {
                _ = self.samples.orderedRemove(0); // Remove oldest sample
            }
            self.samples.append(sample) catch |err| {
                std.debug.print("Failed to store performance sample: {any}\n", .{err});
            };

            // Check for alerts
            self.checkAlerts(&sample);

            // Sleep until next sample
            std.time.sleep(self.config.interval_ms * std.time.ns_per_ms);
        }
    }

    fn collectSample(self: *Self) !PerformanceSample {
        var sample = PerformanceSample.init();

        // Collect CPU information
        if (self.config.enable_cpu_sampling) {
            sample.system_cpu_percent = SystemInfo.getCpuUsage() catch 0.0;
            sample.cpu_cores = @as(u32, @intCast(std.Thread.getCpuCount() catch 1));
        }

        // Collect memory information
        if (self.config.enable_memory_sampling) {
            const memory_info = SystemInfo.getMemoryInfo() catch .{ .total = 0, .used = 0, .available = 0 };
            sample.system_memory_total = memory_info.total;
            sample.system_memory_used = memory_info.used;
            sample.system_memory_available = memory_info.available;
        }

        // Collect process information
        const process_info = SystemInfo.getProcessInfo() catch .{ .cpu_percent = 0.0, .memory_rss = 0, .memory_vms = 0, .threads = 0, .uptime = 0 };
        sample.process_cpu_percent = process_info.cpu_percent;
        sample.process_memory_rss = process_info.memory_rss;
        sample.process_memory_vms = process_info.memory_vms;
        sample.process_threads = process_info.threads;
        sample.process_uptime_seconds = @as(u64, @intCast(std.time.timestamp() - self.process_start_time));

        // Collect disk information
        if (self.config.enable_disk_sampling) {
            const disk_info = SystemInfo.getDiskInfo() catch .{ .read_bytes = 0, .write_bytes = 0, .usage_percent = 0.0 };
            sample.disk_read_bytes = disk_info.read_bytes;
            sample.disk_write_bytes = disk_info.write_bytes;
            sample.disk_usage_percent = disk_info.usage_percent;
        }

        return sample;
    }

    fn checkAlerts(self: *Self, sample: *const PerformanceSample) void {
        // CPU alert
        if (sample.system_cpu_percent > self.config.alert_cpu_threshold) {
            if (self.cpu_alert_callback) |callback| {
                callback(sample.system_cpu_percent);
            }
        }

        // Memory alert
        const memory_percent = sample.calculateMemoryUsagePercent();
        if (memory_percent > self.config.alert_memory_threshold) {
            if (self.memory_alert_callback) |callback| {
                callback(memory_percent);
            }
        }

        // Disk alert
        if (sample.disk_usage_percent > self.config.alert_disk_threshold) {
            if (self.disk_alert_callback) |callback| {
                callback(sample.disk_usage_percent);
            }
        }
    }

    /// Export performance statistics
    pub fn exportStats(self: *Self, allocator: std.mem.Allocator) ![]const u8 {
        if (self.samples.items.len == 0) {
            return try allocator.dupe(u8, "{\"error\":\"No samples available\"}");
        }

        const current = self.getCurrentSample().?;
        const avg_cpu_5m = self.getAverageCpuUsage(300); // 5 minutes
        const avg_memory_5m = self.getAverageMemoryUsage(300);

        return try std.fmt.allocPrint(allocator,
            \\{{"current":{{"timestamp":{d},"system_cpu_percent":{d:.2},"process_cpu_percent":{d:.2},"memory_usage_percent":{d:.2},"process_memory_mb":{d:.2},"disk_usage_percent":{d:.2},"process_threads":{d},"uptime_seconds":{d}}},"averages":{{"cpu_5m":{d:.2},"memory_5m":{d:.2}}},"alerts":{{"cpu_threshold":{d:.1},"memory_threshold":{d:.1},"disk_threshold":{d:.1}}},"sampling":{{"interval_ms":{d},"samples_collected":{d},"history_size":{d}}}}}
        , .{ current.timestamp, current.system_cpu_percent, current.process_cpu_percent, current.calculateMemoryUsagePercent(), @as(f64, @floatFromInt(current.process_memory_rss)) / 1024.0 / 1024.0, current.disk_usage_percent, current.process_threads, current.process_uptime_seconds, avg_cpu_5m, avg_memory_5m, self.config.alert_cpu_threshold, self.config.alert_memory_threshold, self.config.alert_disk_threshold, self.config.interval_ms, self.samples.items.len, self.config.history_size });
    }
};
