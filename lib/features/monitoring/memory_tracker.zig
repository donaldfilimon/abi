//! Memory Usage Tracking and Profiling System
//!
//! This module provides comprehensive memory monitoring capabilities including:
//! - Memory allocation tracking
//! - Memory usage statistics
//! - Memory leak detection
//! - Performance monitoring
//! - Memory profiling tools

const std = @import("std");
const builtin = @import("builtin");

/// Memory allocation record
pub const AllocationRecord = struct {
    /// Unique allocation ID
    id: u64,
    /// Allocation size in bytes
    size: usize,
    /// Alignment requirement
    alignment: u29,
    /// Source file where allocation occurred
    file: []const u8,
    /// Line number where allocation occurred
    line: u32,
    /// Function name where allocation occurred
    function: []const u8,
    /// Timestamp when allocation occurred (nanoseconds since epoch)
    timestamp: u64,
    /// Stack trace (optional)
    stack_trace: ?[]const u8,
    /// Whether this allocation has been freed
    freed: bool,
    /// Timestamp when freed (if applicable)
    freed_timestamp: ?u64,

    /// Calculate memory usage for this allocation
    pub fn memoryUsage(self: AllocationRecord) usize {
        return if (self.freed) 0 else self.size;
    }

    /// Get allocation age in nanoseconds
    pub fn age(self: AllocationRecord, current_time: u64) u64 {
        return current_time - self.timestamp;
    }

    /// Check if allocation is a potential leak
    pub fn isPotentialLeak(self: AllocationRecord, current_time: u64, leak_threshold_ns: u64) bool {
        return !self.freed and self.age(current_time) > leak_threshold_ns;
    }
};

/// Memory statistics snapshot
pub const MemoryStats = struct {
    /// Total bytes currently allocated
    total_allocated: usize = 0,
    /// Total bytes freed
    total_freed: usize = 0,
    /// Current number of active allocations
    active_allocations: usize = 0,
    /// Peak memory usage
    peak_usage: usize = 0,
    /// Total number of allocations
    total_allocation_count: u64 = 0,
    /// Total number of deallocations
    total_deallocation_count: u64 = 0,
    /// Average allocation size
    average_allocation_size: f64 = 0,
    /// Largest single allocation
    largest_allocation: usize = 0,
    /// Smallest single allocation
    smallest_allocation: usize = std.math.maxInt(usize),
    /// Timestamp when stats were captured
    timestamp: u64 = 0,

    /// Calculate current memory usage
    pub fn currentUsage(self: MemoryStats) usize {
        return self.total_allocated - self.total_freed;
    }

    /// Calculate memory efficiency (1.0 = no waste, lower = more fragmentation)
    pub fn efficiency(self: MemoryStats) f64 {
        if (self.total_allocation_count == 0) return 1.0;
        return @as(f64, @floatFromInt(self.total_freed)) / @as(f64, @floatFromInt(self.total_allocated));
    }

    /// Get allocation success rate
    pub fn allocationSuccessRate(self: MemoryStats) f64 {
        if (self.total_allocation_count == 0) return 1.0;
        return @as(f64, @floatFromInt(self.active_allocations)) / @as(f64, @floatFromInt(self.total_allocation_count));
    }
};

/// Memory profiler configuration
pub const MemoryProfilerConfig = struct {
    /// Enable detailed stack traces
    enable_stack_traces: bool = false,
    /// Memory leak detection threshold (nanoseconds)
    leak_threshold_ns: u64 = 1_000_000_000, // 1 second
    /// Enable periodic statistics collection
    enable_periodic_stats: bool = true,
    /// Statistics collection interval (nanoseconds)
    stats_interval_ns: u64 = 100_000_000, // 100ms
    /// Maximum number of allocation records to keep
    max_records: usize = 10_000,
    /// Enable memory usage warnings
    enable_warnings: bool = true,
    /// Memory usage warning threshold (bytes)
    warning_threshold: usize = 100 * 1024 * 1024, // 100MB
    /// Memory usage critical threshold (bytes)
    critical_threshold: usize = 500 * 1024 * 1024, // 500MB
};

/// Memory profiler main structure
pub const MemoryProfiler = struct {
    /// Configuration
    config: MemoryProfilerConfig,
    /// Memory allocator
    allocator: std.mem.Allocator,
    /// Active allocation records
    records: std.ArrayListUnmanaged(AllocationRecord),
    /// Statistics
    stats: MemoryStats,
    /// Next allocation ID
    next_id: u64 = 1,
    /// Mutex for thread safety
    mutex: std.Thread.Mutex = .{},
    /// Timer for periodic stats
    timer: ?std.time.Timer = null,
    /// Last stats collection time
    last_stats_time: u64 = 0,

    /// Initialize memory profiler
    pub fn init(allocator: std.mem.Allocator, config: MemoryProfilerConfig) !*MemoryProfiler {
        const self = try allocator.create(MemoryProfiler);
        errdefer allocator.destroy(self);

        self.* = .{
            .config = config,
            .allocator = allocator,
            .records = try std.ArrayListUnmanaged(AllocationRecord).initCapacity(allocator, config.max_records / 4),
            .stats = .{},
            .timer = if (config.enable_periodic_stats) try std.time.Timer.start() else null,
            .last_stats_time = @as(u64, @intCast(std.time.nanoTimestamp())),
        };

        return self;
    }

    /// Deinitialize memory profiler
    pub fn deinit(self: *MemoryProfiler) void {
        // Lock mutex for the entire cleanup process
        self.mutex.lock();

        // Clean up records - check for null pointers and avoid double-free
        for (self.records.items) |record| {
            // Free file string if it exists
            if (record.file.len > 0) {
                self.allocator.free(record.file);
            }
            // Free function string if it exists
            if (record.function.len > 0) {
                self.allocator.free(record.function);
            }
            // Free stack trace if it exists and is not null
            if (record.stack_trace) |trace| {
                if (trace.len > 0) {
                    self.allocator.free(trace);
                }
            }
        }
        self.records.deinit(self.allocator);

        // Clear the global profiler reference first
        if (global_profiler) |profiler| {
            if (profiler == self) {
                global_profiler = null;
            }
        }

        // Unlock before destroying self to prevent issues
        self.mutex.unlock();

        // Clean up self - use defer to ensure this always happens
        self.allocator.destroy(self);
    }

    /// Record a memory allocation
    pub fn recordAllocation(
        self: *MemoryProfiler,
        size: usize,
        alignment: u29,
        file: []const u8,
        line: u32,
        function: []const u8,
        stack_trace: ?[]const u8,
    ) !u64 {
        self.mutex.lock();
        defer self.mutex.unlock();

        const id = self.next_id;
        self.next_id += 1;

        // Fix timestamp issue: Use monotonic time source for consistent timestamps
        const timestamp = if (self.timer) |*timer| timer.read() else std.time.nanoTimestamp();

        // Create allocation record
        const record = AllocationRecord{
            .id = id,
            .size = size,
            .alignment = alignment,
            .file = try self.allocator.dupe(u8, file),
            .line = line,
            .function = try self.allocator.dupe(u8, function),
            .timestamp = @as(u64, @intCast(timestamp)),
            .stack_trace = if (stack_trace) |trace| try self.allocator.dupe(u8, trace) else null,
            .freed = false,
            .freed_timestamp = null,
        };

        // Add to records (with size limit)
        if (self.records.items.len >= self.config.max_records) {
            // Remove oldest freed record if at capacity
            var oldest_freed_index: ?usize = null;
            var oldest_freed_time: u64 = std.math.maxInt(u64);

            for (self.records.items, 0..) |rec, i| {
                if (rec.freed and rec.freed_timestamp.? < oldest_freed_time) {
                    oldest_freed_time = rec.freed_timestamp.?;
                    oldest_freed_index = i;
                }
            }

            if (oldest_freed_index) |idx| {
                const old_record = self.records.swapRemove(idx);
                // Safe cleanup with null checks
                if (old_record.file.len > 0) {
                    self.allocator.free(old_record.file);
                }
                if (old_record.function.len > 0) {
                    self.allocator.free(old_record.function);
                }
                if (old_record.stack_trace) |trace| {
                    if (trace.len > 0) {
                        self.allocator.free(trace);
                    }
                }
            } else {
                // If no freed records, remove oldest record
                const old_record = self.records.orderedRemove(0);
                // Safe cleanup with null checks
                if (old_record.file.len > 0) {
                    self.allocator.free(old_record.file);
                }
                if (old_record.function.len > 0) {
                    self.allocator.free(old_record.function);
                }
                if (old_record.stack_trace) |trace| {
                    if (trace.len > 0) {
                        self.allocator.free(trace);
                    }
                }
            }
        }

        try self.records.append(self.allocator, record);

        // Update statistics
        self.stats.total_allocated += size;
        self.stats.active_allocations += 1;
        self.stats.total_allocation_count += 1;
        self.stats.average_allocation_size = @as(f64, @floatFromInt(self.stats.total_allocated)) / @as(f64, @floatFromInt(self.stats.total_allocation_count));

        if (size > self.stats.largest_allocation) {
            self.stats.largest_allocation = size;
        }
        if (size < self.stats.smallest_allocation) {
            self.stats.smallest_allocation = size;
        }

        const current_usage = self.stats.currentUsage();
        if (current_usage > self.stats.peak_usage) {
            self.stats.peak_usage = current_usage;
        }

        // Check thresholds and warn
        if (self.config.enable_warnings) {
            if (current_usage > self.config.critical_threshold) {
                std.log.err("CRITICAL: Memory usage exceeded critical threshold: {d} bytes", .{current_usage});
            } else if (current_usage > self.config.warning_threshold) {
                std.log.warn("WARNING: Memory usage exceeded warning threshold: {d} bytes", .{current_usage});
            }
        }

        return id;
    }

    /// Record a memory deallocation
    pub fn recordDeallocation(self: *MemoryProfiler, id: u64) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Find the allocation record
        for (self.records.items) |*record| {
            if (record.id == id and !record.freed) {
                record.freed = true;
                record.freed_timestamp = @as(u64, @intCast(if (self.timer) |*timer| timer.read() else std.time.nanoTimestamp()));

                // Update statistics
                self.stats.total_freed += record.size;
                self.stats.active_allocations -= 1;
                self.stats.total_deallocation_count += 1;

                break;
            }
        }
    }

    /// Get current memory statistics
    pub fn getStats(self: *MemoryProfiler) MemoryStats {
        self.mutex.lock();
        defer self.mutex.unlock();

        var stats = self.stats;
        stats.timestamp = @as(u64, @intCast(if (self.timer) |*timer| timer.read() else std.time.nanoTimestamp()));
        return stats;
    }

    /// Get potential memory leaks
    pub fn getPotentialLeaks(self: *MemoryProfiler, allocator: std.mem.Allocator) ![]AllocationRecord {
        self.mutex.lock();
        defer self.mutex.unlock();

        const current_time = std.time.nanoTimestamp();
        var leaks = std.ArrayListUnmanaged(AllocationRecord){};

        for (self.records.items) |record| {
            if (record.isPotentialLeak(current_time, self.config.leak_threshold_ns)) {
                try leaks.append(allocator, record);
            }
        }

        return try leaks.toOwnedSlice(allocator);
    }

    /// Generate memory usage report
    pub fn generateReport(self: *MemoryProfiler, allocator: std.mem.Allocator) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        const stats = self.getStats();
        const current_time = std.time.nanoTimestamp();

        var report = std.ArrayListUnmanaged(u8){};
        errdefer report.deinit(allocator);

        const writer = report.writer(allocator);

        try writer.print("=== Memory Usage Report ===\n", .{});
        try writer.print("Timestamp: {d}\n", .{current_time});
        try writer.print("Current Usage: {d} bytes\n", .{stats.currentUsage()});
        try writer.print("Peak Usage: {d} bytes\n", .{stats.peak_usage});
        try writer.print("Total Allocated: {d} bytes\n", .{stats.total_allocated});
        try writer.print("Total Freed: {d} bytes\n", .{stats.total_freed});
        try writer.print("Active Allocations: {d}\n", .{stats.active_allocations});
        try writer.print("Total Allocation Count: {d}\n", .{stats.total_allocation_count});
        try writer.print("Total Deallocation Count: {d}\n", .{stats.total_deallocation_count});
        try writer.print("Average Allocation Size: {d:.2} bytes\n", .{stats.average_allocation_size});
        try writer.print("Largest Allocation: {d} bytes\n", .{stats.largest_allocation});
        try writer.print("Smallest Allocation: {d} bytes\n", .{stats.smallest_allocation});
        try writer.print("Memory Efficiency: {d:.3}\n", .{stats.efficiency()});
        try writer.print("Allocation Success Rate: {d:.3}\n", .{stats.allocationSuccessRate()});

        // Check for potential leaks
        var leak_count: usize = 0;
        for (self.records.items) |record| {
            if (record.isPotentialLeak(current_time, self.config.leak_threshold_ns)) {
                leak_count += 1;
            }
        }

        try writer.print("Potential Leaks: {d}\n", .{leak_count});

        if (leak_count > 0) {
            try writer.print("\n=== Potential Memory Leaks ===\n", .{});
            for (self.records.items) |record| {
                if (record.isPotentialLeak(current_time, self.config.leak_threshold_ns)) {
                    try writer.print("Leak {d}: {d} bytes, age: {d:.3}s\n", .{
                        record.id,
                        record.size,
                        @as(f64, @floatFromInt(record.age(current_time))) / 1_000_000_000.0,
                    });
                    try writer.print("  File: {s}:{d}\n", .{ record.file, record.line });
                    try writer.print("  Function: {s}\n", .{record.function});
                }
            }
        }

        try writer.print("\n=== Top Memory Users ===\n", .{});

        // Find top memory users
        var top_users = std.ArrayListUnmanaged(struct { file: []const u8, total: usize }){};
        defer {
            for (top_users.items) |item| {
                allocator.free(item.file);
            }
            top_users.deinit(allocator);
        }

        // Group by file
        var file_totals = std.StringHashMapUnmanaged(usize){};
        defer file_totals.deinit(allocator);

        for (self.records.items) |record| {
            if (!record.freed) {
                const total = file_totals.get(record.file) orelse 0;
                try file_totals.put(allocator, record.file, total + record.size);
            }
        }

        // Convert to sorted list
        var file_iter = file_totals.iterator();
        while (file_iter.next()) |entry| {
            try top_users.append(allocator, .{
                .file = try allocator.dupe(u8, entry.key_ptr.*),
                .total = entry.value_ptr.*,
            });
        }

        // Sort by total (descending)
        std.sort.insertion(struct { file: []const u8, total: usize }, top_users.items, {}, struct {
            fn lessThan(_: void, a: struct { file: []const u8, total: usize }, b: struct { file: []const u8, total: usize }) bool {
                return a.total > b.total;
            }
        }.lessThan);

        // Show top 10
        const show_count = @min(10, top_users.items.len);
        for (top_users.items[0..show_count]) |item| {
            try writer.print("  {s}: {d} bytes\n", .{ item.file, item.total });
        }

        try writer.print("\n=== End Report ===\n", .{});

        return try report.toOwnedSlice(allocator);
    }

    /// Reset statistics
    pub fn resetStats(self: *MemoryProfiler) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.stats = .{};
        self.next_id = 1;
    }

    /// Collect periodic statistics
    pub fn collectPeriodicStats(self: *MemoryProfiler) void {
        if (!self.config.enable_periodic_stats or self.timer == null) return;

        const current_time = self.timer.?.read();
        if (current_time - self.last_stats_time >= self.config.stats_interval_ns) {
            const stats = self.getStats();

            // Log periodic stats
            std.log.info("Memory stats - Usage: {d} bytes, Peak: {d} bytes, Active: {d}", .{ stats.currentUsage(), stats.peak_usage, stats.active_allocations });

            self.last_stats_time = current_time;
        }
    }
};

/// Global memory profiler instance
var global_profiler: ?*MemoryProfiler = null;

/// Initialize global memory profiler
pub fn initGlobalProfiler(allocator: std.mem.Allocator, config: MemoryProfilerConfig) !void {
    if (global_profiler != null) {
        deinitGlobalProfiler();
    }
    global_profiler = try MemoryProfiler.init(allocator, config);
}

/// Deinitialize global memory profiler
pub fn deinitGlobalProfiler() void {
    if (global_profiler) |profiler| {
        // Set to null first to prevent re-entrant calls
        global_profiler = null;
        profiler.deinit();
    }
}

/// Get global memory profiler instance
pub fn getGlobalProfiler() ?*MemoryProfiler {
    return global_profiler;
}

/// Tracked allocator that integrates with memory profiler
pub const TrackedAllocator = struct {
    /// Parent allocator
    parent_allocator: std.mem.Allocator,
    /// Memory profiler
    profiler: *MemoryProfiler,

    /// Initialize tracked allocator
    pub fn init(parent_allocator: std.mem.Allocator, profiler: *MemoryProfiler) TrackedAllocator {
        return .{
            .parent_allocator = parent_allocator,
            .profiler = profiler,
        };
    }

    /// Get allocator interface
    pub fn allocator(self: *TrackedAllocator) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .remap = remap,
                .free = free,
            },
        };
    }

    /// Allocation function
    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *TrackedAllocator = @ptrCast(@alignCast(ctx));

        // Perform actual allocation
        const result = self.parent_allocator.rawAlloc(len, alignment, ret_addr);
        if (result == null) return null;

        // Record allocation (simplified - in real implementation, we'd extract file/line info)
        const id = self.profiler.recordAllocation(
            len,
            @intCast(@as(usize, 1) << @intFromEnum(alignment)),
            "unknown", // Would extract from debug info
            0, // Would extract from debug info
            "unknown", // Would extract from debug info
            null, // Stack trace if enabled
        ) catch {
            // If recording fails, still return the allocation
            return result;
        };

        // Store allocation ID for later deallocation tracking
        // (In a real implementation, we'd store this in a map or use a different approach)
        _ = id;

        return result;
    }

    /// Resize function
    fn resize(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *TrackedAllocator = @ptrCast(@alignCast(ctx));
        return self.parent_allocator.rawResize(buf, alignment, new_len, ret_addr);
    }

    /// Remap function
    fn remap(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *TrackedAllocator = @ptrCast(@alignCast(ctx));
        return self.parent_allocator.rawRemap(memory, alignment, new_len, ret_addr);
    }

    /// Free function
    fn free(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *TrackedAllocator = @ptrCast(@alignCast(ctx));

        // Record deallocation
        // (In a real implementation, we'd look up the allocation ID from the buffer)
        // For now, we skip the deallocation recording as it requires more complex tracking

        // Perform actual free
        self.parent_allocator.rawFree(buf, alignment, ret_addr);
    }
};

/// Memory usage monitor
pub const MemoryMonitor = struct {
    /// Memory profiler
    profiler: *MemoryProfiler,
    /// Monitoring thread
    monitor_thread: ?std.Thread = null,
    /// Stop monitoring flag
    stop_monitoring: bool = false,
    /// Monitoring interval (nanoseconds)
    interval_ns: u64 = 1_000_000_000, // 1 second

    /// Initialize memory monitor
    pub fn init(profiler: *MemoryProfiler) !*MemoryMonitor {
        const self = try profiler.allocator.create(MemoryMonitor);
        self.* = .{
            .profiler = profiler,
        };
        return self;
    }

    /// Start monitoring thread
    pub fn start(self: *MemoryMonitor) !void {
        if (self.monitor_thread != null) return;

        self.stop_monitoring = false;
        self.monitor_thread = try std.Thread.spawn(.{}, monitorLoop, .{self});
    }

    /// Stop monitoring
    pub fn stop(self: *MemoryMonitor) void {
        if (self.monitor_thread) |thread| {
            self.stop_monitoring = true;
            thread.join();
            self.monitor_thread = null;
        }
    }

    /// Deinitialize monitor
    pub fn deinit(self: *MemoryMonitor) void {
        self.stop();
        self.profiler.allocator.destroy(self);
    }

    /// Monitoring loop
    fn monitorLoop(self: *MemoryMonitor) void {
        while (!self.stop_monitoring) {
            // Collect periodic statistics
            self.profiler.collectPeriodicStats();

            // Check for memory leaks
            const leaks = self.profiler.getPotentialLeaks(self.profiler.allocator) catch {
                std.log.err("Failed to check for memory leaks", .{});
                continue;
            };
            defer self.profiler.allocator.free(leaks);

            if (leaks.len > 0) {
                std.log.warn("Detected {d} potential memory leaks", .{leaks.len});
            }

            // Sleep for interval
            std.Thread.sleep(self.interval_ns);
        }
    }
};

/// Performance monitoring utilities
pub const PerformanceMonitor = struct {
    /// Start time for measurement
    start_time: u64 = 0,
    /// End time for measurement
    end_time: u64 = 0,
    /// Memory usage at start
    start_memory: usize = 0,
    /// Memory usage at end
    end_memory: usize = 0,
    /// Profiler reference
    profiler: ?*MemoryProfiler = null,

    /// Start performance measurement
    pub fn start(self: *PerformanceMonitor) void {
        self.start_time = std.time.nanoTimestamp();
        if (self.profiler) |profiler| {
            self.start_memory = profiler.getStats().currentUsage();
        }
    }

    /// End performance measurement
    pub fn end(self: *PerformanceMonitor) void {
        self.end_time = std.time.nanoTimestamp();
        if (self.profiler) |profiler| {
            self.end_memory = profiler.getStats().currentUsage();
        }
    }

    /// Get elapsed time in nanoseconds
    pub fn elapsedTime(self: PerformanceMonitor) u64 {
        return self.end_time - self.start_time;
    }

    /// Get memory usage delta
    pub fn memoryDelta(self: PerformanceMonitor) i64 {
        return @as(i64, @intCast(self.end_memory)) - @as(i64, @intCast(self.start_memory));
    }

    /// Generate performance report
    pub fn generateReport(self: PerformanceMonitor, allocator: std.mem.Allocator, operation_name: []const u8) ![]u8 {
        var report = std.ArrayListUnmanaged(u8){};
        errdefer report.deinit(allocator);

        const writer = report.writer(allocator);

        const elapsed_ns = self.elapsedTime();
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
        const memory_delta = self.memoryDelta();

        try writer.print("=== Performance Report: {s} ===\n", .{operation_name});
        try writer.print("Elapsed Time: {d:.3} ms ({d} ns)\n", .{ elapsed_ms, elapsed_ns });
        try writer.print("Memory Delta: {d} bytes\n", .{memory_delta});
        if (elapsed_ns > 0) {
            const memory_rate = @as(f64, @floatFromInt(memory_delta)) / (@as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0);
            try writer.print("Memory Rate: {d:.2} bytes/sec\n", .{memory_rate});
        }
        try writer.print("Start Memory: {d} bytes\n", .{self.start_memory});
        try writer.print("End Memory: {d} bytes\n", .{self.end_memory});

        return try report.toOwnedSlice(allocator);
    }
};

/// Utility functions for memory profiling
pub const utils = struct {
    /// Create a simple memory profiler configuration
    pub fn simpleConfig() MemoryProfilerConfig {
        return .{
            .enable_stack_traces = false,
            .leak_threshold_ns = 5_000_000_000, // 5 seconds
            .enable_periodic_stats = true,
            .stats_interval_ns = 1_000_000_000, // 1 second
            .max_records = 5_000,
            .enable_warnings = true,
            .warning_threshold = 50 * 1024 * 1024, // 50MB
            .critical_threshold = 200 * 1024 * 1024, // 200MB
        };
    }

    /// Create a development configuration with more detailed tracking
    pub fn developmentConfig() MemoryProfilerConfig {
        return .{
            .enable_stack_traces = true,
            .leak_threshold_ns = 1_000_000_000, // 1 second
            .enable_periodic_stats = true,
            .stats_interval_ns = 100_000_000, // 100ms
            .max_records = 20_000,
            .enable_warnings = true,
            .warning_threshold = 10 * 1024 * 1024, // 10MB
            .critical_threshold = 50 * 1024 * 1024, // 50MB
        };
    }

    /// Create a production configuration with minimal overhead
    pub fn productionConfig() MemoryProfilerConfig {
        return .{
            .enable_stack_traces = false,
            .leak_threshold_ns = 30_000_000_000, // 30 seconds
            .enable_periodic_stats = true,
            .stats_interval_ns = 10_000_000_000, // 10 seconds
            .max_records = 1_000,
            .enable_warnings = true,
            .warning_threshold = 500 * 1024 * 1024, // 500MB
            .critical_threshold = 2 * 1024 * 1024 * 1024, // 2GB
        };
    }
};
