//! Performance Profiling Infrastructure
//!
//! This module provides comprehensive performance profiling capabilities including:
//! - CPU performance monitoring
//! - Function call tracing
//! - Performance counters
//! - Hot path analysis
//! - Integration with memory tracking

const std = @import("std");
const memory_tracker = @import("memory_tracker.zig");

/// Performance profiling configuration
pub const ProfilingConfig = struct {
    /// Enable CPU profiling
    enable_cpu_profiling: bool = true,
    /// Enable function call tracing
    enable_call_tracing: bool = false,
    /// Enable performance counters
    enable_counters: bool = true,
    /// Sampling interval (nanoseconds)
    sampling_interval_ns: u64 = 1_000_000, // 1ms
    /// Maximum number of call stack frames
    max_stack_depth: usize = 64,
    /// Enable memory integration
    enable_memory_integration: bool = true,
    /// Performance report output interval
    report_interval_ns: u64 = 10_000_000_000, // 10 seconds
};

/// Function call record
pub const CallRecord = struct {
    /// Function name
    function_name: []const u8,
    /// Source file
    file: []const u8,
    /// Line number
    line: u32,
    /// Entry timestamp
    entry_time: u64,
    /// Exit timestamp (0 if not exited)
    exit_time: u64 = 0,
    /// Call depth
    depth: u32,
    /// Parent call ID
    parent_id: ?u64,
    /// Unique call ID
    call_id: u64,
    /// Thread ID
    thread_id: std.Thread.Id,

    /// Calculate call duration
    pub fn duration(self: CallRecord) u64 {
        if (self.exit_time == 0) return 0;
        return self.exit_time - self.entry_time;
    }

    /// Check if call is complete
    pub fn isComplete(self: CallRecord) bool {
        return self.exit_time != 0;
    }
};

/// Performance counter
pub const PerformanceCounter = struct {
    /// Counter name
    name: []const u8,
    /// Counter value
    value: u64 = 0,
    /// Unit of measurement
    unit: []const u8,
    /// Description
    description: []const u8,
    /// Last update timestamp
    last_update: u64 = 0,

    /// Increment counter
    pub fn increment(self: *PerformanceCounter) void {
        self.value += 1;
        self.last_update = std.time.nanoTimestamp();
    }

    /// Add value to counter
    pub fn add(self: *PerformanceCounter, delta: u64) void {
        self.value += delta;
        self.last_update = std.time.nanoTimestamp();
    }

    /// Set counter value
    pub fn set(self: *PerformanceCounter, new_value: u64) void {
        self.value = new_value;
        self.last_update = std.time.nanoTimestamp();
    }

    /// Reset counter
    pub fn reset(self: *PerformanceCounter) void {
        self.value = 0;
        self.last_update = std.time.nanoTimestamp();
    }
};

/// Performance profile data
pub const PerformanceProfile = struct {
    /// Total execution time
    total_time: u64 = 0,
    /// CPU time spent
    cpu_time: u64 = 0,
    /// Memory allocations during profiling
    allocations: u64 = 0,
    /// Memory deallocations during profiling
    deallocations: u64 = 0,
    /// Peak memory usage during profiling
    peak_memory: usize = 0,
    /// Function call records
    call_records: std.ArrayListUnmanaged(CallRecord),
    /// Performance counters
    counters: std.StringHashMapUnmanaged(PerformanceCounter),
    /// Start timestamp
    start_time: u64,
    /// End timestamp (0 if profiling active)
    end_time: u64 = 0,

    /// Calculate profiling duration
    pub fn duration(self: PerformanceProfile) u64 {
        if (self.end_time == 0) return std.time.nanoTimestamp() - self.start_time;
        return self.end_time - self.start_time;
    }

    /// Get profiling duration in seconds
    pub fn durationSeconds(self: PerformanceProfile) f64 {
        return @as(f64, @floatFromInt(self.duration())) / 1_000_000_000.0;
    }

    /// Calculate CPU utilization
    pub fn cpuUtilization(self: PerformanceProfile) f64 {
        const total_duration = self.duration();
        if (total_duration == 0) return 0.0;
        return @as(f64, @floatFromInt(self.cpu_time)) / @as(f64, @floatFromInt(total_duration));
    }
};

/// Function profiler for instrumenting functions
pub const FunctionProfiler = struct {
    /// Function name
    function_name: []const u8,
    /// File name
    file_name: []const u8,
    /// Line number
    line_number: u32,
    /// Total call count
    call_count: u64 = 0,
    /// Total execution time
    total_time: u64 = 0,
    /// Minimum execution time
    min_time: u64 = std.math.maxInt(u64),
    /// Maximum execution time
    max_time: u64 = 0,
    /// Average execution time
    average_time: f64 = 0,
    /// Last call timestamp
    last_call_time: u64 = 0,

    /// Record function entry
    pub fn enter(self: *FunctionProfiler) u64 {
        const entry_time = std.time.nanoTimestamp();
        self.call_count += 1;
        self.last_call_time = entry_time;
        return entry_time;
    }

    /// Record function exit
    pub fn exit(self: *FunctionProfiler, entry_time: u64) void {
        const exit_time = std.time.nanoTimestamp();
        const duration = exit_time - entry_time;

        self.total_time += duration;

        if (duration < self.min_time) {
            self.min_time = duration;
        }
        if (duration > self.max_time) {
            self.max_time = duration;
        }

        // Update rolling average
        if (self.call_count == 1) {
            self.average_time = @as(f64, @floatFromInt(duration));
        } else {
            const alpha = 0.1; // Smoothing factor
            self.average_time = self.average_time * (1.0 - alpha) + @as(f64, @floatFromInt(duration)) * alpha;
        }
    }

    /// Get average execution time in nanoseconds
    pub fn averageExecutionTime(self: FunctionProfiler) u64 {
        if (self.call_count == 0) return 0;
        return @intFromFloat(self.average_time);
    }
};

/// Main performance profiler
pub const PerformanceProfiler = struct {
    /// Configuration
    config: ProfilingConfig,
    /// Memory allocator
    allocator: std.mem.Allocator,
    /// Function profilers
    function_profilers: std.StringHashMapUnmanaged(FunctionProfiler),
    /// Current performance profile
    current_profile: ?PerformanceProfile = null,
    /// Call stack for tracing
    call_stack: std.ArrayListUnmanaged(CallRecord),
    /// Performance counters
    counters: std.StringHashMapUnmanaged(PerformanceCounter),
    /// Next call ID
    next_call_id: u64 = 1,
    /// Memory tracker integration
    memory_tracker: ?*memory_tracker.MemoryProfiler = null,
    /// Mutex for thread safety
    mutex: std.Thread.Mutex = .{},
    /// Profiling thread
    profiling_thread: ?std.Thread = null,
    /// Stop profiling flag
    stop_profiling: bool = false,

    /// Initialize performance profiler
    pub fn init(allocator: std.mem.Allocator, config: ProfilingConfig) !*PerformanceProfiler {
        const self = try allocator.create(PerformanceProfiler);
        errdefer allocator.destroy(self);

        self.* = .{
            .config = config,
            .allocator = allocator,
            .function_profilers = std.StringHashMapUnmanaged(FunctionProfiler){},
            .call_stack = try std.ArrayListUnmanaged(CallRecord).initCapacity(allocator, config.max_stack_depth),
            .counters = std.StringHashMapUnmanaged(PerformanceCounter){},
        };

        // Initialize default performance counters
        try self.initDefaultCounters();

        return self;
    }

    /// Deinitialize performance profiler
    pub fn deinit(self: *PerformanceProfiler) void {
        self.stop();

        self.mutex.lock();
        defer self.mutex.unlock();

        // Clean up function profilers
        var profiler_iter = self.function_profilers.iterator();
        while (profiler_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.function_profilers.deinit(self.allocator);

        // Clean up current profile
        if (self.current_profile) |*profile| {
            profile.call_records.deinit(self.allocator);
            var counter_iter = profile.counters.iterator();
            while (counter_iter.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
            }
            profile.counters.deinit(self.allocator);
        }

        // Clean up call stack
        self.call_stack.deinit(self.allocator);

        // Clean up counters
        var counter_iter = self.counters.iterator();
        while (counter_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.counters.deinit(self.allocator);

        self.allocator.destroy(self);
    }

    /// Initialize default performance counters
    fn initDefaultCounters(self: *PerformanceProfiler) !void {
        const default_counters = [_]struct {
            name: []const u8,
            unit: []const u8,
            description: []const u8,
        }{
            .{ .name = "cpu_cycles", .unit = "cycles", .description = "CPU cycles consumed" },
            .{ .name = "instructions", .unit = "instructions", .description = "Instructions executed" },
            .{ .name = "cache_misses", .unit = "misses", .description = "Cache misses" },
            .{ .name = "branch_misses", .unit = "misses", .description = "Branch prediction misses" },
            .{ .name = "allocations", .unit = "count", .description = "Memory allocations" },
            .{ .name = "deallocations", .unit = "count", .description = "Memory deallocations" },
            .{ .name = "function_calls", .unit = "count", .description = "Function calls" },
        };

        for (default_counters) |counter_def| {
            const name_copy = try self.allocator.dupe(u8, counter_def.name);
            const unit_copy = try self.allocator.dupe(u8, counter_def.unit);
            const desc_copy = try self.allocator.dupe(u8, counter_def.description);

            try self.counters.put(self.allocator, name_copy, .{
                .name = name_copy,
                .unit = unit_copy,
                .description = desc_copy,
            });
        }
    }

    /// Start profiling session
    pub fn startSession(self: *PerformanceProfiler, session_name: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.current_profile != null) {
            return error.ProfilingAlreadyActive;
        }

        const start_time = std.time.nanoTimestamp();

        self.current_profile = .{
            .call_records = try std.ArrayListUnmanaged(CallRecord).initCapacity(self.allocator, 1000),
            .counters = std.StringHashMapUnmanaged(PerformanceCounter){},
            .start_time = start_time,
        };

        // Copy current counters to profile
        var counter_iter = self.counters.iterator();
        while (counter_iter.next()) |entry| {
            const name_copy = try self.allocator.dupe(u8, entry.key_ptr.*);
            try self.current_profile.?.counters.put(self.allocator, name_copy, entry.value_ptr.*);
        }

        std.log.info("Started performance profiling session: {s}", .{session_name});

        if (self.config.enable_cpu_profiling) {
            self.startProfilingThread();
        }
    }

    /// End profiling session
    pub fn endSession(self: *PerformanceProfiler) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.current_profile == null) {
            return error.NoActiveProfilingSession;
        }

        var profile = self.current_profile.?;
        profile.end_time = std.time.nanoTimestamp();

        // Calculate profile statistics
        profile.total_time = profile.duration();

        // Calculate CPU time from call records
        var total_cpu_time: u64 = 0;
        for (profile.call_records.items) |record| {
            if (record.isComplete()) {
                total_cpu_time += record.duration();
            }
        }
        profile.cpu_time = total_cpu_time;

        // Get memory statistics if integrated
        if (self.memory_tracker) |tracker| {
            const stats = tracker.getStats();
            profile.peak_memory = stats.peak_usage;
            profile.allocations = stats.total_allocation_count;
            profile.deallocations = stats.total_deallocation_count;
        }

        // Generate report
        const report = try self.generateProfileReport(&profile);

        // Clean up profile
        profile.call_records.deinit(self.allocator);
        var counter_iter = profile.counters.iterator();
        while (counter_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        profile.counters.deinit(self.allocator);

        self.current_profile = null;

        return report;
    }

    /// Start function call
    pub fn startFunctionCall(self: *PerformanceProfiler, function_name: []const u8, file: []const u8, line: u32) !u64 {
        if (!self.config.enable_call_tracing) return 0;

        self.mutex.lock();
        defer self.mutex.unlock();

        const entry_time = std.time.nanoTimestamp();
        const call_id = self.next_call_id;
        self.next_call_id += 1;

        const parent_id = if (self.call_stack.items.len > 0)
            self.call_stack.items[self.call_stack.items.len - 1].call_id
        else
            null;

        const record = CallRecord{
            .function_name = try self.allocator.dupe(u8, function_name),
            .file = try self.allocator.dupe(u8, file),
            .line = line,
            .entry_time = entry_time,
            .depth = @intCast(self.call_stack.items.len),
            .parent_id = parent_id,
            .call_id = call_id,
            .thread_id = std.Thread.getCurrentId(),
        };

        try self.call_stack.append(self.allocator, record);

        // Update function profiler
        const key = try std.fmt.allocPrint(self.allocator, "{s}:{s}:{d}", .{ function_name, file, line });
        defer self.allocator.free(key);

        const gop = try self.function_profilers.getOrPut(self.allocator, key);
        if (!gop.found_existing) {
            gop.value_ptr.* = .{
                .function_name = try self.allocator.dupe(u8, function_name),
                .file_name = try self.allocator.dupe(u8, file),
                .line_number = line,
            };
        }

        const profiler_entry_time = gop.value_ptr.enter();

        // Add to current profile if active
        if (self.current_profile) |*profile| {
            try profile.call_records.append(self.allocator, record);
        }

        // Update function calls counter
        if (self.counters.getPtr("function_calls")) |counter| {
            counter.increment();
        }

        return profiler_entry_time;
    }

    /// End function call
    pub fn endFunctionCall(self: *PerformanceProfiler, entry_time: u64) void {
        if (!self.config.enable_call_tracing) return;

        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.call_stack.items.len == 0) return;

        const record = self.call_stack.pop();
        const exit_time = std.time.nanoTimestamp();

        // Update record
        if (self.current_profile) |*profile| {
            for (profile.call_records.items) |*rec| {
                if (rec.call_id == record.call_id) {
                    rec.exit_time = exit_time;
                    break;
                }
            }
        }

        // Update function profiler
        const key = std.fmt.allocPrint(self.allocator, "{s}:{s}:{d}", .{ record.function_name, record.file, record.line }) catch return;
        defer self.allocator.free(key);

        if (self.function_profilers.getPtr(key)) |profiler| {
            profiler.exit(entry_time);
        }

        // Clean up record strings
        self.allocator.free(record.function_name);
        self.allocator.free(record.file);
    }

    /// Update performance counter
    pub fn updateCounter(self: *PerformanceProfiler, name: []const u8, delta: u64) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.counters.getPtr(name)) |counter| {
            counter.add(delta);
        } else if (self.config.enable_counters) {
            // Create new counter
            const name_copy = self.allocator.dupe(u8, name) catch return;
            const counter = PerformanceCounter{
                .name = name_copy,
                .value = delta,
                .unit = "count",
                .description = "Auto-generated counter",
                .last_update = std.time.nanoTimestamp(),
            };

            self.counters.put(self.allocator, name_copy, counter) catch {
                self.allocator.free(name_copy);
                return;
            };
        }
    }

    /// Get function profiler statistics
    pub fn getFunctionStats(self: *PerformanceProfiler, allocator: std.mem.Allocator) ![]FunctionProfiler {
        self.mutex.lock();
        defer self.mutex.unlock();

        var stats = std.ArrayListUnmanaged(FunctionProfiler){};
        errdefer stats.deinit(allocator);

        var iter = self.function_profilers.iterator();
        while (iter.next()) |entry| {
            try stats.append(allocator, entry.value_ptr.*);
        }

        return try stats.toOwnedSlice(allocator);
    }

    /// Generate profiling report
    fn generateProfileReport(self: *PerformanceProfiler, profile: *PerformanceProfile) ![]u8 {
        var report = std.ArrayListUnmanaged(u8){};
        errdefer report.deinit(self.allocator);

        const writer = report.writer(self.allocator);

        try writer.print("=== Performance Profile Report ===\n", .{});
        try writer.print("Duration: {d:.3} seconds\n", .{profile.durationSeconds()});
        try writer.print("Total CPU Time: {d} ns\n", .{profile.cpu_time});
        try writer.print("CPU Utilization: {d:.1}%\n", .{profile.cpuUtilization() * 100.0});

        if (self.memory_tracker != null) {
            try writer.print("Peak Memory Usage: {d} bytes\n", .{profile.peak_memory});
            try writer.print("Memory Allocations: {d}\n", .{profile.allocations});
            try writer.print("Memory Deallocations: {d}\n", .{profile.deallocations});
        }

        try writer.print("Function Calls Recorded: {d}\n", .{profile.call_records.items.len});

        // Performance counters
        try writer.print("\n=== Performance Counters ===\n", .{});
        var counter_iter = profile.counters.iterator();
        while (counter_iter.next()) |entry| {
            try writer.print("  {s}: {d} {s} - {s}\n", .{
                entry.value_ptr.name,
                entry.value_ptr.value,
                entry.value_ptr.unit,
                entry.value_ptr.description,
            });
        }

        // Top functions by execution time
        try writer.print("\n=== Top Functions by Execution Time ===\n", .{});
        const function_stats = try self.getFunctionStats(self.allocator);
        defer self.allocator.free(function_stats);

        // Sort by total time (descending)
        std.sort.insertion(FunctionProfiler, function_stats, {}, struct {
            fn lessThan(_: void, a: FunctionProfiler, b: FunctionProfiler) bool {
                return a.total_time > b.total_time;
            }
        }.lessThan);

        const top_count = @min(10, function_stats.len);
        for (function_stats[0..top_count], 0..) |func, i| {
            const total_ms = @as(f64, @floatFromInt(func.total_time)) / 1_000_000.0;
            const avg_ns = func.averageExecutionTime();
            try writer.print("  {d}. {s} ({s}:{d})\n", .{ i + 1, func.function_name, func.file_name, func.line_number });
            try writer.print("     Calls: {d}, Total: {d:.3}ms, Avg: {d}ns, Min: {d}ns, Max: {d}ns\n", .{
                func.call_count, total_ms, avg_ns, func.min_time, func.max_time,
            });
        }

        // Call tree analysis (simplified)
        if (profile.call_records.items.len > 0) {
            try writer.print("\n=== Call Tree Analysis ===\n", .{});
            try writer.print("Total function calls: {d}\n", .{profile.call_records.items.len});

            var completed_calls: usize = 0;
            var total_call_time: u64 = 0;

            for (profile.call_records.items) |record| {
                if (record.isComplete()) {
                    completed_calls += 1;
                    total_call_time += record.duration();
                }
            }

            try writer.print("Completed calls: {d}\n", .{completed_calls});
            try writer.print("Average call duration: {d} ns\n", .{total_call_time / @max(1, completed_calls)});
        }

        try writer.print("\n=== End Report ===\n", .{});

        return try report.toOwnedSlice(self.allocator);
    }

    /// Start profiling thread for periodic sampling
    fn startProfilingThread(self: *PerformanceProfiler) void {
        if (self.profiling_thread != null) return;

        self.stop_profiling = false;
        self.profiling_thread = std.Thread.spawn(.{}, profilingLoop, .{self}) catch null;
    }

    /// Stop profiling thread
    pub fn stop(self: *PerformanceProfiler) void {
        if (self.profiling_thread) |thread| {
            self.stop_profiling = true;
            thread.join();
            self.profiling_thread = null;
        }
    }

    /// Profiling thread loop
    fn profilingLoop(self: *PerformanceProfiler) void {
        var last_report_time = std.time.nanoTimestamp();

        while (!self.stop_profiling) {
            // Periodic sampling (if implemented)
            // This would collect CPU samples, stack traces, etc.

            // Generate periodic reports
            const current_time = std.time.nanoTimestamp();
            if (current_time - last_report_time >= self.config.report_interval_ns) {
                // Generate and log report
                if (self.current_profile) |*profile| {
                    const report = self.generateProfileReport(profile) catch continue;
                    defer self.allocator.free(report);

                    std.log.info("Periodic performance report:\n{s}", .{report});
                }
                last_report_time = current_time;
            }

            std.time.sleep(self.config.sampling_interval_ns);
        }
    }

    /// Integrate with memory tracker
    pub fn setMemoryTracker(self: *PerformanceProfiler, tracker: *memory_tracker.MemoryProfiler) void {
        self.memory_tracker = tracker;
    }

    /// Create performance scope for measuring code blocks
    pub fn createScope(self: *PerformanceProfiler, name: []const u8) Scope {
        return .{
            .profiler = self,
            .name = name,
            .start_time = std.time.nanoTimestamp(),
            .memory_start = if (self.memory_tracker) |tracker| tracker.getStats().currentUsage() else 0,
        };
    }
};

/// Performance measurement scope
pub const Scope = struct {
    /// Associated profiler
    profiler: *PerformanceProfiler,
    /// Scope name
    name: []const u8,
    /// Start timestamp
    start_time: u64,
    /// Memory usage at start
    memory_start: usize,

    /// End the scope and record measurements
    pub fn end(self: Scope) void {
        const end_time = std.time.nanoTimestamp();
        const duration = end_time - self.start_time;

        // Update performance counters
        self.profiler.updateCounter("cpu_time", duration);

        // Log scope information
        std.log.debug("Performance scope '{s}': {d} ns", .{ self.name, duration });

        // Memory tracking
        if (self.profiler.memory_tracker) |tracker| {
            const memory_end = tracker.getStats().currentUsage();
            const memory_delta = if (memory_end > self.memory_start) memory_end - self.memory_start else 0;
            if (memory_delta > 0) {
                std.log.debug("  Memory delta: +{d} bytes", .{memory_delta});
            }
        }
    }
};

/// Global performance profiler instance
var global_profiler: ?*PerformanceProfiler = null;

/// Initialize global performance profiler
pub fn initGlobalProfiler(allocator: std.mem.Allocator, config: ProfilingConfig) !void {
    if (global_profiler != null) {
        deinitGlobalProfiler();
    }
    global_profiler = try PerformanceProfiler.init(allocator, config);
}

/// Deinitialize global performance profiler
pub fn deinitGlobalProfiler() void {
    if (global_profiler) |profiler| {
        profiler.deinit();
        global_profiler = null;
    }
}

/// Get global performance profiler instance
pub fn getGlobalProfiler() ?*PerformanceProfiler {
    return global_profiler;
}

/// Convenience function to start a performance scope
pub fn startScope(name: []const u8) ?Scope {
    if (global_profiler) |profiler| {
        return profiler.createScope(name);
    }
    return null;
}

/// Convenience macro for profiling function calls (would be implemented as a macro in real usage)
pub fn profileFunctionCall(profiler: ?*PerformanceProfiler, function_name: []const u8, file: []const u8, line: u32) FunctionCall {
    return .{
        .profiler = profiler,
        .function_name = function_name,
        .file = file,
        .line = line,
        .entry_time = if (profiler) |p| p.startFunctionCall(function_name, file, line) catch 0 else 0,
    };
}

/// Function call scope for automatic profiling
pub const FunctionCall = struct {
    profiler: ?*PerformanceProfiler,
    function_name: []const u8,
    file: []const u8,
    line: u32,
    entry_time: u64,

    pub fn end(self: FunctionCall) void {
        if (self.profiler) |profiler| {
            profiler.endFunctionCall(self.entry_time);
        }
    }
};

/// Performance monitoring utilities
pub const utils = struct {
    /// Create a development profiling configuration
    pub fn developmentConfig() ProfilingConfig {
        return .{
            .enable_cpu_profiling = true,
            .enable_call_tracing = true,
            .enable_counters = true,
            .sampling_interval_ns = 100_000, // 100μs
            .max_stack_depth = 64,
            .enable_memory_integration = true,
            .report_interval_ns = 5_000_000_000, // 5 seconds
        };
    }

    /// Create a production profiling configuration
    pub fn productionConfig() ProfilingConfig {
        return .{
            .enable_cpu_profiling = false,
            .enable_call_tracing = false,
            .enable_counters = true,
            .sampling_interval_ns = 10_000_000, // 10ms
            .max_stack_depth = 32,
            .enable_memory_integration = true,
            .report_interval_ns = 60_000_000_000, // 60 seconds
        };
    }

    /// Create a minimal profiling configuration
    pub fn minimalConfig() ProfilingConfig {
        return .{
            .enable_cpu_profiling = false,
            .enable_call_tracing = false,
            .enable_counters = false,
            .sampling_interval_ns = 1_000_000_000, // 1 second
            .max_stack_depth = 8,
            .enable_memory_integration = false,
            .report_interval_ns = 300_000_000_000, // 5 minutes
        };
    }
};
