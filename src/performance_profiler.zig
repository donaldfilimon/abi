//! Performance Profiling Infrastructure
//!
//! This module provides comprehensive and extensible performance profiling capabilities including:
//! - CPU performance monitoring (sampling and counters)
//! - Function call tracing and call tree analysis
//! - Performance counters (custom and built-in)
//! - Hot path and bottleneck analysis
//! - Integration with memory tracking
//! - Thread-aware profiling
//! - Flexible configuration for development and production

const std = @import("std");
const memory_tracker = @import("memory_tracker.zig");

/// Performance profiling configuration
pub const ProfilingConfig = struct {
    /// Enable CPU profiling (sampling and counters)
    enable_cpu_profiling: bool = true,
    /// Enable function call tracing (call stacks, call tree)
    enable_call_tracing: bool = false,
    /// Enable performance counters (custom and built-in)
    enable_counters: bool = true,
    /// Sampling interval (nanoseconds) for periodic sampling
    sampling_interval_ns: u64 = 1_000_000, // 1ms
    /// Maximum number of call stack frames to record
    max_stack_depth: usize = 64,
    /// Enable memory integration (track allocations/deallocations)
    enable_memory_integration: bool = true,
    /// Performance report output interval (nanoseconds)
    report_interval_ns: u64 = 10_000_000_000, // 10 seconds
    /// Maximum number of function profilers to keep
    max_function_profilers: usize = 1024,
    /// Maximum number of call records to keep in a session
    max_call_records: usize = 10_000,
};

/// Function call record (for call tracing and call tree)
pub const CallRecord = struct {
    function_name: []const u8,
    file: []const u8,
    line: u32,
    entry_time: u64,
    exit_time: u64 = 0,
    depth: u32,
    parent_id: ?u64,
    call_id: u64,
    thread_id: std.Thread.Id,

    /// Calculate call duration (nanoseconds)
    pub fn duration(self: CallRecord) u64 {
        if (self.exit_time == 0) return 0;
        return self.exit_time - self.entry_time;
    }

    /// Check if call is complete (has exit time)
    pub fn isComplete(self: CallRecord) bool {
        return self.exit_time != 0;
    }
};

/// Performance counter (for custom and built-in metrics)
pub const PerformanceCounter = struct {
    name: []const u8,
    value: u64 = 0,
    unit: []const u8,
    description: []const u8,
    last_update: u64 = 0,

    pub fn increment(self: *PerformanceCounter) void {
        self.value += 1;
        self.last_update = std.time.nanoTimestamp();
    }

    pub fn add(self: *PerformanceCounter, delta: u64) void {
        self.value += delta;
        self.last_update = std.time.nanoTimestamp();
    }

    pub fn set(self: *PerformanceCounter, new_value: u64) void {
        self.value = new_value;
        self.last_update = std.time.nanoTimestamp();
    }

    pub fn reset(self: *PerformanceCounter) void {
        self.value = 0;
        self.last_update = std.time.nanoTimestamp();
    }
};

/// Performance profile data (per session)
pub const PerformanceProfile = struct {
    total_time: u64 = 0,
    cpu_time: u64 = 0,
    allocations: u64 = 0,
    deallocations: u64 = 0,
    peak_memory: usize = 0,
    call_records: std.ArrayListUnmanaged(CallRecord),
    counters: std.StringHashMapUnmanaged(PerformanceCounter),
    start_time: u64,
    end_time: u64 = 0,
    session_name: []const u8 = "",

    pub fn duration(self: PerformanceProfile) u64 {
        if (self.end_time == 0) return std.time.nanoTimestamp() - self.start_time;
        return self.end_time - self.start_time;
    }

    pub fn durationSeconds(self: PerformanceProfile) f64 {
        return @as(f64, @floatFromInt(self.duration())) / 1_000_000_000.0;
    }

    pub fn cpuUtilization(self: PerformanceProfile) f64 {
        const total_duration = self.duration();
        if (total_duration == 0) return 0.0;
        return @as(f64, @floatFromInt(self.cpu_time)) / @as(f64, @floatFromInt(total_duration));
    }
};

/// Function profiler for instrumenting and aggregating function stats
pub const FunctionProfiler = struct {
    function_name: []const u8,
    file_name: []const u8,
    line_number: u32,
    call_count: u64 = 0,
    total_time: u64 = 0,
    min_time: u64 = std.math.maxInt(u64),
    max_time: u64 = 0,
    average_time: f64 = 0,
    last_call_time: u64 = 0,

    pub fn enter(self: *FunctionProfiler) u64 {
        const entry_time = std.time.nanoTimestamp();
        self.call_count += 1;
        self.last_call_time = entry_time;
        return entry_time;
    }

    pub fn exit(self: *FunctionProfiler, entry_time: u64) void {
        const exit_time = std.time.nanoTimestamp();
        const duration = exit_time - entry_time;

        self.total_time += duration;

        if (duration < self.min_time) self.min_time = duration;
        if (duration > self.max_time) self.max_time = duration;

        // Exponential moving average for average_time
        if (self.call_count == 1) {
            self.average_time = @as(f64, @floatFromInt(duration));
        } else {
            const alpha = 0.1;
            self.average_time = self.average_time * (1.0 - alpha) + @as(f64, @floatFromInt(duration)) * alpha;
        }
    }

    pub fn averageExecutionTime(self: FunctionProfiler) u64 {
        if (self.call_count == 0) return 0;
        return @intFromFloat(self.average_time);
    }
};

/// Main performance profiler
pub const PerformanceProfiler = struct {
    config: ProfilingConfig,
    allocator: std.mem.Allocator,
    function_profilers: std.StringHashMapUnmanaged(FunctionProfiler),
    current_profile: ?PerformanceProfile = null,
    call_stack: std.ArrayListUnmanaged(CallRecord),
    counters: std.StringHashMapUnmanaged(PerformanceCounter),
    next_call_id: u64 = 1,
    memory_tracker: ?*memory_tracker.MemoryProfiler = null,
    mutex: std.Thread.Mutex = .{},
    profiling_thread: ?std.Thread = null,
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

        try self.initDefaultCounters();
        return self;
    }

    /// Deinitialize performance profiler and free all resources
    pub fn deinit(self: *PerformanceProfiler) void {
        self.stop();

        self.mutex.lock();
        defer self.mutex.unlock();

        // Clean up function profilers
        var profiler_iter = self.function_profilers.iterator();
        while (profiler_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.function_name);
            self.allocator.free(entry.value_ptr.file_name);
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
            if (profile.session_name.len > 0) self.allocator.free(profile.session_name);
        }

        // Clean up call stack
        for (self.call_stack.items) |record| {
            self.allocator.free(record.function_name);
            self.allocator.free(record.file);
        }
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
            .{ .name = "cpu_time", .unit = "ns", .description = "CPU time measured" },
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
        const session_name_copy = try self.allocator.dupe(u8, session_name);

        self.current_profile = .{
            .call_records = try std.ArrayListUnmanaged(CallRecord).initCapacity(self.allocator, self.config.max_call_records),
            .counters = std.StringHashMapUnmanaged(PerformanceCounter){},
            .start_time = start_time,
            .session_name = session_name_copy,
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

    /// End profiling session and return report
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
        for (profile.call_records.items) |record| {
            self.allocator.free(record.function_name);
            self.allocator.free(record.file);
        }
        profile.call_records.deinit(self.allocator);
        var counter_iter = profile.counters.iterator();
        while (counter_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        profile.counters.deinit(self.allocator);
        if (profile.session_name.len > 0) self.allocator.free(profile.session_name);

        self.current_profile = null;

        return report;
    }

    /// Start function call (for call tracing)
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

    /// End function call (for call tracing)
    pub fn endFunctionCall(self: *PerformanceProfiler, entry_time: u64) void {
        if (!self.config.enable_call_tracing) return;

        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.call_stack.items.len == 0) return;

        const record = self.call_stack.pop();
        const exit_time = std.time.nanoTimestamp();

        // Update record in current profile
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

    /// Update or create a performance counter
    pub fn updateCounter(self: *PerformanceProfiler, name: []const u8, delta: u64) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.counters.getPtr(name)) |counter| {
            counter.add(delta);
        } else if (self.config.enable_counters) {
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

    /// Get function profiler statistics (sorted by total_time descending)
    pub fn getFunctionStats(self: *PerformanceProfiler, allocator: std.mem.Allocator) ![]FunctionProfiler {
        self.mutex.lock();
        defer self.mutex.unlock();

        var stats = std.ArrayListUnmanaged(FunctionProfiler){};
        errdefer stats.deinit(allocator);

        var iter = self.function_profilers.iterator();
        while (iter.next()) |entry| {
            try stats.append(allocator, entry.value_ptr.*);
        }

        // Sort by total_time descending
        std.sort.insertion(FunctionProfiler, stats.items, {}, struct {
            fn lessThan(_: void, a: FunctionProfiler, b: FunctionProfiler) bool {
                return a.total_time > b.total_time;
            }
        }.lessThan);

        return try stats.toOwnedSlice(allocator);
    }

    /// Generate profiling report (returns owned slice)
    fn generateProfileReport(self: *PerformanceProfiler, profile: *PerformanceProfile) ![]u8 {
        var report = std.ArrayListUnmanaged(u8){};
        errdefer report.deinit(self.allocator);

        const writer = report.writer(self.allocator);

        try writer.print("=== Performance Profile Report ===\n", .{});
        if (profile.session_name.len > 0)
            try writer.print("Session: {s}\n", .{profile.session_name});
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
            try writer.print("Average call duration: {d} ns\n", .{if (completed_calls > 0) total_call_time / completed_calls else 0});
        }

        try writer.print("\n=== End Report ===\n", .{});

        return try report.toOwnedSlice(self.allocator);
    }

    /// Start profiling thread for periodic sampling/reporting
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

    /// Profiling thread loop (periodic sampling/reporting)
    fn profilingLoop(self: *PerformanceProfiler) void {
        var last_report_time = std.time.nanoTimestamp();

        while (!self.stop_profiling) {
            // TODO: Implement periodic sampling (CPU, stack, etc.)

            // Generate periodic reports
            const current_time = std.time.nanoTimestamp();
            if (current_time - last_report_time >= self.config.report_interval_ns) {
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

/// Performance measurement scope (RAII-style)
pub const Scope = struct {
    profiler: *PerformanceProfiler,
    name: []const u8,
    start_time: u64,
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

/// Global performance profiler instance (singleton)
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

/// Convenience function for profiling function calls (to be used with defer)
pub fn profileFunctionCall(profiler: ?*PerformanceProfiler, function_name: []const u8, file: []const u8, line: u32) FunctionCall {
    return .{
        .profiler = profiler,
        .function_name = function_name,
        .file = file,
        .line = line,
        .entry_time = if (profiler) |p| p.startFunctionCall(function_name, file, line) catch 0 else 0,
    };
}

/// Function call scope for automatic profiling (RAII-style)
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

/// Performance monitoring utilities and presets
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
            .max_function_profilers = 2048,
            .max_call_records = 100_000,
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
            .max_function_profilers = 512,
            .max_call_records = 10_000,
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
            .max_function_profilers = 64,
            .max_call_records = 100,
        };
    }
};
