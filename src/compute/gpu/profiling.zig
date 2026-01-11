//! GPU profiling and performance metrics.
//!
//! Provides timing, occupancy calculation, and performance counters
//! for GPU operations and kernel launches.

const std = @import("std");
const time = @import("../../shared/utils/time.zig");

pub const ProfilingError = error{
    TimerFailed,
    CounterFailed,
    InvalidState,
    NotSupported,
};

pub const TimingInfo = struct {
    kernel_name: []const u8,
    duration_ms: f64,
    duration_ns: u64,
    start_time: i64,
    end_time: i64,
    device_id: i32,
    stream_id: ?u64 = null,

    pub fn format(
        self: TimingInfo,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;

        try writer.print("{s}: {d:.3} ms ({d} ns)", .{
            self.kernel_name,
            self.duration_ms,
            self.duration_ns,
        });

        if (self.stream_id) |sid| {
            try writer.print(" [Stream: {d}]", .{sid});
        }
    }
};

pub const OccupancyInfo = struct {
    occupancy_percentage: f32,
    active_warps_per_multiprocessor: i32,
    active_threads_per_multiprocessor: i32,
    warp_size: i32,
    max_blocks_per_multiprocessor: i32,
    min_blocks_per_multiprocessor: i32,
    max_warps_per_multiprocessor: i32,

    pub fn isGood(self: OccupancyInfo) bool {
        return self.occupancy_percentage >= 75.0;
    }

    pub fn isExcellent(self: OccupancyInfo) bool {
        return self.occupancy_percentage >= 90.0;
    }

    pub fn format(
        self: OccupancyInfo,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;

        try writer.print("Occupancy: {d:.1}%\n", .{self.occupancy_percentage});
        try writer.print("  Active Warps/SM: {d}/{d}\n", .{
            self.active_warps_per_multiprocessor,
            self.max_warps_per_multiprocessor,
        });
        try writer.print("  Active Threads/SM: {d}\n", .{self.active_threads_per_multiprocessor});
        try writer.print("  Warp Size: {d}\n", .{self.warp_size});
    }
};

pub const MemoryThroughput = struct {
    host_to_device_bytes_per_sec: f64,
    device_to_host_bytes_per_sec: f64,
    device_to_device_bytes_per_sec: f64,
    host_to_device_mb_per_sec: f64,
    device_to_host_mb_per_sec: f64,
    device_to_device_mb_per_sec: f64,
    total_bytes_transferred: u64,

    pub fn format(
        self: MemoryThroughput,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;

        try writer.print("Memory Throughput:\n", .{});
        try writer.print("  Host->Device: {d:.2} MB/s ({d:.2} GB/s)\n", .{
            self.host_to_device_mb_per_sec,
            self.host_to_device_bytes_per_sec / 1e9,
        });
        try writer.print("  Device->Host: {d:.2} MB/s ({d:.2} GB/s)\n", .{
            self.device_to_host_mb_per_sec,
            self.device_to_host_bytes_per_sec / 1e9,
        });
        try writer.print("  Device->Device: {d:.2} MB/s ({d:.2} GB/s)\n", .{
            self.device_to_device_mb_per_sec,
            self.device_to_device_bytes_per_sec / 1e9,
        });
    }
};

pub const Profiler = struct {
    timings: std.ArrayListUnmanaged(TimingInfo),
    memory_transfers: std.ArrayListUnmanaged(MemoryThroughput),
    enabled: bool,
    start_timer: ?std.time.Timer,
    current_stream_id: ?u64,

    pub fn init(allocator: std.mem.Allocator) Profiler {
        _ = allocator;
        return .{
            .timings = std.ArrayListUnmanaged(TimingInfo).empty,
            .memory_transfers = std.ArrayListUnmanaged(MemoryThroughput).empty,
            .enabled = false,
            .start_timer = null,
            .current_stream_id = null,
        };
    }

    pub fn deinit(self: *Profiler, allocator: std.mem.Allocator) void {
        for (self.timings.items) |*timing| {
            allocator.free(timing.kernel_name);
        }
        self.timings.deinit(allocator);
        self.memory_transfers.deinit(allocator);
        self.* = undefined;
    }

    pub fn enable(self: *Profiler) void {
        self.enabled = true;
        self.start_timer = std.time.Timer.start() catch null;
    }

    pub fn disable(self: *Profiler) void {
        self.enabled = false;
    }

    pub fn startTiming(self: *Profiler, kernel_name: []const u8, allocator: std.mem.Allocator, device_id: i32) !void {
        if (!self.enabled) return;

        const name_copy = try allocator.dupe(u8, kernel_name);

        try self.timings.append(allocator, .{
            .kernel_name = name_copy,
            .start_time = @intCast(time.nowNanoseconds()),
            .device_id = device_id,
            .stream_id = self.current_stream_id,
            .duration_ms = 0.0,
            .duration_ns = 0,
            .end_time = 0,
        });
    }

    pub fn endTiming(self: *Profiler, allocator: std.mem.Allocator) !void {
        _ = allocator;
        if (!self.enabled or self.timings.items.len == 0) return;

        const timing = &self.timings.items[self.timings.items.len - 1];
        const end_time: i64 = @intCast(time.nowNanoseconds());
        timing.end_time = end_time;

        const duration_ns = if (end_time >= timing.start_time)
            @as(u64, @intCast(end_time - timing.start_time))
        else
            0;

        timing.duration_ns = duration_ns;
        timing.duration_ms = @as(f64, @floatFromInt(duration_ns)) / 1_000_000.0;
    }

    pub fn recordMemoryTransfer(
        self: *Profiler,
        allocator: std.mem.Allocator,
        h2d_time_ms: f64,
        d2h_time_ms: f64,
        bytes: usize,
    ) !void {
        if (!self.enabled) return;

        const h2d_bps = if (h2d_time_ms > 0)
            @as(f64, @floatFromInt(bytes)) / (h2d_time_ms / 1000.0)
        else
            0.0;

        const d2h_bps = if (d2h_time_ms > 0)
            @as(f64, @floatFromInt(bytes)) / (d2h_time_ms / 1000.0)
        else
            0.0;

        try self.memory_transfers.append(allocator, .{
            .host_to_device_bytes_per_sec = h2d_bps,
            .device_to_host_bytes_per_sec = d2h_bps,
            .device_to_device_bytes_per_sec = 0.0,
            .host_to_device_mb_per_sec = h2d_bps / (1024.0 * 1024.0),
            .device_to_host_mb_per_sec = d2h_bps / (1024.0 * 1024.0),
            .device_to_device_mb_per_sec = 0.0,
            .total_bytes_transferred = @intCast(bytes),
        });
    }

    pub fn getAverageTime(self: *const Profiler, kernel_name: []const u8) ?f64 {
        if (self.timings.items.len == 0) return null;

        var sum: f64 = 0.0;
        var count: usize = 0;

        for (self.timings.items) |timing| {
            if (std.mem.eql(u8, timing.kernel_name, kernel_name)) {
                sum += timing.duration_ms;
                count += 1;
            }
        }

        if (count == 0) return null;
        return sum / @as(f64, @floatFromInt(count));
    }

    pub fn getTotalTime(self: *const Profiler) f64 {
        var total: f64 = 0.0;
        for (self.timings.items) |timing| {
            total += timing.duration_ms;
        }
        return total;
    }

    pub fn getSummary(self: *const Profiler) struct {
        total_kernels: usize,
        total_time_ms: f64,
        average_time_ms: f64,
        max_time_ms: f64,
        min_time_ms: f64,
        total_bytes_transferred: u64,
    } {
        if (self.timings.items.len == 0) {
            return .{
                .total_kernels = 0,
                .total_time_ms = 0.0,
                .average_time_ms = 0.0,
                .max_time_ms = 0.0,
                .min_time_ms = 0.0,
                .total_bytes_transferred = 0,
            };
        }

        var total: f64 = 0.0;
        var max: f64 = 0.0;
        var min: f64 = std.math.floatMax(f64);

        for (self.timings.items) |timing| {
            total += timing.duration_ms;
            if (timing.duration_ms > max) max = timing.duration_ms;
            if (timing.duration_ms < min) min = timing.duration_ms;
        }

        var total_bytes: u64 = 0;
        for (self.memory_transfers.items) |transfer| {
            total_bytes += transfer.total_bytes_transferred;
        }

        return .{
            .total_kernels = self.timings.items.len,
            .total_time_ms = total,
            .average_time_ms = total / @as(f64, @floatFromInt(self.timings.items.len)),
            .max_time_ms = max,
            .min_time_ms = min,
            .total_bytes_transferred = total_bytes,
        };
    }
};

pub fn calculateOccupancy(
    block_size: u32,
    shared_mem_per_block: u32,
    device_max_threads: u32,
    device_shared_mem: u32,
) OccupancyInfo {
    const blocks_per_sm = @min(
        device_max_threads / block_size,
        if (device_shared_mem > 0)
            @divFloor(device_shared_mem, shared_mem_per_block)
        else
            999,
    );

    const threads_per_sm = blocks_per_sm * block_size;
    const occupancy = if (device_max_threads > 0)
        @as(f32, @floatFromInt(threads_per_sm)) * 100.0 / @as(f32, @floatFromInt(device_max_threads))
    else
        0.0;

    const warp_size: u32 = 32;
    const warps_per_sm = threads_per_sm / warp_size;
    const max_warps_per_sm = device_max_threads / warp_size;

    return .{
        .occupancy_percentage = occupancy,
        .active_warps_per_multiprocessor = @intCast(warps_per_sm),
        .active_threads_per_multiprocessor = @intCast(threads_per_sm),
        .warp_size = @intCast(warp_size),
        .max_blocks_per_multiprocessor = @intCast(blocks_per_sm),
        .min_blocks_per_multiprocessor = 0,
        .max_warps_per_multiprocessor = @intCast(max_warps_per_sm),
    };
}

pub fn formatSummary(profiler: *const Profiler, writer: anytype) !void {
    const summary = profiler.getSummary();

    try writer.print("GPU Profiling Summary\n", .{});
    try writer.print("=====================\n", .{});
    try writer.print("Total Kernels: {d}\n", .{summary.total_kernels});
    try writer.print("Total Time: {d:.3} ms\n", .{summary.total_time_ms});
    try writer.print("Average Time: {d:.3} ms\n", .{summary.average_time_ms});
    try writer.print("Max Time: {d:.3} ms\n", .{summary.max_time_ms});
    try writer.print("Min Time: {d:.3} ms\n", .{summary.min_time_ms});

    if (summary.total_bytes_transferred > 0) {
        try writer.print("Total Data Transferred: {d:.2} MB\n", .{
            @as(f64, @floatFromInt(summary.total_bytes_transferred)) / (1024.0 * 1024.0),
        });
    }

    if (profiler.memory_transfers.items.len > 0) {
        const avg_transfer = profiler.memory_transfers.items[profiler.memory_transfers.items.len - 1];
        try writer.print("Avg H2D Throughput: {d:.2} MB/s\n", .{avg_transfer.host_to_device_mb_per_sec});
        try writer.print("Avg D2H Throughput: {d:.2} MB/s\n", .{avg_transfer.device_to_host_mb_per_sec});
    }
}

test "occupancy calculation" {
    const occupancy = calculateOccupancy(256, 0, 1024, 49152);

    try std.testing.expect(occupancy.occupancy_percentage > 0.0);
    try std.testing.expectEqual(@as(i32, 32), occupancy.warp_size);
}

test "profiler tracks timings" {
    const allocator = std.testing.allocator;
    var profiler = Profiler.init(allocator);
    defer profiler.deinit(allocator);

    profiler.enable();

    try profiler.startTiming("test_kernel", allocator, 0);

    var i: usize = 0;
    while (i < 100) : (i += 1) {
        std.atomic.spinLoopHint();
    }

    try profiler.endTiming(allocator);

    const summary = profiler.getSummary();
    try std.testing.expectEqual(@as(usize, 1), summary.total_kernels);
    try std.testing.expect(summary.total_time_ms > 0.0);
}

