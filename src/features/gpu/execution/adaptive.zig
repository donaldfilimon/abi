//! Adaptive Threshold Manager
//!
//! Learns optimal GPU/SIMD thresholds from runtime performance samples.
//! Uses statistical analysis with outlier rejection and EMA smoothing.

const std = @import("std");
const coordinator = @import("../internal/execution_coordinator.zig");

const OperationType = coordinator.OperationType;
const ExecutionMethod = coordinator.ExecutionMethod;
const CoordinatorConfig = coordinator.CoordinatorConfig;

/// Performance sample for adaptive threshold learning
pub const PerformanceSample = struct {
    size: usize,
    method: ExecutionMethod,
    time_ns: u64,
    operation: OperationType,
};

/// Adaptive threshold manager
pub const AdaptiveThresholds = struct {
    allocator: std.mem.Allocator,
    /// Performance samples per operation type
    samples: std.AutoHashMapUnmanaged(OperationType, std.ArrayListUnmanaged(PerformanceSample)),
    /// Learned thresholds per operation type
    gpu_thresholds: std.AutoHashMapUnmanaged(OperationType, usize),
    simd_thresholds: std.AutoHashMapUnmanaged(OperationType, usize),
    /// Default thresholds
    default_gpu_threshold: usize,
    default_simd_threshold: usize,
    /// Configuration
    sample_window: usize,
    min_improvement: f64,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, config: CoordinatorConfig) Self {
        return .{
            .allocator = allocator,
            .samples = .{},
            .gpu_thresholds = .{},
            .simd_thresholds = .{},
            .default_gpu_threshold = config.gpu_threshold_size,
            .default_simd_threshold = config.simd_threshold_size,
            .sample_window = config.adaptive_sample_window,
            .min_improvement = config.adaptive_min_improvement,
        };
    }

    pub fn deinit(self: *Self) void {
        var samples_iter = self.samples.iterator();
        while (samples_iter.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
        }
        self.samples.deinit(self.allocator);
        self.gpu_thresholds.deinit(self.allocator);
        self.simd_thresholds.deinit(self.allocator);
    }

    /// Record a performance sample
    pub fn recordSample(self: *Self, sample: PerformanceSample) !void {
        const entry = try self.samples.getOrPut(self.allocator, sample.operation);
        if (!entry.found_existing) {
            entry.value_ptr.* = .{};
        }

        // Keep only recent samples
        if (entry.value_ptr.items.len >= self.sample_window) {
            _ = entry.value_ptr.orderedRemove(0);
        }

        try entry.value_ptr.append(self.allocator, sample);

        // Recalculate thresholds periodically
        if (entry.value_ptr.items.len % 10 == 0) {
            self.recalculateThreshold(sample.operation);
        }
    }

    /// Get learned threshold for GPU execution
    pub fn getGpuThreshold(self: *const Self, op: OperationType) usize {
        return self.gpu_thresholds.get(op) orelse self.default_gpu_threshold;
    }

    /// Get learned threshold for SIMD execution
    pub fn getSimdThreshold(self: *const Self, op: OperationType) usize {
        return self.simd_thresholds.get(op) orelse self.default_simd_threshold;
    }

    /// Recalculate thresholds based on collected samples
    fn recalculateThreshold(self: *Self, op: OperationType) void {
        const samples_list = self.samples.get(op) orelse return;
        if (samples_list.items.len < 20) return; // Need enough samples

        // Group samples by size ranges and method
        var gpu_times: [16]struct { count: usize, total_ns: u64, sum_squares: u64 } =
            .{.{ .count = 0, .total_ns = 0, .sum_squares = 0 }} ** 16;
        var simd_times: [16]struct { count: usize, total_ns: u64, sum_squares: u64 } =
            .{.{ .count = 0, .total_ns = 0, .sum_squares = 0 }} ** 16;
        var scalar_times: [16]struct { count: usize, total_ns: u64, sum_squares: u64 } =
            .{.{ .count = 0, .total_ns = 0, .sum_squares = 0 }} ** 16;

        // First pass: collect statistics
        for (samples_list.items) |sample| {
            const bucket = @min(15, std.math.log2_int(usize, @max(1, sample.size / 64)));

            switch (sample.method) {
                .accelerate, .gpu => {
                    gpu_times[bucket].count += 1;
                    gpu_times[bucket].total_ns += sample.time_ns;
                    gpu_times[bucket].sum_squares += sample.time_ns * sample.time_ns;
                },
                .simd => {
                    simd_times[bucket].count += 1;
                    simd_times[bucket].total_ns += sample.time_ns;
                    simd_times[bucket].sum_squares += sample.time_ns * sample.time_ns;
                },
                .scalar => {
                    scalar_times[bucket].count += 1;
                    scalar_times[bucket].total_ns += sample.time_ns;
                    scalar_times[bucket].sum_squares += sample.time_ns * sample.time_ns;
                },
                .failed => {},
            }
        }

        // Find crossover points where GPU becomes faster than SIMD/scalar
        var gpu_threshold: usize = self.default_gpu_threshold;
        var simd_threshold: usize = self.default_simd_threshold;

        for (0..16) |bucket| {
            const size = @as(usize, 64) << @intCast(bucket);

            if (gpu_times[bucket].count > 2 and scalar_times[bucket].count > 2) {
                const gpu_avg = @as(f64, @floatFromInt(gpu_times[bucket].total_ns)) /
                    @as(f64, @floatFromInt(gpu_times[bucket].count));
                const scalar_avg = @as(f64, @floatFromInt(scalar_times[bucket].total_ns)) /
                    @as(f64, @floatFromInt(scalar_times[bucket].count));

                const gpu_variance = (@as(f64, @floatFromInt(gpu_times[bucket].sum_squares)) /
                    @as(f64, @floatFromInt(gpu_times[bucket].count))) -
                    (gpu_avg * gpu_avg);
                const gpu_stddev = if (gpu_variance > 0) @sqrt(gpu_variance) else 0.0;

                const scalar_variance = (@as(f64, @floatFromInt(scalar_times[bucket].sum_squares)) /
                    @as(f64, @floatFromInt(scalar_times[bucket].count))) -
                    (scalar_avg * scalar_avg);
                const scalar_stddev = if (scalar_variance > 0) @sqrt(scalar_variance) else 0.0;

                const gpu_outlier_threshold = gpu_avg + 2.0 * gpu_stddev;
                const scalar_outlier_threshold = scalar_avg + 2.0 * scalar_stddev;

                if (scalar_avg / gpu_avg >= self.min_improvement and
                    gpu_avg < gpu_outlier_threshold and
                    scalar_avg < scalar_outlier_threshold)
                {
                    gpu_threshold = @min(gpu_threshold, size);
                }
            }

            if (simd_times[bucket].count > 2 and scalar_times[bucket].count > 2) {
                const simd_avg = @as(f64, @floatFromInt(simd_times[bucket].total_ns)) /
                    @as(f64, @floatFromInt(simd_times[bucket].count));
                const scalar_avg = @as(f64, @floatFromInt(scalar_times[bucket].total_ns)) /
                    @as(f64, @floatFromInt(scalar_times[bucket].count));

                const simd_variance = (@as(f64, @floatFromInt(simd_times[bucket].sum_squares)) /
                    @as(f64, @floatFromInt(simd_times[bucket].count))) -
                    (simd_avg * simd_avg);
                const simd_stddev = if (simd_variance > 0) @sqrt(simd_variance) else 0.0;

                const scalar_variance = (@as(f64, @floatFromInt(scalar_times[bucket].sum_squares)) /
                    @as(f64, @floatFromInt(scalar_times[bucket].count))) -
                    (scalar_avg * scalar_avg);
                const scalar_stddev = if (scalar_variance > 0) @sqrt(scalar_variance) else 0.0;

                const simd_outlier_threshold = simd_avg + 2.0 * simd_stddev;
                const scalar_outlier_threshold_local = scalar_avg + 2.0 * scalar_stddev;
                if (scalar_avg / simd_avg >= self.min_improvement and
                    simd_avg < simd_outlier_threshold and
                    scalar_avg < scalar_outlier_threshold_local)
                {
                    simd_threshold = @min(simd_threshold, size);
                }
            }
        }

        // Update thresholds with smoothing to prevent rapid fluctuations
        const current_gpu_threshold = self.gpu_thresholds.get(op) orelse self.default_gpu_threshold;
        const current_simd_threshold = self.simd_thresholds.get(op) orelse self.default_simd_threshold;

        const smoothed_gpu_threshold = @as(usize, @intFromFloat(0.7 * @as(f64, @floatFromInt(gpu_threshold)) +
            0.3 * @as(f64, @floatFromInt(current_gpu_threshold))));

        const smoothed_simd_threshold = @as(usize, @intFromFloat(0.7 * @as(f64, @floatFromInt(simd_threshold)) +
            0.3 * @as(f64, @floatFromInt(current_simd_threshold))));

        self.gpu_thresholds.put(self.allocator, op, smoothed_gpu_threshold) catch |err| {
            std.log.debug("Failed to update GPU threshold for {t}: {t}", .{ op, err });
        };
        self.simd_thresholds.put(self.allocator, op, smoothed_simd_threshold) catch |err| {
            std.log.debug("Failed to update SIMD threshold for {t}: {t}", .{ op, err });
        };
    }

    /// Get adaptive threshold statistics
    pub fn getStats(self: *const Self) struct {
        total_samples: usize,
        operations_tracked: usize,
        gpu_threshold_adjustments: usize,
        simd_threshold_adjustments: usize,
    } {
        var total: usize = 0;
        var samples_iter = self.samples.iterator();
        while (samples_iter.next()) |entry| {
            total += entry.value_ptr.items.len;
        }

        return .{
            .total_samples = total,
            .operations_tracked = self.samples.count(),
            .gpu_threshold_adjustments = self.gpu_thresholds.count(),
            .simd_threshold_adjustments = self.simd_thresholds.count(),
        };
    }
};
