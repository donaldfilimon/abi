//! Unified Execution Coordinator
//!
//! Provides seamless fallback: GPU → SIMD → scalar
//! Automatically selects the best execution method based on:
//! - Hardware availability
//! - Data size
//! - Operation type
//! - User preferences
//!
//! ## Thread Safety
//!
//! `ExecutionCoordinator` is **NOT thread-safe**. Each thread should have its own
//! coordinator instance, or external synchronization must be used. The coordinator
//! holds mutable state (gpu_backend pointer) that is not protected by locks.
//!
//! For multi-threaded workloads, either:
//! 1. Create one coordinator per thread
//! 2. Use external locking (e.g., `sync.Mutex`) around all coordinator calls
//! 3. Use the stateless functions in `simd` module directly for thread-safe SIMD ops
//!
//! ## Example (Thread-Local)
//!
//! ```zig
//! threadlocal var coordinator: ?ExecutionCoordinator = null;
//!
//! fn getCoordinator(allocator: std.mem.Allocator) !*ExecutionCoordinator {
//!     if (coordinator == null) {
//!         coordinator = try ExecutionCoordinator.init(allocator, .{});
//!     }
//!     return &coordinator.?;
//! }
//! ```

const std = @import("std");
const sync = @import("../../services/shared/sync.zig");
const backend_factory = @import("backend_factory.zig");
const simd = @import("../../services/shared/simd.zig");
const dispatcher_mod = @import("dispatcher.zig");
const device_mod = @import("device.zig");
const unified_buffer = @import("unified_buffer.zig");

const KernelDispatcher = dispatcher_mod.KernelDispatcher;
const LaunchConfig = dispatcher_mod.LaunchConfig;
const KernelArgs = dispatcher_mod.KernelArgs;
const Buffer = unified_buffer.Buffer;
const Device = device_mod.Device;

pub const ExecutionMethod = enum {
    gpu,
    simd,
    scalar,
    failed,
};

pub const CoordinatorConfig = struct {
    prefer_gpu: bool = true,
    fallback_chain: []const ExecutionMethod = &.{ .gpu, .simd, .scalar },
    gpu_threshold_size: usize = 1024, // Min elements for GPU
    simd_threshold_size: usize = 4, // Min elements for SIMD
    backend_timeout_ms: u64 = 1000,
    /// Enable logging when fallback occurs (useful for debugging)
    log_fallbacks: bool = false,
    /// Enable adaptive threshold tuning based on runtime performance
    enable_adaptive_thresholds: bool = true,
    /// Sample window for adaptive threshold calculations
    adaptive_sample_window: usize = 100,
    /// Minimum improvement factor to change method (1.1 = 10% faster)
    adaptive_min_improvement: f64 = 1.1,
};

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
            // Map size to bucket (log2-ish)
            const bucket = @min(15, std.math.log2_int(usize, @max(1, sample.size / 64)));

            switch (sample.method) {
                .gpu => {
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

                // Calculate standard deviations for outlier detection
                const gpu_variance = (@as(f64, @floatFromInt(gpu_times[bucket].sum_squares)) /
                    @as(f64, @floatFromInt(gpu_times[bucket].count))) -
                    (gpu_avg * gpu_avg);
                const gpu_stddev = if (gpu_variance > 0) @sqrt(gpu_variance) else 0.0;

                const scalar_variance = (@as(f64, @floatFromInt(scalar_times[bucket].sum_squares)) /
                    @as(f64, @floatFromInt(scalar_times[bucket].count))) -
                    (scalar_avg * scalar_avg);
                const scalar_stddev = if (scalar_variance > 0) @sqrt(scalar_variance) else 0.0;

                // Reject outliers that are more than 2 standard deviations away
                const gpu_outlier_threshold = gpu_avg + 2.0 * gpu_stddev;
                const scalar_outlier_threshold = scalar_avg + 2.0 * scalar_stddev;

                // GPU is beneficial if significantly faster and within normal variation
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

                // Calculate standard deviations for outlier detection
                const simd_variance = (@as(f64, @floatFromInt(simd_times[bucket].sum_squares)) /
                    @as(f64, @floatFromInt(simd_times[bucket].count))) -
                    (simd_avg * simd_avg);
                const simd_stddev = if (simd_variance > 0) @sqrt(simd_variance) else 0.0;

                const scalar_variance = (@as(f64, @floatFromInt(scalar_times[bucket].sum_squares)) /
                    @as(f64, @floatFromInt(scalar_times[bucket].count))) -
                    (scalar_avg * scalar_avg);
                const scalar_stddev = if (scalar_variance > 0) @sqrt(scalar_variance) else 0.0;

                // SIMD is beneficial if significantly faster and within normal variation
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

        // Apply exponential moving average with 0.7 weight for smoothing
        const smoothed_gpu_threshold = @as(usize, @intFromFloat(0.7 * @as(f64, @floatFromInt(gpu_threshold)) +
            0.3 * @as(f64, @floatFromInt(current_gpu_threshold))));

        const smoothed_simd_threshold = @as(usize, @intFromFloat(0.7 * @as(f64, @floatFromInt(simd_threshold)) +
            0.3 * @as(f64, @floatFromInt(current_simd_threshold))));

        // Update thresholds
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

pub const ExecutionCoordinator = struct {
    allocator: std.mem.Allocator,
    config: CoordinatorConfig,
    gpu_backend: ?*backend_factory.BackendInstance = null,
    gpu_available: bool = false,
    simd_available: bool = false,
    dispatcher: ?KernelDispatcher = null,
    device: ?Device = null,
    /// Adaptive thresholds for method selection
    adaptive: ?AdaptiveThresholds = null,
    /// Performance statistics
    total_gpu_ops: u64 = 0,
    total_simd_ops: u64 = 0,
    total_scalar_ops: u64 = 0,
    total_time_ns: u64 = 0,

    pub fn init(allocator: std.mem.Allocator, config: CoordinatorConfig) !ExecutionCoordinator {
        var coord = ExecutionCoordinator{
            .allocator = allocator,
            .config = config,
            .simd_available = simd.hasSimdSupport(),
            .adaptive = if (config.enable_adaptive_thresholds)
                AdaptiveThresholds.init(allocator, config)
            else
                null,
        };

        // Try to initialize GPU
        if (config.prefer_gpu) {
            coord.gpu_backend = backend_factory.createBestBackend(allocator) catch null;
            coord.gpu_available = coord.gpu_backend != null;

            // Initialize dispatcher if GPU is available
            if (coord.gpu_backend) |backend| {
                // Create a device representation for the dispatcher
                coord.device = Device{
                    .id = 0,
                    .backend = backend.backend_type,
                    .name = "GPU Device",
                    .device_type = .discrete,
                    .total_memory = backend.total_memory,
                    .available_memory = null,
                    .is_emulated = backend.is_emulated,
                    .capability = .{},
                    .compute_units = null,
                    .clock_mhz = null,
                };

                // Initialize the kernel dispatcher
                coord.dispatcher = KernelDispatcher.init(
                    allocator,
                    backend.backend_type,
                    &coord.device.?,
                ) catch null;

                // Set the backend interface if dispatcher was created
                if (coord.dispatcher != null) {
                    coord.dispatcher.?.setBackendInterface(backend.backend);
                }
            }
        }

        return coord;
    }

    pub fn deinit(self: *ExecutionCoordinator) void {
        if (self.adaptive) |*adap| {
            adap.deinit();
        }
        if (self.dispatcher) |*disp| {
            disp.deinit();
        }
        if (self.gpu_backend) |backend| {
            backend_factory.destroyBackend(backend);
        }
    }

    /// Get coordinator performance statistics
    pub fn getPerformanceStats(self: *const ExecutionCoordinator) struct {
        total_gpu_ops: u64,
        total_simd_ops: u64,
        total_scalar_ops: u64,
        total_time_ns: u64,
        gpu_percentage: f64,
        simd_percentage: f64,
        adaptive_stats: ?struct {
            total_samples: usize,
            operations_tracked: usize,
            gpu_threshold_adjustments: usize,
            simd_threshold_adjustments: usize,
        },
    } {
        const total_ops = self.total_gpu_ops + self.total_simd_ops + self.total_scalar_ops;
        const total_f = if (total_ops > 0) @as(f64, @floatFromInt(total_ops)) else 1.0;

        return .{
            .total_gpu_ops = self.total_gpu_ops,
            .total_simd_ops = self.total_simd_ops,
            .total_scalar_ops = self.total_scalar_ops,
            .total_time_ns = self.total_time_ns,
            .gpu_percentage = @as(f64, @floatFromInt(self.total_gpu_ops)) / total_f * 100.0,
            .simd_percentage = @as(f64, @floatFromInt(self.total_simd_ops)) / total_f * 100.0,
            .adaptive_stats = if (self.adaptive) |*adap| adap.getStats() else null,
        };
    }

    /// Vector addition with automatic method selection
    pub fn vectorAdd(
        self: *ExecutionCoordinator,
        a: []const f32,
        b: []const f32,
        result: []f32,
    ) !ExecutionMethod {
        const method = self.selectMethod(a.len, .vector_add);
        return self.vectorAddWithMethod(a, b, result, method);
    }

    /// Vector addition with explicit method
    pub fn vectorAddWithMethod(
        self: *ExecutionCoordinator,
        a: []const f32,
        b: []const f32,
        result: []f32,
        method: ExecutionMethod,
    ) !ExecutionMethod {
        return switch (method) {
            .gpu => self.vectorAddGpu(a, b, result) catch |err| blk: {
                // Fallback on GPU failure
                if (self.config.log_fallbacks) {
                    std.log.warn("GPU vector add failed: {}, falling back to SIMD", .{err});
                }
                break :blk try self.vectorAddWithMethod(a, b, result, .simd);
            },
            .simd => blk: {
                if (self.simd_available and a.len >= self.config.simd_threshold_size) {
                    simd.vectorAdd(a, b, result);
                    break :blk .simd;
                } else {
                    // Fall through to scalar
                    if (self.config.log_fallbacks) {
                        std.log.info("SIMD unavailable or data too small (len={}), falling back to scalar", .{a.len});
                    }
                    break :blk try self.vectorAddWithMethod(a, b, result, .scalar);
                }
            },
            .scalar => blk: {
                for (a, b, 0..) |av, bv, i| {
                    result[i] = av + bv;
                }
                break :blk .scalar;
            },
            .failed => .failed,
        };
    }

    fn vectorAddGpu(
        self: *ExecutionCoordinator,
        a: []const f32,
        b: []const f32,
        result: []f32,
    ) !ExecutionMethod {
        if (self.gpu_backend == null) return error.GpuNotAvailable;
        if (self.dispatcher == null) return error.GpuNotAvailable;
        if (self.device == null) return error.GpuNotAvailable;

        var disp = &self.dispatcher.?;
        const device = &self.device.?;

        // Get or compile the vector_add kernel
        const kernel = disp.getBuiltinKernel(.vector_add) catch |err| {
            if (self.config.log_fallbacks) {
                std.log.warn("Failed to get vector_add kernel: {}", .{err});
            }
            return error.KernelCompilationFailed;
        };

        // Create unified buffers for the operation
        var buf_a = Buffer.init(self.allocator, a.len * @sizeOf(f32), device, .{
            .mode = .explicit,
            .element_type = .f32,
            .initial_data = std.mem.sliceAsBytes(a),
        }) catch return error.OutOfMemory;
        defer buf_a.deinit();

        var buf_b = Buffer.init(self.allocator, b.len * @sizeOf(f32), device, .{
            .mode = .explicit,
            .element_type = .f32,
            .initial_data = std.mem.sliceAsBytes(b),
        }) catch return error.OutOfMemory;
        defer buf_b.deinit();

        var buf_result = Buffer.init(self.allocator, result.len * @sizeOf(f32), device, .{
            .mode = .explicit,
            .element_type = .f32,
        }) catch return error.OutOfMemory;
        defer buf_result.deinit();

        // Configure kernel launch
        const config = LaunchConfig.for1D(a.len, kernel.workgroup_size[0]);

        // Execute the kernel
        var buffers = [_]*Buffer{ &buf_a, &buf_b, &buf_result };
        const args = KernelArgs{
            .buffers = &buffers,
        };

        _ = disp.execute(kernel, config, args) catch |err| {
            if (self.config.log_fallbacks) {
                std.log.warn("GPU vector_add execution failed: {}", .{err});
            }
            return error.ExecutionFailed;
        };

        // Copy result back to host
        buf_result.toHost() catch return error.TransferFailed;
        buf_result.read(f32, result) catch return error.TransferFailed;

        return .gpu;
    }

    /// Vector multiplication with automatic method selection
    pub fn vectorMul(
        self: *ExecutionCoordinator,
        a: []const f32,
        b: []const f32,
        result: []f32,
    ) !ExecutionMethod {
        const method = self.selectMethod(a.len, .vector_multiply);
        return self.vectorMulWithMethod(a, b, result, method);
    }

    /// Vector multiplication with explicit method
    pub fn vectorMulWithMethod(
        self: *ExecutionCoordinator,
        a: []const f32,
        b: []const f32,
        result: []f32,
        method: ExecutionMethod,
    ) !ExecutionMethod {
        return switch (method) {
            .gpu => self.vectorMulGpu(a, b, result) catch |err| blk: {
                if (self.config.log_fallbacks) {
                    std.log.warn("GPU vector mul failed: {}, falling back to SIMD", .{err});
                }
                break :blk try self.vectorMulWithMethod(a, b, result, .simd);
            },
            .simd => blk: {
                if (self.simd_available and a.len >= self.config.simd_threshold_size) {
                    simd.vectorMul(a, b, result);
                    break :blk .simd;
                } else {
                    if (self.config.log_fallbacks) {
                        std.log.info("SIMD unavailable or data too small (len={}), falling back to scalar", .{a.len});
                    }
                    break :blk try self.vectorMulWithMethod(a, b, result, .scalar);
                }
            },
            .scalar => blk: {
                for (a, b, 0..) |av, bv, i| {
                    result[i] = av * bv;
                }
                break :blk .scalar;
            },
            .failed => .failed,
        };
    }

    fn vectorMulGpu(
        self: *ExecutionCoordinator,
        a: []const f32,
        b: []const f32,
        result: []f32,
    ) !ExecutionMethod {
        if (self.gpu_backend == null) return error.GpuNotAvailable;
        if (self.dispatcher == null) return error.GpuNotAvailable;
        if (self.device == null) return error.GpuNotAvailable;

        var disp = &self.dispatcher.?;
        const device = &self.device.?;

        // Get or compile the vector_mul kernel
        const kernel = disp.getBuiltinKernel(.vector_mul) catch |err| {
            if (self.config.log_fallbacks) {
                std.log.warn("Failed to get vector_mul kernel: {}", .{err});
            }
            return error.KernelCompilationFailed;
        };

        // Create unified buffers for the operation
        var buf_a = Buffer.init(self.allocator, a.len * @sizeOf(f32), device, .{
            .mode = .explicit,
            .element_type = .f32,
            .initial_data = std.mem.sliceAsBytes(a),
        }) catch return error.OutOfMemory;
        defer buf_a.deinit();

        var buf_b = Buffer.init(self.allocator, b.len * @sizeOf(f32), device, .{
            .mode = .explicit,
            .element_type = .f32,
            .initial_data = std.mem.sliceAsBytes(b),
        }) catch return error.OutOfMemory;
        defer buf_b.deinit();

        var buf_result = Buffer.init(self.allocator, result.len * @sizeOf(f32), device, .{
            .mode = .explicit,
            .element_type = .f32,
        }) catch return error.OutOfMemory;
        defer buf_result.deinit();

        // Configure kernel launch
        const config = LaunchConfig.for1D(a.len, kernel.workgroup_size[0]);

        // Execute the kernel
        var buffers = [_]*Buffer{ &buf_a, &buf_b, &buf_result };
        const args = KernelArgs{
            .buffers = &buffers,
        };

        _ = disp.execute(kernel, config, args) catch |err| {
            if (self.config.log_fallbacks) {
                std.log.warn("GPU vector_mul execution failed: {}", .{err});
            }
            return error.ExecutionFailed;
        };

        // Copy result back to host
        buf_result.toHost() catch return error.TransferFailed;
        buf_result.read(f32, result) catch return error.TransferFailed;

        return .gpu;
    }

    /// Matrix multiplication with automatic method selection
    pub fn matmul(
        self: *ExecutionCoordinator,
        a: []const f32,
        b: []const f32,
        result: []f32,
        m: u32, // rows of A and C
        n: u32, // cols of B and C
        k: u32, // cols of A, rows of B
    ) !ExecutionMethod {
        const total_ops = @as(usize, m) * n * k;
        const method = self.selectMethod(total_ops, .matrix_multiply);
        return self.matmulWithMethod(a, b, result, m, n, k, method);
    }

    /// Matrix multiplication with explicit method
    pub fn matmulWithMethod(
        self: *ExecutionCoordinator,
        a: []const f32,
        b: []const f32,
        result: []f32,
        m: u32,
        n: u32,
        k: u32,
        method: ExecutionMethod,
    ) !ExecutionMethod {
        return switch (method) {
            .gpu => self.matmulGpu(a, b, result, m, n, k) catch |err| blk: {
                if (self.config.log_fallbacks) {
                    std.log.warn("GPU matmul failed: {}, falling back to scalar", .{err});
                }
                break :blk try self.matmulWithMethod(a, b, result, m, n, k, .scalar);
            },
            .simd, .scalar => blk: {
                // Naive scalar matrix multiplication (SIMD optimization could be added)
                const m_usize: usize = @intCast(m);
                const n_usize: usize = @intCast(n);
                const k_usize: usize = @intCast(k);

                for (0..m_usize) |i| {
                    for (0..n_usize) |j| {
                        var sum: f32 = 0;
                        for (0..k_usize) |kk| {
                            sum += a[i * k_usize + kk] * b[kk * n_usize + j];
                        }
                        result[i * n_usize + j] = sum;
                    }
                }
                break :blk .scalar;
            },
            .failed => .failed,
        };
    }

    fn matmulGpu(
        self: *ExecutionCoordinator,
        a: []const f32,
        b: []const f32,
        result: []f32,
        m: u32,
        n: u32,
        k: u32,
    ) !ExecutionMethod {
        if (self.gpu_backend == null) return error.GpuNotAvailable;
        if (self.dispatcher == null) return error.GpuNotAvailable;
        if (self.device == null) return error.GpuNotAvailable;

        var disp = &self.dispatcher.?;
        const device = &self.device.?;

        // Get or compile the matrix_multiply kernel
        const kernel = disp.getBuiltinKernel(.matrix_multiply) catch |err| {
            if (self.config.log_fallbacks) {
                std.log.warn("Failed to get matrix_multiply kernel: {}", .{err});
            }
            return error.KernelCompilationFailed;
        };

        // Create unified buffers for the operation
        const a_size = @as(usize, m) * k * @sizeOf(f32);
        const b_size = @as(usize, k) * n * @sizeOf(f32);
        const c_size = @as(usize, m) * n * @sizeOf(f32);

        var buf_a = Buffer.init(self.allocator, a_size, device, .{
            .mode = .explicit,
            .element_type = .f32,
            .initial_data = std.mem.sliceAsBytes(a[0..(@as(usize, m) * k)]),
        }) catch return error.OutOfMemory;
        defer buf_a.deinit();

        var buf_b = Buffer.init(self.allocator, b_size, device, .{
            .mode = .explicit,
            .element_type = .f32,
            .initial_data = std.mem.sliceAsBytes(b[0..(@as(usize, k) * n)]),
        }) catch return error.OutOfMemory;
        defer buf_b.deinit();

        var buf_result = Buffer.init(self.allocator, c_size, device, .{
            .mode = .explicit,
            .element_type = .f32,
        }) catch return error.OutOfMemory;
        defer buf_result.deinit();

        // Configure kernel launch for 2D execution (n columns, m rows)
        // The kernel uses global_size[0] = n, global_size[1] = m, global_size[2] = k for dimension info
        const config = LaunchConfig.for2D(n, m, kernel.workgroup_size[0], kernel.workgroup_size[1]);

        // Execute the kernel
        var buffers = [_]*Buffer{ &buf_a, &buf_b, &buf_result };
        const args = KernelArgs{
            .buffers = &buffers,
        };

        _ = disp.execute(kernel, config, args) catch |err| {
            if (self.config.log_fallbacks) {
                std.log.warn("GPU matrix_multiply execution failed: {}", .{err});
            }
            return error.ExecutionFailed;
        };

        // Copy result back to host
        buf_result.toHost() catch return error.TransferFailed;
        const result_slice = result[0..(@as(usize, m) * n)];
        buf_result.read(f32, result_slice) catch return error.TransferFailed;

        return .gpu;
    }

    /// Dot product with automatic method selection
    pub fn dotProduct(
        self: *ExecutionCoordinator,
        a: []const f32,
        b: []const f32,
    ) !struct { result: f32, method: ExecutionMethod } {
        const method = self.selectMethod(a.len, .dot_product);
        return self.dotProductWithMethod(a, b, method);
    }

    /// Dot product with explicit method
    pub fn dotProductWithMethod(
        self: *ExecutionCoordinator,
        a: []const f32,
        b: []const f32,
        method: ExecutionMethod,
    ) !struct { result: f32, method: ExecutionMethod } {
        return switch (method) {
            .gpu => self.dotProductGpu(a, b) catch |err| blk: {
                if (self.config.log_fallbacks) {
                    std.log.warn("GPU dot product failed: {}, falling back to SIMD", .{err});
                }
                break :blk try self.dotProductWithMethod(a, b, .simd);
            },
            .simd => blk: {
                if (self.simd_available and a.len >= self.config.simd_threshold_size) {
                    const result = simd.dotProduct(a, b);
                    break :blk .{ .result = result, .method = .simd };
                } else {
                    if (self.config.log_fallbacks) {
                        std.log.info("SIMD unavailable or data too small (len={}), falling back to scalar", .{a.len});
                    }
                    break :blk try self.dotProductWithMethod(a, b, .scalar);
                }
            },
            .scalar => blk: {
                var sum: f32 = 0;
                const len = @min(a.len, b.len);
                for (0..len) |i| {
                    sum += a[i] * b[i];
                }
                break :blk .{ .result = sum, .method = .scalar };
            },
            .failed => .{ .result = 0, .method = .failed },
        };
    }

    fn dotProductGpu(
        self: *ExecutionCoordinator,
        a: []const f32,
        b: []const f32,
    ) !struct { result: f32, method: ExecutionMethod } {
        if (self.gpu_backend == null) return error.GpuNotAvailable;
        if (self.dispatcher == null) return error.GpuNotAvailable;
        if (self.device == null) return error.GpuNotAvailable;

        var disp = &self.dispatcher.?;
        const device = &self.device.?;

        // Get or compile the dot_product kernel
        const kernel = disp.getBuiltinKernel(.dot_product) catch |err| {
            if (self.config.log_fallbacks) {
                std.log.warn("Failed to get dot_product kernel: {}", .{err});
            }
            return error.KernelCompilationFailed;
        };

        // Create unified buffers for the operation
        var buf_a = Buffer.init(self.allocator, a.len * @sizeOf(f32), device, .{
            .mode = .explicit,
            .element_type = .f32,
            .initial_data = std.mem.sliceAsBytes(a),
        }) catch return error.OutOfMemory;
        defer buf_a.deinit();

        var buf_b = Buffer.init(self.allocator, b.len * @sizeOf(f32), device, .{
            .mode = .explicit,
            .element_type = .f32,
            .initial_data = std.mem.sliceAsBytes(b),
        }) catch return error.OutOfMemory;
        defer buf_b.deinit();

        // Output buffer for the result (single f32)
        var buf_result = Buffer.init(self.allocator, @sizeOf(f32), device, .{
            .mode = .explicit,
            .element_type = .f32,
        }) catch return error.OutOfMemory;
        defer buf_result.deinit();

        // Zero the result buffer
        buf_result.fill(f32, 0.0) catch return error.OutOfMemory;

        // Configure kernel launch
        const config = LaunchConfig.for1D(a.len, kernel.workgroup_size[0]);

        // Execute the kernel
        var buffers = [_]*Buffer{ &buf_a, &buf_b, &buf_result };
        const args = KernelArgs{
            .buffers = &buffers,
        };

        _ = disp.execute(kernel, config, args) catch |err| {
            if (self.config.log_fallbacks) {
                std.log.warn("GPU dot_product execution failed: {}", .{err});
            }
            return error.ExecutionFailed;
        };

        // Copy result back to host
        buf_result.toHost() catch return error.TransferFailed;
        var result_arr: [1]f32 = undefined;
        buf_result.read(f32, &result_arr) catch return error.TransferFailed;

        return .{ .result = result_arr[0], .method = .gpu };
    }

    /// Reduce sum with automatic method selection
    pub fn reduceSum(
        self: *ExecutionCoordinator,
        input: []const f32,
    ) !struct { result: f32, method: ExecutionMethod } {
        const method = self.selectMethod(input.len, .vector_add);
        return self.reduceSumWithMethod(input, method);
    }

    /// Reduce sum with explicit method
    pub fn reduceSumWithMethod(
        self: *ExecutionCoordinator,
        input: []const f32,
        method: ExecutionMethod,
    ) !struct { result: f32, method: ExecutionMethod } {
        return switch (method) {
            .gpu => self.reduceSumGpu(input) catch |err| blk: {
                if (self.config.log_fallbacks) {
                    std.log.warn("GPU reduce sum failed: {}, falling back to SIMD", .{err});
                }
                break :blk try self.reduceSumWithMethod(input, .simd);
            },
            .simd => blk: {
                if (self.simd_available and input.len >= self.config.simd_threshold_size) {
                    const result = simd.reduceSum(input);
                    break :blk .{ .result = result, .method = .simd };
                } else {
                    if (self.config.log_fallbacks) {
                        std.log.info("SIMD unavailable or data too small (len={}), falling back to scalar", .{input.len});
                    }
                    break :blk try self.reduceSumWithMethod(input, .scalar);
                }
            },
            .scalar => blk: {
                var sum: f32 = 0;
                for (input) |v| {
                    sum += v;
                }
                break :blk .{ .result = sum, .method = .scalar };
            },
            .failed => .{ .result = 0, .method = .failed },
        };
    }

    fn reduceSumGpu(
        self: *ExecutionCoordinator,
        input: []const f32,
    ) !struct { result: f32, method: ExecutionMethod } {
        if (self.gpu_backend == null) return error.GpuNotAvailable;
        if (self.dispatcher == null) return error.GpuNotAvailable;
        if (self.device == null) return error.GpuNotAvailable;

        var disp = &self.dispatcher.?;
        const device = &self.device.?;

        // Get or compile the reduce_sum kernel
        const kernel = disp.getBuiltinKernel(.reduce_sum) catch |err| {
            if (self.config.log_fallbacks) {
                std.log.warn("Failed to get reduce_sum kernel: {}", .{err});
            }
            return error.KernelCompilationFailed;
        };

        // Create unified buffers for the operation
        var buf_input = Buffer.init(self.allocator, input.len * @sizeOf(f32), device, .{
            .mode = .explicit,
            .element_type = .f32,
            .initial_data = std.mem.sliceAsBytes(input),
        }) catch return error.OutOfMemory;
        defer buf_input.deinit();

        // Output buffer for the result (single f32)
        var buf_result = Buffer.init(self.allocator, @sizeOf(f32), device, .{
            .mode = .explicit,
            .element_type = .f32,
        }) catch return error.OutOfMemory;
        defer buf_result.deinit();

        // Zero the result buffer (for atomic add)
        buf_result.fill(f32, 0.0) catch return error.OutOfMemory;

        // Configure kernel launch
        const config = LaunchConfig.for1D(input.len, kernel.workgroup_size[0]);

        // Execute the kernel
        var buffers = [_]*Buffer{ &buf_input, &buf_result };
        const args = KernelArgs{
            .buffers = &buffers,
        };

        _ = disp.execute(kernel, config, args) catch |err| {
            if (self.config.log_fallbacks) {
                std.log.warn("GPU reduce_sum execution failed: {}", .{err});
            }
            return error.ExecutionFailed;
        };

        // Copy result back to host
        buf_result.toHost() catch return error.TransferFailed;
        var result_arr: [1]f32 = undefined;
        buf_result.read(f32, &result_arr) catch return error.TransferFailed;

        return .{ .result = result_arr[0], .method = .gpu };
    }

    /// Select best execution method for operation
    fn selectMethod(self: *ExecutionCoordinator, size: usize, op: OperationType) ExecutionMethod {
        // Use adaptive thresholds if available
        const gpu_threshold = if (self.adaptive) |*adap|
            adap.getGpuThreshold(op)
        else
            self.config.gpu_threshold_size;

        const simd_threshold = if (self.adaptive) |*adap|
            adap.getSimdThreshold(op)
        else
            self.config.simd_threshold_size;

        // Try methods in fallback chain order
        for (self.config.fallback_chain) |method| {
            if (self.canUseMethodWithThresholds(method, size, gpu_threshold, simd_threshold)) {
                return method;
            }
        }

        // Last resort: scalar
        return .scalar;
    }

    fn canUseMethod(self: *ExecutionCoordinator, method: ExecutionMethod, size: usize) bool {
        return self.canUseMethodWithThresholds(
            method,
            size,
            self.config.gpu_threshold_size,
            self.config.simd_threshold_size,
        );
    }

    fn canUseMethodWithThresholds(
        self: *ExecutionCoordinator,
        method: ExecutionMethod,
        size: usize,
        gpu_threshold: usize,
        simd_threshold: usize,
    ) bool {
        return switch (method) {
            .gpu => self.gpu_available and size >= gpu_threshold,
            .simd => self.simd_available and size >= simd_threshold,
            .scalar => true,
            .failed => false,
        };
    }

    /// Record operation result for adaptive learning
    fn recordResult(self: *ExecutionCoordinator, method: ExecutionMethod, size: usize, time_ns: u64, op: OperationType) void {
        // Update statistics
        switch (method) {
            .gpu => self.total_gpu_ops += 1,
            .simd => self.total_simd_ops += 1,
            .scalar => self.total_scalar_ops += 1,
            .failed => {},
        }
        self.total_time_ns += time_ns;

        // Record sample for adaptive learning
        if (self.adaptive) |*adap| {
            adap.recordSample(.{
                .size = size,
                .method = method,
                .time_ns = time_ns,
                .operation = op,
            }) catch |err| {
                std.log.debug("Failed to record adaptive learning sample: {t}", .{err});
            };
        }
    }
};

const OperationType = enum {
    vector_add,
    vector_multiply,
    matrix_multiply,
    dot_product,
};
