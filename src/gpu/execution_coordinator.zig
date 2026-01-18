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
//! 2. Use external locking (e.g., `std.Thread.Mutex`) around all coordinator calls
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
const backend_factory = @import("backend_factory.zig");
const simd = @import("../shared/simd.zig");
const dispatcher_mod = @import("dispatcher.zig");
const device_mod = @import("device.zig");
const unified_buffer = @import("unified_buffer.zig");
const dsl = @import("dsl/mod.zig");

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
};

pub const ExecutionCoordinator = struct {
    allocator: std.mem.Allocator,
    config: CoordinatorConfig,
    gpu_backend: ?*backend_factory.BackendInstance = null,
    gpu_available: bool = false,
    simd_available: bool = false,
    dispatcher: ?KernelDispatcher = null,
    device: ?Device = null,

    pub fn init(allocator: std.mem.Allocator, config: CoordinatorConfig) !ExecutionCoordinator {
        var coord = ExecutionCoordinator{
            .allocator = allocator,
            .config = config,
            .simd_available = simd.hasSimdSupport(),
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
        if (self.dispatcher) |*disp| {
            disp.deinit();
        }
        if (self.gpu_backend) |backend| {
            backend_factory.destroyBackend(backend);
        }
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
        _ = op; // Reserved for operation-specific heuristics

        // Try methods in fallback chain order
        for (self.config.fallback_chain) |method| {
            if (self.canUseMethod(method, size)) {
                return method;
            }
        }

        // Last resort: scalar
        return .scalar;
    }

    fn canUseMethod(self: *ExecutionCoordinator, method: ExecutionMethod, size: usize) bool {
        return switch (method) {
            .gpu => self.gpu_available and size >= self.config.gpu_threshold_size,
            .simd => self.simd_available and size >= self.config.simd_threshold_size,
            .scalar => true,
            .failed => false,
        };
    }
};

const OperationType = enum {
    vector_add,
    vector_multiply,
    matrix_multiply,
    dot_product,
};
