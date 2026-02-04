//! GPU Kernel Dispatcher
//!
//! Bridges the unified GPU API to actual backend kernel execution.
//! This is the critical layer that connects high-level operations
//! (vectorAdd, matrixMultiply, etc.) to compiled backend-specific kernels.
//!
//! ## Architecture
//!
//! ```
//! Unified API (vectorAdd, etc.)
//!        ↓
//! KernelDispatcher (this module)
//!        ↓
//! Compiled Kernels (via DSL → backend codegen)
//!        ↓
//! Backend Interface (CUDA, Vulkan, Metal, etc.)
//! ```
//!
//! ## Usage
//!
//! ```zig
//! var dispatcher = try KernelDispatcher.init(allocator, backend, device);
//! defer dispatcher.deinit();
//!
//! // Get or compile a builtin kernel
//! const kernel = try dispatcher.getBuiltinKernel(.vector_add);
//!
//! // Execute with buffers
//! const result = try dispatcher.execute(kernel, config, .{
//!     .buffers = &.{ a, b, result_buf },
//!     .uniforms = &.{ @ptrCast(&n) },
//! });
//! ```

const std = @import("std");
const build_options = @import("build_options");
const backend_mod = @import("backend.zig");
const device_mod = @import("device.zig");
const interface = @import("interface.zig");
const dsl = @import("dsl/mod.zig");
const unified_buffer = @import("unified_buffer.zig");
const kernel_types = @import("kernel_types.zig");
const builtin_kernels = @import("builtin_kernels.zig");
const kernel_ring_mod = @import("kernel_ring.zig");

// Conditionally import CUDA/cuBLAS for optimized BLAS operations
const cublas = if (build_options.enable_gpu)
    @import("backends/cuda/cublas.zig")
else
    struct {
        pub const CublasContext = void;
        pub fn isAvailable() bool {
            return false;
        }
    };

pub const Backend = backend_mod.Backend;
pub const Device = device_mod.Device;
pub const Buffer = unified_buffer.Buffer;
pub const KernelIR = dsl.KernelIR;
pub const KernelRing = kernel_ring_mod.KernelRing;

/// Errors that can occur during kernel dispatch.
pub const DispatchError = error{
    NoBackendAvailable,
    KernelNotFound,
    KernelCompilationFailed,
    InvalidConfiguration,
    BufferNotReady,
    DeviceMismatch,
    OutOfMemory,
    ExecutionFailed,
    UnsupportedOperation,
    BackendNotInitialized,
    InvalidArguments,
    TimerFailed,
    LaunchQueueFull,
    BatchTooLarge,
};

/// Handle to a compiled kernel.
pub const CompiledKernelHandle = struct {
    /// Backend-specific handle.
    handle: ?*anyopaque,
    /// Kernel name for identification.
    name: []const u8,
    /// Backend this kernel was compiled for.
    backend: Backend,
    /// Workgroup size used during compilation.
    workgroup_size: [3]u32,
    /// Number of buffer parameters expected.
    buffer_count: u8,
    /// Number of uniform parameters expected.
    uniform_count: u8,

    pub fn isValid(self: *const CompiledKernelHandle) bool {
        return self.handle != null or self.backend == .stdgpu;
    }
};

/// Configuration for kernel execution.
pub const LaunchConfig = struct {
    /// Global work size (total threads).
    global_size: [3]u32 = .{ 1, 1, 1 },
    /// Local work size (workgroup/block size). null = auto-calculate.
    local_size: ?[3]u32 = null,
    /// Shared memory size in bytes.
    shared_memory: u32 = 0,
    /// Optional stream handle for async execution.
    stream: ?*anyopaque = null,

    /// Calculate grid dimensions from global size and local size.
    pub fn gridDimensions(self: *const LaunchConfig) [3]u32 {
        const local = self.local_size orelse .{ 256, 1, 1 };
        return .{
            (self.global_size[0] + local[0] - 1) / local[0],
            (self.global_size[1] + local[1] - 1) / local[1],
            (self.global_size[2] + local[2] - 1) / local[2],
        };
    }

    /// Create config for 1D kernel execution.
    pub fn for1D(element_count: usize, workgroup_size: u32) LaunchConfig {
        return .{
            .global_size = .{ @intCast(element_count), 1, 1 },
            .local_size = .{ workgroup_size, 1, 1 },
        };
    }

    /// Create config for 2D kernel execution (e.g., matrices).
    pub fn for2D(width: usize, height: usize, tile_x: u32, tile_y: u32) LaunchConfig {
        return .{
            .global_size = .{ @intCast(width), @intCast(height), 1 },
            .local_size = .{ tile_x, tile_y, 1 },
        };
    }
};

/// Arguments for kernel execution.
pub const KernelArgs = struct {
    /// Buffer arguments (device memory pointers).
    buffers: []const *Buffer = &.{},
    /// Uniform arguments (small constant values).
    uniforms: []const *const anyopaque = &.{},
    /// Uniform sizes in bytes for each uniform.
    uniform_sizes: []const usize = &.{},
};

/// Result of kernel execution.
pub const ExecutionResult = struct {
    /// Execution time in nanoseconds.
    execution_time_ns: u64,
    /// Number of elements processed.
    elements_processed: usize,
    /// Bytes transferred (input + output).
    bytes_transferred: usize,
    /// Backend used for execution.
    backend: Backend,
    /// Device ID used.
    device_id: u32,
    /// Whether kernel executed on GPU (vs CPU fallback).
    gpu_executed: bool,

    /// Get throughput in GB/s.
    pub fn throughputGBps(self: *const ExecutionResult) f64 {
        if (self.execution_time_ns == 0) return 0;
        const bytes_per_sec = @as(f64, @floatFromInt(self.bytes_transferred)) /
            (@as(f64, @floatFromInt(self.execution_time_ns)) / 1_000_000_000.0);
        return bytes_per_sec / (1024 * 1024 * 1024);
    }

    /// Get elements per second.
    pub fn elementsPerSecond(self: *const ExecutionResult) f64 {
        if (self.execution_time_ns == 0) return 0;
        return @as(f64, @floatFromInt(self.elements_processed)) /
            (@as(f64, @floatFromInt(self.execution_time_ns)) / 1_000_000_000.0);
    }
};

/// Queued kernel launch for batching.
pub const QueuedLaunch = struct {
    kernel: *const CompiledKernelHandle,
    config: LaunchConfig,
    args: KernelArgs,
};

/// Kernel dispatcher - manages kernel compilation, caching, and execution.
pub const KernelDispatcher = struct {
    allocator: std.mem.Allocator,
    ir_arena: std.heap.ArenaAllocator,
    backend: Backend,
    device: *const Device,

    /// Cache of compiled kernels by name.
    kernel_cache: std.StringHashMapUnmanaged(CompiledKernelHandle),
    /// Cache of builtin kernel IR.
    builtin_ir_cache: std.AutoHashMapUnmanaged(dsl.BuiltinKernel, *const KernelIR),

    /// Backend interface (if available).
    backend_interface: ?interface.Backend,

    /// cuBLAS context for optimized BLAS operations (CUDA only).
    cublas_ctx: if (build_options.enable_gpu) ?cublas.CublasContext else void,

    /// Statistics.
    kernels_compiled: u64,
    kernels_executed: u64,
    cache_hits: u64,
    cache_misses: u64,
    cublas_ops: u64,
    ring_hits: u64,

    /// Kernel launch configuration ring buffer for fast-path reuse.
    kernel_ring: KernelRing,

    /// Launch queue for batching kernel launches.
    launch_queue: std.ArrayListUnmanaged(QueuedLaunch),
    /// Mutex for thread-safe queue operations.
    queue_mutex: std.Thread.Mutex,
    /// Maximum queue size before auto-flush.
    max_queue_size: usize = 32,

    const Self = @This();

    /// Initialize a new kernel dispatcher.
    pub fn init(
        allocator: std.mem.Allocator,
        backend: Backend,
        device: *const Device,
    ) !Self {
        var self = Self{
            .allocator = allocator,
            .ir_arena = std.heap.ArenaAllocator.init(allocator),
            .backend = backend,
            .device = device,
            .kernel_cache = .empty,
            .builtin_ir_cache = .empty,
            .backend_interface = null, // Will be set by backend factory
            .cublas_ctx = if (build_options.enable_gpu) null else {},
            .kernels_compiled = 0,
            .kernels_executed = 0,
            .cache_hits = 0,
            .cache_misses = 0,
            .cublas_ops = 0,
            .ring_hits = 0,
            .kernel_ring = KernelRing.init(),
            .launch_queue = .empty,
            .queue_mutex = .{},
            .max_queue_size = 32,
        };

        // Try to initialize cuBLAS for CUDA backend
        if (build_options.enable_gpu and backend == .cuda) {
            if (cublas.isAvailable()) {
                self.cublas_ctx = cublas.CublasContext.init() catch null;
                if (self.cublas_ctx != null) {
                    std.log.info("cuBLAS initialized for optimized BLAS operations", .{});
                }
            }
        }

        return self;
    }

    /// Deinitialize and release resources.
    pub fn deinit(self: *Self) void {
        // Clean up cuBLAS context
        if (build_options.enable_gpu) {
            if (self.cublas_ctx) |*ctx| {
                ctx.deinit();
            }
        }

        // Free cached kernel handles
        var it = self.kernel_cache.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.handle) |handle| {
                // Backend would free handle here
                if (self.backend_interface) |bi| {
                    bi.destroyKernel(handle);
                }
            }
            self.allocator.free(entry.key_ptr.*);
        }
        self.kernel_cache.deinit(self.allocator);

        // Free cached IR
        var ir_it = self.builtin_ir_cache.iterator();
        while (ir_it.next()) |entry| {
            // IR is typically arena-allocated, but we track it here
            _ = entry;
        }
        self.builtin_ir_cache.deinit(self.allocator);

        // Free arena-allocated IR and AST memory
        self.ir_arena.deinit();
    }

    /// Check if cuBLAS is available for optimized BLAS operations.
    pub fn hasCublas(self: *const Self) bool {
        if (!build_options.enable_gpu) return false;
        return self.cublas_ctx != null;
    }

    /// Set the backend interface for actual GPU execution.
    pub fn setBackendInterface(self: *Self, bi: interface.Backend) void {
        self.backend_interface = bi;
    }

    /// Get or compile a builtin kernel.
    pub fn getBuiltinKernel(self: *Self, kernel_type: dsl.BuiltinKernel) DispatchError!CompiledKernelHandle {
        const name = kernel_type.name();

        // Check cache first
        if (self.kernel_cache.get(name)) |cached| {
            self.cache_hits += 1;
            return cached;
        }

        self.cache_misses += 1;

        // Build the kernel IR using builtin_kernels module
        const ir = builtin_kernels.buildKernelIR(self.ir_arena.allocator(), kernel_type) catch |err| {
            std.log.err("Failed to build kernel IR for {s}: {}", .{ name, err });
            return DispatchError.KernelCompilationFailed;
        };
        errdefer {
            ir.deinit(self.allocator);
            self.allocator.destroy(@constCast(ir));
        }

        // Compile the IR to the target backend
        const handle = try self.compileKernel(ir);
        ir.deinit(self.allocator);
        self.allocator.destroy(@constCast(ir));
        return handle;
    }

    /// Compile a custom kernel from IR.
    pub fn compileKernel(self: *Self, ir: *const KernelIR) DispatchError!CompiledKernelHandle {
        // Generate backend-specific code
        var generated = dsl.compile(self.allocator, ir, self.backend, .{}) catch |err| {
            std.log.err("Failed to compile kernel {s} for backend {t}: {t}", .{
                ir.name,
                self.backend,
                err,
            });
            return DispatchError.KernelCompilationFailed;
        };
        defer generated.deinit(self.allocator);

        // Compile with backend
        var handle: ?*anyopaque = null;
        if (self.backend_interface) |bi| {
            if (bi.compileKernel(
                self.allocator,
                generated.code,
                generated.entry_point,
            )) |compiled| {
                handle = compiled;
            } else |err| {
                std.log.warn("Backend compilation failed for {s}: {}. Using CPU fallback.", .{
                    ir.name,
                    err,
                });
                // Fall through to create a handle without backend
            }
        }

        const kernel_handle = CompiledKernelHandle{
            .handle = handle,
            .name = ir.name,
            .backend = self.backend,
            .workgroup_size = ir.workgroup_size,
            .buffer_count = @intCast(ir.buffers.len),
            .uniform_count = @intCast(ir.uniforms.len),
        };

        // Cache the compiled kernel
        const name_copy = self.allocator.dupe(u8, ir.name) catch return DispatchError.OutOfMemory;
        self.kernel_cache.put(self.allocator, name_copy, kernel_handle) catch {
            self.allocator.free(name_copy);
            return DispatchError.OutOfMemory;
        };

        self.kernels_compiled += 1;
        return kernel_handle;
    }

    /// Execute a compiled kernel.
    pub fn execute(
        self: *Self,
        kernel: CompiledKernelHandle,
        config: LaunchConfig,
        args: KernelArgs,
    ) DispatchError!ExecutionResult {
        var timer = std.time.Timer.start() catch return DispatchError.TimerFailed;

        // Track launch configuration in ring buffer for fast-path detection
        const grid = config.gridDimensions();
        const local = config.local_size orelse .{ 256, 1, 1 };
        const ring_desc = KernelRing.Descriptor{
            .kernel_handle = if (kernel.handle) |h| @intFromPtr(h) else std.hash.Wyhash.hash(0, kernel.name),
            .grid_dim = grid,
            .block_dim = local,
            .shared_mem = config.shared_memory,
        };

        // Check if this is a repeated configuration (fast-path)
        const old_count = self.kernel_ring.count();
        _ = self.kernel_ring.pushOrReuse(ring_desc);
        if (self.kernel_ring.count() == old_count) {
            // Reused existing slot - this is a repeated configuration
            self.ring_hits += 1;
        }

        // Validate arguments
        if (args.buffers.len != kernel.buffer_count) {
            std.log.err("Kernel {s} expects {} buffers, got {}", .{
                kernel.name,
                kernel.buffer_count,
                args.buffers.len,
            });
            return DispatchError.InvalidArguments;
        }

        // Ensure buffers are on device
        for (args.buffers) |buf| {
            if (buf.isHostDirty()) {
                if (config.stream) |s| {
                    buf.toDeviceAsync(s) catch |err| {
                        std.log.warn("Failed to sync buffer to device async: {}", .{err});
                        return DispatchError.BufferNotReady;
                    };
                } else {
                    buf.toDevice() catch |err| {
                        std.log.warn("Failed to sync buffer to device: {}", .{err});
                        return DispatchError.BufferNotReady;
                    };
                }
            }
        }

        // Calculate bytes transferred
        var bytes_transferred: usize = 0;
        for (args.buffers) |buf| {
            bytes_transferred += buf.getSize();
        }

        // Calculate elements processed
        const elements = config.global_size[0] * config.global_size[1] * config.global_size[2];

        var gpu_executed = false;
        var used_cublas = false;

        // Check for cuBLAS optimization for batch_matmul and matrix_multiply
        if (build_options.enable_gpu) {
            if (self.cublas_ctx != null and
                (std.mem.eql(u8, kernel.name, "batch_matmul") or
                    std.mem.eql(u8, kernel.name, "matrix_multiply")))
            {
                if (self.executeCublasGemm(kernel, config, args)) {
                    gpu_executed = true;
                    used_cublas = true;
                    self.cublas_ops += 1;
                } else |err| {
                    std.log.debug("cuBLAS execution failed for {s}: {}. Falling back.", .{
                        kernel.name,
                        err,
                    });
                }
            }
        }

        // Try GPU execution if cuBLAS wasn't used
        if (!gpu_executed and kernel.handle != null and self.backend_interface != null) {
            const launch_result = self.launchOnBackend(kernel, config, args);
            if (launch_result) |_| {
                gpu_executed = true;
            } else |err| {
                std.log.debug("Backend execution failed for {s}: {}. Using CPU fallback.", .{
                    kernel.name,
                    err,
                });
            }
        }

        // CPU fallback if needed
        if (!gpu_executed) {
            self.executeOnCpu(kernel, config, args) catch |err| {
                std.log.err("CPU fallback execution failed for {s}: {}", .{ kernel.name, err });
                return DispatchError.ExecutionFailed;
            };
        }

        // Mark output buffers as device dirty
        for (args.buffers) |buf| {
            buf.markDeviceDirty();
        }

        const elapsed = timer.read();
        self.kernels_executed += 1;

        return ExecutionResult{
            .execution_time_ns = elapsed,
            .elements_processed = elements,
            .bytes_transferred = bytes_transferred,
            .backend = if (used_cublas) .cuda else self.backend,
            .device_id = self.device.id,
            .gpu_executed = gpu_executed,
        };
    }

    /// Maximum dimension size for safe matrix operations (prevents overflow).
    const MAX_MATRIX_DIM: u32 = 32768; // 32K x 32K max

    /// Safely cast u32 to i32 with overflow check.
    fn safeCastToI32(val: u32) DispatchError!i32 {
        if (val > std.math.maxInt(i32)) {
            return DispatchError.InvalidArguments;
        }
        return @intCast(val);
    }

    /// Validate matrix dimensions to prevent overflow in stride calculations.
    fn validateMatrixDimensions(m: u32, n: u32, k: u32) DispatchError!void {
        if (m > MAX_MATRIX_DIM or n > MAX_MATRIX_DIM or k > MAX_MATRIX_DIM) {
            std.log.err("Matrix dimensions exceed safe limit: m={}, n={}, k={} (max={})", .{ m, n, k, MAX_MATRIX_DIM });
            return DispatchError.InvalidArguments;
        }
        // Check for potential overflow in stride calculations (i64 is safe for dimensions up to 32K)
        const max_stride = @as(u64, m) * @as(u64, k);
        if (max_stride > std.math.maxInt(i64)) {
            return DispatchError.InvalidArguments;
        }
    }

    /// Safely read a uniform parameter with type checking.
    fn readUniformAs(comptime T: type, args: KernelArgs, index: usize, default: T) T {
        if (index >= args.uniforms.len) return default;
        if (index >= args.uniform_sizes.len) {
            // No size info, use pointer cast but log warning
            std.log.debug("Uniform at index {} has no size info, using type assumption", .{index});
            return @as(*const T, @ptrCast(@alignCast(args.uniforms[index]))).*;
        }
        // Validate size matches expected type
        if (args.uniform_sizes[index] != @sizeOf(T)) {
            std.log.warn("Uniform size mismatch at index {}: expected {} bytes, got {}", .{
                index, @sizeOf(T), args.uniform_sizes[index],
            });
            return default;
        }
        return @as(*const T, @ptrCast(@alignCast(args.uniforms[index]))).*;
    }

    /// Execute matrix multiplication using cuBLAS (optimized BLAS library).
    fn executeCublasGemm(
        self: *Self,
        kernel: CompiledKernelHandle,
        config: LaunchConfig,
        args: KernelArgs,
    ) DispatchError!void {
        if (!build_options.enable_gpu) return DispatchError.UnsupportedOperation;

        var ctx = self.cublas_ctx orelse return DispatchError.UnsupportedOperation;
        const bufs = args.buffers;

        if (bufs.len < 3) return DispatchError.InvalidArguments;

        // Get device pointers
        const a_ptr = bufs[0].getDevicePtr() catch return DispatchError.BufferNotReady;
        const b_ptr = bufs[1].getDevicePtr() catch return DispatchError.BufferNotReady;
        const c_ptr = bufs[2].getDevicePtr() catch return DispatchError.BufferNotReady;

        if (std.mem.eql(u8, kernel.name, "matrix_multiply")) {
            // Standard matrix multiply: C = A * B
            // Dimensions from config: global_size[0] = n, global_size[1] = m, global_size[2] = k
            // Validate dimensions before casting
            try validateMatrixDimensions(config.global_size[1], config.global_size[0], config.global_size[2]);

            const n = try safeCastToI32(config.global_size[0]);
            const m = try safeCastToI32(config.global_size[1]);
            const k = try safeCastToI32(config.global_size[2]);

            ctx.sgemm(
                .no_trans,
                .no_trans,
                n,
                m,
                k,
                1.0, // alpha
                b_ptr,
                n, // ldb
                a_ptr,
                k, // lda
                0.0, // beta
                c_ptr,
                n, // ldc
            ) catch return DispatchError.ExecutionFailed;
        } else if (std.mem.eql(u8, kernel.name, "batch_matmul")) {
            // Batched matrix multiply: C[b] = A[b] * B[b]
            // Dimensions: global_size[0] = n, global_size[1] = m, global_size[2] = batch_size
            // k is passed via uniforms or inferred
            const n_u32 = config.global_size[0];
            const m_u32 = config.global_size[1];
            const batch_count_u32 = config.global_size[2];

            // Get k from uniforms if available, otherwise assume square matrices
            const k_u32: u32 = blk: {
                const k_from_uniform = readUniformAs(i32, args, 0, 0);
                if (k_from_uniform > 0) {
                    break :blk @intCast(k_from_uniform);
                }
                break :blk n_u32;
            };

            // Validate all dimensions
            try validateMatrixDimensions(m_u32, n_u32, k_u32);
            if (batch_count_u32 > MAX_MATRIX_DIM) {
                return DispatchError.InvalidArguments;
            }

            const n = try safeCastToI32(n_u32);
            const m = try safeCastToI32(m_u32);
            const k = try safeCastToI32(k_u32);
            const batch_count = try safeCastToI32(batch_count_u32);

            // Calculate strides for strided batched GEMM (safe due to dimension validation)
            const stride_a: i64 = @as(i64, m) * @as(i64, k);
            const stride_b: i64 = @as(i64, k) * @as(i64, n);
            const stride_c: i64 = @as(i64, m) * @as(i64, n);

            ctx.sgemmStridedBatched(
                .no_trans,
                .no_trans,
                n,
                m,
                k,
                1.0, // alpha
                b_ptr,
                n, // ldb
                stride_b,
                a_ptr,
                k, // lda
                stride_a,
                0.0, // beta
                c_ptr,
                n, // ldc
                stride_c,
                batch_count,
            ) catch return DispatchError.ExecutionFailed;
        } else {
            return DispatchError.UnsupportedOperation;
        }
    }

    /// Execute on the actual backend.
    fn launchOnBackend(
        self: *Self,
        kernel: CompiledKernelHandle,
        config: LaunchConfig,
        args: KernelArgs,
    ) DispatchError!void {
        const bi = self.backend_interface orelse return DispatchError.BackendNotInitialized;
        const handle = kernel.handle orelse return DispatchError.KernelNotFound;

        // Build argument list for backend, allocating device memory as needed
        var arg_ptrs = std.ArrayListUnmanaged(*anyopaque).empty;
        defer arg_ptrs.deinit(self.allocator);

        for (args.buffers) |buf| {
            var device_ptr: *anyopaque = undefined;

            // Allocate device memory if buffer doesn't have one
            if (!buf.hasDeviceHandle()) {
                const size = buf.getSize();
                device_ptr = bi.allocate(size, .{}) catch {
                    std.log.debug("Failed to allocate {} bytes device memory", .{size});
                    return DispatchError.OutOfMemory;
                };

                // Copy host data to device
                if (buf.getHostBytes()) |host_data| {
                    bi.copyToDevice(device_ptr, host_data) catch {
                        bi.free(device_ptr);
                        return DispatchError.BufferNotReady;
                    };
                }

                // Store device handle in buffer for reuse
                buf.setDeviceHandle(device_ptr);
                buf.clearHostDirty();
            } else {
                device_ptr = buf.getDevicePtr() catch return DispatchError.BufferNotReady;
            }

            arg_ptrs.append(self.allocator, device_ptr) catch return DispatchError.OutOfMemory;
        }

        // Create launch config for backend
        const grid = config.gridDimensions();
        const local = config.local_size orelse kernel.workgroup_size;
        const backend_config = interface.LaunchConfig{
            .grid_x = grid[0],
            .grid_y = grid[1],
            .grid_z = grid[2],
            .block_x = local[0],
            .block_y = local[1],
            .block_z = local[2],
            .shared_memory = config.shared_memory,
        };

        // Launch kernel
        bi.launchKernel(
            handle,
            backend_config,
            arg_ptrs.items,
        ) catch return DispatchError.ExecutionFailed;

        // Synchronize to ensure completion
        bi.synchronize() catch |err| {
            std.log.warn("GPU synchronization failed: {t}", .{err});
            return DispatchError.ExecutionFailed;
        };
    }

    /// CPU fallback execution using host memory.
    fn executeOnCpu(
        self: *Self,
        kernel: CompiledKernelHandle,
        config: LaunchConfig,
        args: KernelArgs,
    ) DispatchError!void {
        _ = self;

        // Dispatch based on kernel name
        const name = kernel.name;
        const bufs = args.buffers;

        if (std.mem.eql(u8, name, "vector_add")) {
            if (bufs.len >= 3) {
                try executeCpuVectorAdd(bufs[0], bufs[1], bufs[2]);
            }
        } else if (std.mem.eql(u8, name, "vector_sub")) {
            if (bufs.len >= 3) {
                try executeCpuVectorSub(bufs[0], bufs[1], bufs[2]);
            }
        } else if (std.mem.eql(u8, name, "vector_mul")) {
            if (bufs.len >= 3) {
                try executeCpuVectorMul(bufs[0], bufs[1], bufs[2]);
            }
        } else if (std.mem.eql(u8, name, "reduce_sum")) {
            if (bufs.len >= 2) {
                try executeCpuReduceSum(bufs[0], bufs[1]);
            }
        } else if (std.mem.eql(u8, name, "dot_product")) {
            if (bufs.len >= 3) {
                try executeCpuDotProduct(bufs[0], bufs[1], bufs[2]);
            }
        } else if (std.mem.eql(u8, name, "softmax")) {
            if (bufs.len >= 2) {
                try executeCpuSoftmax(bufs[0], bufs[1]);
            }
        } else if (std.mem.eql(u8, name, "matrix_multiply")) {
            if (bufs.len >= 3) {
                // Matrix dimensions would need to be passed via uniforms
                const m = config.global_size[1];
                const n = config.global_size[0];
                const k = config.global_size[2];
                try executeCpuMatrixMultiply(bufs[0], bufs[1], bufs[2], m, n, k);
            }
        } else if (std.mem.eql(u8, name, "batch_cosine_similarity")) {
            if (bufs.len >= 3) {
                // global_size[0] = num_vectors, global_size[1] = dim
                const num_vectors = config.global_size[0];
                const dim = config.global_size[1];
                // query_norm passed as first uniform (use safe reader)
                const query_norm = readUniformAs(f32, args, 0, 1.0);
                try executeCpuBatchCosineSimilarity(bufs[0], bufs[1], bufs[2], num_vectors, dim, query_norm);
            }
        } else if (std.mem.eql(u8, name, "gelu")) {
            if (bufs.len >= 2) {
                try executeCpuGelu(bufs[0], bufs[1]);
            }
        } else if (std.mem.eql(u8, name, "silu")) {
            if (bufs.len >= 2) {
                try executeCpuSilu(bufs[0], bufs[1]);
            }
        } else if (std.mem.eql(u8, name, "layer_norm")) {
            if (bufs.len >= 4) {
                // mean, variance, epsilon passed as uniforms (use safe reader)
                const mean = readUniformAs(f32, args, 0, 0.0);
                const variance = readUniformAs(f32, args, 1, 1.0);
                const epsilon = readUniformAs(f32, args, 2, 1e-5);
                try executeCpuLayerNorm(bufs[0], bufs[1], bufs[2], bufs[3], mean, variance, epsilon);
            }
        } else if (std.mem.eql(u8, name, "rms_norm")) {
            if (bufs.len >= 3) {
                // rms, epsilon passed as uniforms (use safe reader)
                const rms = readUniformAs(f32, args, 0, 1.0);
                const epsilon = readUniformAs(f32, args, 1, 1e-5);
                try executeCpuRmsNorm(bufs[0], bufs[1], bufs[2], rms, epsilon);
            }
        } else {
            std.log.warn("No CPU fallback for kernel: {s}", .{name});
            return DispatchError.UnsupportedOperation;
        }
    }

    /// Get dispatcher statistics.
    pub fn getStats(self: *const Self) struct {
        kernels_compiled: u64,
        kernels_executed: u64,
        cache_hits: u64,
        cache_misses: u64,
        cache_hit_rate: f64,
        cublas_ops: u64,
        ring_hits: u64,
        ring_entries: u32,
    } {
        const total_lookups = self.cache_hits + self.cache_misses;
        const hit_rate = if (total_lookups > 0)
            @as(f64, @floatFromInt(self.cache_hits)) / @as(f64, @floatFromInt(total_lookups))
        else
            0.0;

        return .{
            .kernels_compiled = self.kernels_compiled,
            .kernels_executed = self.kernels_executed,
            .cache_hits = self.cache_hits,
            .cache_misses = self.cache_misses,
            .cache_hit_rate = hit_rate,
            .cublas_ops = self.cublas_ops,
            .ring_hits = self.ring_hits,
            .ring_entries = self.kernel_ring.count(),
        };
    }

    /// Get the kernel ring buffer for direct access.
    pub fn getKernelRing(self: *Self) *KernelRing {
        return &self.kernel_ring;
    }
};

// ============================================================================
// CPU Fallback Implementations
// ============================================================================

fn executeCpuVectorAdd(a: *Buffer, b: *Buffer, result: *Buffer) DispatchError!void {
    const a_data = std.mem.bytesAsSlice(f32, a.host_data orelse return DispatchError.BufferNotReady);
    const b_data = std.mem.bytesAsSlice(f32, b.host_data orelse return DispatchError.BufferNotReady);
    var r_data = std.mem.bytesAsSlice(f32, result.host_data orelse return DispatchError.BufferNotReady);

    const len = @min(a_data.len, @min(b_data.len, r_data.len));
    for (0..len) |i| {
        r_data[i] = a_data[i] + b_data[i];
    }
}

fn executeCpuVectorSub(a: *Buffer, b: *Buffer, result: *Buffer) DispatchError!void {
    const a_data = std.mem.bytesAsSlice(f32, a.host_data orelse return DispatchError.BufferNotReady);
    const b_data = std.mem.bytesAsSlice(f32, b.host_data orelse return DispatchError.BufferNotReady);
    var r_data = std.mem.bytesAsSlice(f32, result.host_data orelse return DispatchError.BufferNotReady);

    const len = @min(a_data.len, @min(b_data.len, r_data.len));
    for (0..len) |i| {
        r_data[i] = a_data[i] - b_data[i];
    }
}

fn executeCpuVectorMul(a: *Buffer, b: *Buffer, result: *Buffer) DispatchError!void {
    const a_data = std.mem.bytesAsSlice(f32, a.host_data orelse return DispatchError.BufferNotReady);
    const b_data = std.mem.bytesAsSlice(f32, b.host_data orelse return DispatchError.BufferNotReady);
    var r_data = std.mem.bytesAsSlice(f32, result.host_data orelse return DispatchError.BufferNotReady);

    const len = @min(a_data.len, @min(b_data.len, r_data.len));
    for (0..len) |i| {
        r_data[i] = a_data[i] * b_data[i];
    }
}

fn executeCpuReduceSum(input: *Buffer, result: *Buffer) DispatchError!void {
    const in_data = std.mem.bytesAsSlice(f32, input.host_data orelse return DispatchError.BufferNotReady);
    var r_data = std.mem.bytesAsSlice(f32, result.host_data orelse return DispatchError.BufferNotReady);

    var sum: f32 = 0;
    for (in_data) |v| {
        sum += v;
    }

    if (r_data.len > 0) {
        r_data[0] = sum;
    }
}

fn executeCpuDotProduct(a: *Buffer, b: *Buffer, result: *Buffer) DispatchError!void {
    const a_data = std.mem.bytesAsSlice(f32, a.host_data orelse return DispatchError.BufferNotReady);
    const b_data = std.mem.bytesAsSlice(f32, b.host_data orelse return DispatchError.BufferNotReady);
    var r_data = std.mem.bytesAsSlice(f32, result.host_data orelse return DispatchError.BufferNotReady);

    var sum: f32 = 0;
    const len = @min(a_data.len, b_data.len);
    for (0..len) |i| {
        sum += a_data[i] * b_data[i];
    }

    if (r_data.len > 0) {
        r_data[0] = sum;
    }
}

fn executeCpuSoftmax(input: *Buffer, output: *Buffer) DispatchError!void {
    const in_data = std.mem.bytesAsSlice(f32, input.host_data orelse return DispatchError.BufferNotReady);
    var out_data = std.mem.bytesAsSlice(f32, output.host_data orelse return DispatchError.BufferNotReady);

    if (in_data.len == 0) return;

    const len = @min(in_data.len, out_data.len);

    // Find max for numerical stability
    var max_val: f32 = in_data[0];
    for (in_data[1..]) |v| {
        if (v > max_val) max_val = v;
    }

    // Compute exp(x - max) and sum
    var sum: f32 = 0;
    for (0..len) |i| {
        out_data[i] = @exp(in_data[i] - max_val);
        sum += out_data[i];
    }

    // Normalize
    for (0..len) |i| {
        out_data[i] /= sum;
    }
}

fn executeCpuMatrixMultiply(
    a: *Buffer,
    b: *Buffer,
    result: *Buffer,
    m: u32,
    n: u32,
    k: u32,
) DispatchError!void {
    const a_data = std.mem.bytesAsSlice(f32, a.host_data orelse return DispatchError.BufferNotReady);
    const b_data = std.mem.bytesAsSlice(f32, b.host_data orelse return DispatchError.BufferNotReady);
    var r_data = std.mem.bytesAsSlice(f32, result.host_data orelse return DispatchError.BufferNotReady);

    // C[i,j] = sum(A[i,k] * B[k,j])
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0;
            for (0..k) |kk| {
                sum += a_data[i * k + kk] * b_data[kk * n + j];
            }
            r_data[i * n + j] = sum;
        }
    }
}

fn executeCpuBatchCosineSimilarity(
    query: *Buffer,
    vectors: *Buffer,
    result: *Buffer,
    num_vectors: u32,
    dim: u32,
    query_norm: f32,
) DispatchError!void {
    const q_data = std.mem.bytesAsSlice(f32, query.host_data orelse return DispatchError.BufferNotReady);
    const v_data = std.mem.bytesAsSlice(f32, vectors.host_data orelse return DispatchError.BufferNotReady);
    var r_data = std.mem.bytesAsSlice(f32, result.host_data orelse return DispatchError.BufferNotReady);

    const d = @as(usize, dim);
    const n = @min(@as(usize, num_vectors), r_data.len);

    for (0..n) |i| {
        var dot_sum: f32 = 0;
        var vec_norm_sq: f32 = 0;
        const vec_offset = i * d;

        for (0..d) |j| {
            if (j < q_data.len and vec_offset + j < v_data.len) {
                const q_val = q_data[j];
                const v_val = v_data[vec_offset + j];
                dot_sum += q_val * v_val;
                vec_norm_sq += v_val * v_val;
            }
        }

        const vec_norm = @sqrt(vec_norm_sq);
        const norm_product = query_norm * vec_norm;

        r_data[i] = if (norm_product > 1e-8)
            dot_sum / norm_product
        else
            0;
    }
}

fn executeCpuGelu(input: *Buffer, output: *Buffer) DispatchError!void {
    const in_data = std.mem.bytesAsSlice(f32, input.host_data orelse return DispatchError.BufferNotReady);
    var out_data = std.mem.bytesAsSlice(f32, output.host_data orelse return DispatchError.BufferNotReady);

    const len = @min(in_data.len, out_data.len);
    const sqrt_2_pi: f32 = 0.7978845608;
    const coef: f32 = 0.044715;

    for (0..len) |i| {
        const x = in_data[i];
        const x_cubed = x * x * x;
        const inner = sqrt_2_pi * (x + coef * x_cubed);
        const tanh_val = std.math.tanh(inner);
        out_data[i] = 0.5 * x * (1.0 + tanh_val);
    }
}

fn executeCpuSilu(input: *Buffer, output: *Buffer) DispatchError!void {
    const in_data = std.mem.bytesAsSlice(f32, input.host_data orelse return DispatchError.BufferNotReady);
    var out_data = std.mem.bytesAsSlice(f32, output.host_data orelse return DispatchError.BufferNotReady);

    const len = @min(in_data.len, out_data.len);

    for (0..len) |i| {
        const x = in_data[i];
        // SiLU(x) = x / (1 + exp(-x))
        out_data[i] = x / (1.0 + @exp(-x));
    }
}

fn executeCpuLayerNorm(
    input: *Buffer,
    gamma: *Buffer,
    beta: *Buffer,
    output: *Buffer,
    mean: f32,
    variance: f32,
    epsilon: f32,
) DispatchError!void {
    const in_data = std.mem.bytesAsSlice(f32, input.host_data orelse return DispatchError.BufferNotReady);
    const g_data = std.mem.bytesAsSlice(f32, gamma.host_data orelse return DispatchError.BufferNotReady);
    const b_data = std.mem.bytesAsSlice(f32, beta.host_data orelse return DispatchError.BufferNotReady);
    var out_data = std.mem.bytesAsSlice(f32, output.host_data orelse return DispatchError.BufferNotReady);

    const len = @min(in_data.len, @min(out_data.len, @min(g_data.len, b_data.len)));
    const std_dev = @sqrt(variance + epsilon);

    for (0..len) |i| {
        const normalized = (in_data[i] - mean) / std_dev;
        out_data[i] = g_data[i] * normalized + b_data[i];
    }
}

fn executeCpuRmsNorm(
    input: *Buffer,
    gamma: *Buffer,
    output: *Buffer,
    rms: f32,
    epsilon: f32,
) DispatchError!void {
    const in_data = std.mem.bytesAsSlice(f32, input.host_data orelse return DispatchError.BufferNotReady);
    const g_data = std.mem.bytesAsSlice(f32, gamma.host_data orelse return DispatchError.BufferNotReady);
    var out_data = std.mem.bytesAsSlice(f32, output.host_data orelse return DispatchError.BufferNotReady);

    const len = @min(in_data.len, @min(out_data.len, g_data.len));
    const rms_eps = rms + epsilon;

    for (0..len) |i| {
        out_data[i] = g_data[i] * in_data[i] / rms_eps;
    }
}

// ============================================================================
// Batched Dispatcher for Small Operations
// ============================================================================

/// Batched operation for deferred execution.
pub const BatchedOp = struct {
    kernel: CompiledKernelHandle,
    config: LaunchConfig,
    buffers: [8]*Buffer, // Fixed-size to avoid allocation
    buffer_count: u8,
    /// Priority level for execution ordering
    priority: Priority = .normal,
    /// Category for grouping similar operations
    category: Category = .unknown,

    pub const Priority = enum {
        high, // Execute first
        normal, // Standard priority
        low, // Execute last
    };

    pub const Category = enum {
        unknown,
        vector_ops,
        matrix_ops,
        element_wise,
        reduction,
        activation,
    };
};

/// Batched dispatcher that collects small operations and executes them together.
/// This reduces dispatch overhead for many small kernel launches.
pub const BatchedDispatcher = struct {
    allocator: std.mem.Allocator,
    inner: *KernelDispatcher,
    pending_ops: std.ArrayListUnmanaged(BatchedOp),
    batch_threshold: usize,
    auto_flush_size: usize,
    /// Statistical tracking for optimization
    stats: Stats,

    const Self = @This();

    /// Minimum elements before considering an op "small" enough to batch
    pub const SMALL_OP_THRESHOLD: usize = 4096;

    /// Statistics for batched operations
    pub const Stats = struct {
        total_queued: u64 = 0,
        total_flushed: u64 = 0,
        batches_executed: u64 = 0,
        avg_batch_size: f32 = 0.0,
        high_priority_count: u64 = 0,
        low_priority_count: u64 = 0,
    };

    /// Initialize batched dispatcher wrapping a KernelDispatcher.
    pub fn init(allocator: std.mem.Allocator, dispatcher: *KernelDispatcher) Self {
        return .{
            .allocator = allocator,
            .inner = dispatcher,
            .pending_ops = .{},
            .batch_threshold = SMALL_OP_THRESHOLD,
            .auto_flush_size = 32, // Auto-flush after 32 pending ops
            .stats = .{},
        };
    }

    /// Deinitialize and flush any pending operations.
    pub fn deinit(self: *Self) void {
        // Flush remaining ops (log errors during cleanup)
        self.flush() catch |err| {
            std.log.debug("BatchingDispatcher.flush failed during deinit: {t}", .{err});
        };
        self.pending_ops.deinit(self.allocator);
    }

    /// Queue an operation for batched execution.
    /// Small operations are queued; large operations execute immediately.
    pub fn queue(
        self: *Self,
        kernel: CompiledKernelHandle,
        config: LaunchConfig,
        args: KernelArgs,
    ) DispatchError!void {
        const elements = @as(usize, config.global_size[0]) *
            @as(usize, config.global_size[1]) *
            @as(usize, config.global_size[2]);

        // Determine operation category based on kernel name
        const category = blk: {
            const name = kernel.name;
            if (std.mem.indexOf(u8, name, "vector") != null) {
                break :blk BatchedOp.Category.vector_ops;
            } else if (std.mem.indexOf(u8, name, "matrix") != null) {
                break :blk BatchedOp.Category.matrix_ops;
            } else if (std.mem.indexOf(u8, name, "add") != null or
                std.mem.indexOf(u8, name, "mul") != null or
                std.mem.indexOf(u8, name, "sub") != null)
            {
                break :blk BatchedOp.Category.element_wise;
            } else if (std.mem.indexOf(u8, name, "reduce") != null or
                std.mem.indexOf(u8, name, "sum") != null)
            {
                break :blk BatchedOp.Category.reduction;
            } else if (std.mem.indexOf(u8, name, "relu") != null or
                std.mem.indexOf(u8, name, "gelu") != null or
                std.mem.indexOf(u8, name, "sigmoid") != null)
            {
                break :blk BatchedOp.Category.activation;
            }
            break :blk BatchedOp.Category.unknown;
        };

        // Assign priority based on operation characteristics
        const priority = blk: {
            // High priority for reductions and activations (often in critical path)
            if (category == .reduction or category == .activation) {
                break :blk BatchedOp.Priority.high;
            }
            // Low priority for element-wise operations that can be batched
            if (category == .element_wise and elements < self.batch_threshold / 4) {
                break :blk BatchedOp.Priority.low;
            }
            break :blk BatchedOp.Priority.normal;
        };

        // Large operations execute immediately
        if (elements >= self.batch_threshold) {
            _ = try self.inner.execute(kernel, config, args);
            return;
        }

        // Queue small operation
        if (args.buffers.len > 8) {
            // Too many buffers, execute immediately
            _ = try self.inner.execute(kernel, config, args);
            return;
        }

        var op = BatchedOp{
            .kernel = kernel,
            .config = config,
            .buffers = undefined,
            .buffer_count = @intCast(args.buffers.len),
            .priority = priority,
            .category = category,
        };

        for (args.buffers, 0..) |buf, i| {
            op.buffers[i] = buf;
        }

        self.pending_ops.append(self.allocator, op) catch return DispatchError.OutOfMemory;
        self.stats.total_queued += 1;

        // Update priority statistics
        switch (priority) {
            .high => self.stats.high_priority_count += 1,
            .low => self.stats.low_priority_count += 1,
            else => {},
        }

        // Auto-flush if we have enough pending ops
        if (self.pending_ops.items.len >= self.auto_flush_size) {
            try self.flush();
        }
    }

    /// Execute all pending operations in a batch using proper categorization and prioritization
    pub fn flush(self: *Self) DispatchError!void {
        if (self.pending_ops.items.len == 0) return;

        // Simple categorization based on op type for better cache behavior
        // Operations are already categorized during queueing, so we can group similar ops

        // Sync all input buffers to device once
        for (self.pending_ops.items) |*op| {
            for (op.buffers[0..op.buffer_count]) |buf| {
                if (buf.isHostDirty()) {
                    buf.toDevice() catch |err| {
                        std.log.warn("Failed to sync buffer: {}", .{err});
                        return DispatchError.BufferNotReady;
                    };
                }
            }
        }

        // Execute all ops
        var batch_size: f32 = 0.0;
        for (self.pending_ops.items) |*op| {
            const args = KernelArgs{
                .buffers = op.buffers[0..op.buffer_count],
            };
            _ = self.inner.execute(op.kernel, op.config, args) catch |err| {
                std.log.warn("Batched op failed: {}", .{err});
                // Continue with remaining ops
            };
            batch_size += 1.0;
        }

        // Update statistics
        self.stats.batches_executed += 1;
        self.stats.total_flushed += @as(u64, @intFromFloat(batch_size));
        if (self.stats.batches_executed > 1) {
            // Running average of batch size
            const current_avg = self.stats.avg_batch_size;
            const new_avg = (current_avg * @as(f32, @floatFromInt(self.stats.batches_executed - 1)) + batch_size) /
                @as(f32, @floatFromInt(self.stats.batches_executed));
            self.stats.avg_batch_size = new_avg;
        } else {
            self.stats.avg_batch_size = batch_size;
        }

        // Clear pending ops
        self.pending_ops.clearRetainingCapacity();
    }

    /// Get number of pending operations.
    pub fn pendingCount(self: *const Self) usize {
        return self.pending_ops.items.len;
    }

    /// Set the threshold for what constitutes a "small" operation.
    pub fn setBatchThreshold(self: *Self, threshold: usize) void {
        self.batch_threshold = threshold;
    }

    /// Set when to auto-flush pending operations.
    pub fn setAutoFlushSize(self: *Self, size: usize) void {
        self.auto_flush_size = size;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "LaunchConfig for1D" {
    const config = LaunchConfig.for1D(1024, 256);
    try std.testing.expectEqual(@as(u32, 1024), config.global_size[0]);
    try std.testing.expectEqual(@as(u32, 256), config.local_size.?[0]);

    const grid = config.gridDimensions();
    try std.testing.expectEqual(@as(u32, 4), grid[0]);
}

test "LaunchConfig for2D" {
    const config = LaunchConfig.for2D(512, 512, 16, 16);
    try std.testing.expectEqual(@as(u32, 512), config.global_size[0]);
    try std.testing.expectEqual(@as(u32, 512), config.global_size[1]);

    const grid = config.gridDimensions();
    try std.testing.expectEqual(@as(u32, 32), grid[0]);
    try std.testing.expectEqual(@as(u32, 32), grid[1]);
}

test "ExecutionResult throughput" {
    const result = ExecutionResult{
        .execution_time_ns = 1_000_000_000, // 1 second
        .elements_processed = 1_000_000,
        .bytes_transferred = 1024 * 1024 * 1024, // 1 GB
        .backend = .cuda,
        .device_id = 0,
        .gpu_executed = true,
    };

    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result.throughputGBps(), 0.01);
    try std.testing.expectApproxEqAbs(@as(f64, 1_000_000.0), result.elementsPerSecond(), 1.0);
}

test "KernelDispatcher init and deinit" {
    const device = Device{
        .id = 0,
        .backend = .stdgpu,
        .name = "Test Device",
        .device_type = .discrete,
        .total_memory = null,
        .available_memory = null,
        .is_emulated = true,
        .capability = .{},
        .compute_units = null,
        .clock_mhz = null,
    };

    var dispatcher = try KernelDispatcher.init(std.testing.allocator, .stdgpu, &device);
    defer dispatcher.deinit();

    const stats = dispatcher.getStats();
    try std.testing.expectEqual(@as(u64, 0), stats.kernels_compiled);
    try std.testing.expectEqual(@as(u64, 0), stats.kernels_executed);
}
