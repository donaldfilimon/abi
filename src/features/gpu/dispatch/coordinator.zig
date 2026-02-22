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
const time = @import("../../../services/shared/time.zig");
const sync = @import("../../../services/shared/sync.zig");
const build_options = @import("build_options");
const backend_mod = @import("../backend.zig");
const device_mod = @import("../device.zig");
const interface = @import("../interface.zig");
const backend_shared = @import("../backends/shared.zig");
const dsl = @import("../dsl/mod.zig");
const unified_buffer = @import("../unified_buffer.zig");
const kernel_types = @import("../kernel_types.zig");
const builtin_kernels = @import("../builtin_kernels.zig");
const kernel_ring_mod = @import("../kernel_ring.zig");
const policy = @import("../policy/mod.zig");
const std_gpu_integration = @import("../backends/std_gpu_integration.zig");
const StdGpuKernelRegistry = std_gpu_integration.StdGpuKernelRegistry;

const Mutex = sync.Mutex;

/// Only compiles init() when Ctx is not void (CUDA compiled). Avoids type-checking void.init().
fn initCublasOptional(comptime Ctx: type) ?Ctx {
    if (Ctx == void) return null;
    return cublas.CublasContext.init() catch null;
}

// Re-export extracted submodules for build discovery
pub const dispatch_types = @import("types.zig");
pub const batched_dispatch = @import("batch.zig");

// Re-export types from dispatch_types for backward compatibility
pub const DispatchError = dispatch_types.DispatchError;
pub const CompiledKernelHandle = dispatch_types.CompiledKernelHandle;
pub const LaunchConfig = dispatch_types.LaunchConfig;
pub const KernelArgs = dispatch_types.KernelArgs;
pub const ExecutionResult = dispatch_types.ExecutionResult;
pub const QueuedLaunch = dispatch_types.QueuedLaunch;

// Re-export batched dispatcher types
pub const BatchedOp = batched_dispatch.BatchedOp;
pub const BatchedDispatcher = batched_dispatch.BatchedDispatcher;

// Conditionally import CUDA/cuBLAS for optimized BLAS operations
const cublas = if (build_options.enable_gpu and build_options.gpu_cuda and backend_shared.dynlibSupported)
    @import("../backends/cuda/cublas.zig")
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
    cublas_ctx: if (build_options.gpu_cuda) ?cublas.CublasContext else void,

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
    queue_mutex: Mutex,
    /// Maximum queue size before auto-flush.
    max_queue_size: usize = 32,

    const Self = @This();

    /// Initialize a new kernel dispatcher.
    pub fn init(
        allocator: std.mem.Allocator,
        backend: Backend,
        device: *const Device,
    ) !Self {
        const hints = policy.optimizationHintsForPlatform(policy.classifyBuiltin());
        var self = Self{
            .allocator = allocator,
            .ir_arena = std.heap.ArenaAllocator.init(allocator),
            .backend = backend,
            .device = device,
            .kernel_cache = .empty,
            .builtin_ir_cache = .empty,
            .backend_interface = null, // Will be set by backend factory
            .cublas_ctx = if (build_options.gpu_cuda) null else {},
            .kernels_compiled = 0,
            .kernels_executed = 0,
            .cache_hits = 0,
            .cache_misses = 0,
            .cublas_ops = 0,
            .ring_hits = 0,
            .kernel_ring = KernelRing.init(),
            .launch_queue = .empty,
            .queue_mutex = .{},
            .max_queue_size = @as(usize, hints.default_queue_depth) * 8,
        };

        // Try to initialize cuBLAS for CUDA backend (only when field is ?CublasContext, not void)
        if (comptime @TypeOf(@as(Self, undefined).cublas_ctx) != void) {
            if (backend == .cuda and cublas.isAvailable() and cublas.CublasContext != void) {
                self.cublas_ctx = initCublasOptional(cublas.CublasContext);
                if (self.cublas_ctx != null) {
                    std.log.info("cuBLAS initialized for optimized BLAS operations", .{});
                }
            }
        }

        return self;
    }

    /// Deinitialize and release resources.
    pub fn deinit(self: *Self) void {
        // Clean up cuBLAS context (only when field is optional, not void)
        if (comptime @TypeOf(@as(Self, undefined).cublas_ctx) != void) {
            if (self.cublas_ctx) |*ctx| {
                if (@TypeOf(ctx.*) != void) ctx.deinit();
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

        // Free launch queue
        self.launch_queue.deinit(self.allocator);

        // Free cached IR (IR is arena-allocated, freed when ir_arena is deinitialized)
        self.builtin_ir_cache.deinit(self.allocator);

        // Free arena-allocated IR and AST memory
        self.ir_arena.deinit();
    }

    /// Check if cuBLAS is available for optimized BLAS operations.
    pub fn hasCublas(self: *const Self) bool {
        if (!build_options.gpu_cuda) return false;
        return self.cublas_ctx != null;
    }

    /// Set the backend interface for actual GPU execution.
    pub fn setBackendInterface(self: *Self, bi: interface.Backend) void {
        self.backend_interface = bi;
    }

    /// Get or compile a builtin kernel.
    pub fn getBuiltinKernel(self: *Self, kernel_type: dsl.BuiltinKernel) DispatchError!CompiledKernelHandle {
        const kernel_name = kernel_type.name();

        // Check cache first
        if (self.kernel_cache.get(kernel_name)) |cached| {
            self.cache_hits += 1;
            return cached;
        }

        self.cache_misses += 1;

        // For CPU-based backends (stdgpu/simulated), if the native Zig kernel
        // registry supports this kernel type, return a lightweight handle that
        // skips the DSL→IR→codegen pipeline entirely. The actual execution will
        // be handled by StdGpuKernelRegistry in execute().
        if (self.backend == .stdgpu or self.backend == .simulated) {
            // Check if registry would handle this kernel (dry-run with no buffers)
            const native_handle = CompiledKernelHandle{
                .handle = null,
                .name = kernel_name,
                .backend = self.backend,
                .workgroup_size = .{ 256, 1, 1 },
                .buffer_count = kernel_type.minBufferCount(),
                .uniform_count = 0,
            };

            const name_copy = self.allocator.dupe(u8, kernel_name) catch
                return DispatchError.OutOfMemory;
            self.kernel_cache.put(self.allocator, name_copy, native_handle) catch {
                self.allocator.free(name_copy);
                return DispatchError.OutOfMemory;
            };

            self.kernels_compiled += 1;
            return native_handle;
        }

        // Build the kernel IR using builtin_kernels module
        const ir = builtin_kernels.buildKernelIR(self.ir_arena.allocator(), kernel_type) catch |err| {
            std.log.err("Failed to build kernel IR for {s}: {}", .{ kernel_name, err });
            return DispatchError.KernelCompilationFailed;
        };

        // Compile the IR to the target backend
        // IR uses `ir_arena`; reclaim happens when dispatcher deinitializes the arena.
        return try self.compileKernel(ir);
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
                std.log.debug("Backend compilation failed for {s}: {}. Using CPU fallback.", .{
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
        const name_copy = self.allocator.dupe(u8, ir.name) catch {
            // Clean up backend handle on allocation failure
            if (handle) |h| {
                if (self.backend_interface) |bi_cleanup| {
                    bi_cleanup.destroyKernel(h);
                }
            }
            return DispatchError.OutOfMemory;
        };
        self.kernel_cache.put(self.allocator, name_copy, kernel_handle) catch {
            self.allocator.free(name_copy);
            // Clean up backend handle on cache insertion failure
            if (handle) |h| {
                if (self.backend_interface) |bi_cleanup| {
                    bi_cleanup.destroyKernel(h);
                }
            }
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
        const timer_result = time.Timer.start();
        var timer = timer_result catch |err| {
            std.log.debug("Timer unavailable for kernel execution: {t}", .{err});
            // Timer unavailable - execute without timing
            return self.executeWithoutTiming(kernel, config, args);
        };

        // Track launch configuration in ring buffer for fast-path detection
        const grid = config.gridDimensions();
        const local = config.effectiveLocalSize();
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
            std.log.debug("Kernel {s} expects {} buffers, got {}", .{
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
                        std.log.debug("Failed to sync buffer to device async: {}", .{err});
                        return DispatchError.BufferNotReady;
                    };
                } else {
                    buf.toDevice() catch |err| {
                        std.log.debug("Failed to sync buffer to device: {}", .{err});
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

        // Check for cuBLAS optimization for batch_matmul and matrix_multiply (only when field is ?CublasContext)
        if (@TypeOf(@as(Self, undefined).cublas_ctx) != void and cublas.CublasContext != void) {
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

        // Try native Zig kernel execution for CPU-based backends (stdgpu/simulated).
        // Native Zig kernels avoid the DSL→GLSL/SPIR-V codegen path and run the
        // kernel logic directly as optimized Zig code on the CPU.
        if (!gpu_executed and (self.backend == .stdgpu or self.backend == .simulated)) {
            if (dsl.BuiltinKernel.fromName(kernel.name)) |builtin_kernel| {
                if (StdGpuKernelRegistry.executeBuiltin(
                    builtin_kernel,
                    args.buffers,
                    config.global_size,
                    args.uniforms,
                    args.uniform_sizes,
                )) {
                    gpu_executed = true;
                    std.log.debug("Native Zig kernel executed for {s} on {s} backend", .{
                        kernel.name,
                        self.backend.name(),
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
                std.log.debug("CPU fallback execution failed for {s}: {}", .{ kernel.name, err });
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

    /// Execute a kernel without timing (fallback for platforms without timer support).
    fn executeWithoutTiming(
        self: *Self,
        kernel: CompiledKernelHandle,
        config: LaunchConfig,
        args: KernelArgs,
    ) DispatchError!ExecutionResult {
        // Validate arguments
        if (args.buffers.len != kernel.buffer_count) {
            return DispatchError.InvalidArguments;
        }

        // Ensure buffers are on device
        for (args.buffers) |buf| {
            if (buf.isHostDirty()) {
                buf.toDevice() catch return DispatchError.BufferNotReady;
            }
        }

        // Calculate metrics
        var bytes_transferred: usize = 0;
        for (args.buffers) |buf| {
            bytes_transferred += buf.getSize();
        }
        const elements = config.global_size[0] * config.global_size[1] * config.global_size[2];

        // Try native Zig kernel for CPU backends
        var native_executed = false;
        if (self.backend == .stdgpu or self.backend == .simulated) {
            if (dsl.BuiltinKernel.fromName(kernel.name)) |builtin_kernel| {
                if (StdGpuKernelRegistry.executeBuiltin(
                    builtin_kernel,
                    args.buffers,
                    config.global_size,
                    args.uniforms,
                    args.uniform_sizes,
                )) {
                    native_executed = true;
                }
            }
        }

        // Fall back to generic CPU execution if native kernel wasn't available
        if (!native_executed) {
            self.executeOnCpu(kernel, config, args) catch |err| {
                std.log.debug("CPU fallback execution failed for {s}: {}", .{ kernel.name, err });
                return DispatchError.ExecutionFailed;
            };
        }

        // Mark output buffers as device dirty
        for (args.buffers) |buf| {
            buf.markDeviceDirty();
        }

        self.kernels_executed += 1;

        return ExecutionResult{
            .execution_time_ns = 0, // No timing available
            .elements_processed = elements,
            .bytes_transferred = bytes_transferred,
            .backend = self.backend,
            .device_id = self.device.id,
            .gpu_executed = false, // CPU fallback due to no timer
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
    /// Only compiled when CublasContext is not void (CUDA backend built).
    fn executeCublasGemm(
        self: *Self,
        kernel: CompiledKernelHandle,
        config: LaunchConfig,
        args: KernelArgs,
    ) DispatchError!void {
        if (comptime cublas.CublasContext == void) return DispatchError.UnsupportedOperation;
        if (!build_options.enable_gpu) return DispatchError.UnsupportedOperation;

        if (comptime cublas.CublasContext != void) {
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
            std.log.debug("No CPU fallback for kernel: {s}", .{name});
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

// Test discovery for extracted submodules
test {
    _ = @import("types.zig");
    _ = @import("batch.zig");
    _ = @import("../dispatcher_test.zig");
}
