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
const backend_mod = @import("backend.zig");
const device_mod = @import("device.zig");
const interface = @import("interface.zig");
const dsl = @import("dsl/mod.zig");
const unified_buffer = @import("unified_buffer.zig");
const kernel_types = @import("kernel_types.zig");
const builtin_kernels = @import("builtin_kernels.zig");

pub const Backend = backend_mod.Backend;
pub const Device = device_mod.Device;
pub const Buffer = unified_buffer.Buffer;
pub const KernelIR = dsl.KernelIR;

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

/// Kernel dispatcher - manages kernel compilation, caching, and execution.
pub const KernelDispatcher = struct {
    allocator: std.mem.Allocator,
    backend: Backend,
    device: *const Device,

    /// Cache of compiled kernels by name.
    kernel_cache: std.StringHashMapUnmanaged(CompiledKernelHandle),
    /// Cache of builtin kernel IR.
    builtin_ir_cache: std.AutoHashMapUnmanaged(dsl.BuiltinKernel, *const KernelIR),

    /// Backend interface (if available).
    backend_interface: ?interface.Backend,

    /// Statistics.
    kernels_compiled: u64,
    kernels_executed: u64,
    cache_hits: u64,
    cache_misses: u64,

    const Self = @This();

    /// Initialize a new kernel dispatcher.
    pub fn init(
        allocator: std.mem.Allocator,
        backend: Backend,
        device: *const Device,
    ) !Self {
        return .{
            .allocator = allocator,
            .backend = backend,
            .device = device,
            .kernel_cache = .empty,
            .builtin_ir_cache = .empty,
            .backend_interface = null, // Will be set by backend factory
            .kernels_compiled = 0,
            .kernels_executed = 0,
            .cache_hits = 0,
            .cache_misses = 0,
        };
    }

    /// Deinitialize and release resources.
    pub fn deinit(self: *Self) void {
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
        const ir = builtin_kernels.buildKernelIR(self.allocator, kernel_type) catch |err| {
            std.log.err("Failed to build kernel IR for {s}: {}", .{ name, err });
            return DispatchError.KernelCompilationFailed;
        };

        // Compile the IR to the target backend
        return self.compileKernel(ir);
    }

    /// Compile a custom kernel from IR.
    pub fn compileKernel(self: *Self, ir: *const KernelIR) DispatchError!CompiledKernelHandle {
        // Generate backend-specific code
        var generated = dsl.compile(self.allocator, ir, self.backend, .{}) catch |err| {
            std.log.err("Failed to compile kernel {s} for backend {s}: {}", .{
                ir.name,
                @tagName(self.backend),
                err,
            });
            return DispatchError.KernelCompilationFailed;
        };

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
                buf.toDevice() catch |err| {
                    std.log.warn("Failed to sync buffer to device: {}", .{err});
                    return DispatchError.BufferNotReady;
                };
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

        // Try GPU execution first
        if (kernel.handle != null and self.backend_interface != null) {
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
            .backend = self.backend,
            .device_id = self.device.id,
            .gpu_executed = gpu_executed,
        };
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
        bi.synchronize() catch {};
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
        };
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

// ============================================================================
// Batched Dispatcher for Small Operations
// ============================================================================

/// Batched operation for deferred execution.
pub const BatchedOp = struct {
    kernel: CompiledKernelHandle,
    config: LaunchConfig,
    buffers: [8]*Buffer, // Fixed-size to avoid allocation
    buffer_count: u8,
};

/// Batched dispatcher that collects small operations and executes them together.
/// This reduces dispatch overhead for many small kernel launches.
pub const BatchedDispatcher = struct {
    allocator: std.mem.Allocator,
    inner: *KernelDispatcher,
    pending_ops: std.ArrayListUnmanaged(BatchedOp),
    batch_threshold: usize,
    auto_flush_size: usize,

    const Self = @This();

    /// Minimum elements before considering an op "small" enough to batch
    pub const SMALL_OP_THRESHOLD: usize = 4096;

    /// Initialize batched dispatcher wrapping a KernelDispatcher.
    pub fn init(allocator: std.mem.Allocator, dispatcher: *KernelDispatcher) Self {
        return .{
            .allocator = allocator,
            .inner = dispatcher,
            .pending_ops = .{},
            .batch_threshold = SMALL_OP_THRESHOLD,
            .auto_flush_size = 32, // Auto-flush after 32 pending ops
        };
    }

    /// Deinitialize and flush any pending operations.
    pub fn deinit(self: *Self) void {
        // Flush remaining ops (ignore errors during cleanup)
        self.flush() catch {};
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
        };

        for (args.buffers, 0..) |buf, i| {
            op.buffers[i] = buf;
        }

        self.pending_ops.append(self.allocator, op) catch return DispatchError.OutOfMemory;

        // Auto-flush if we have enough pending ops
        if (self.pending_ops.items.len >= self.auto_flush_size) {
            try self.flush();
        }
    }

    /// Execute all pending operations in a batch.
    pub fn flush(self: *Self) DispatchError!void {
        if (self.pending_ops.items.len == 0) return;

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
        for (self.pending_ops.items) |*op| {
            const args = KernelArgs{
                .buffers = op.buffers[0..op.buffer_count],
            };
            _ = self.inner.execute(op.kernel, op.config, args) catch |err| {
                std.log.warn("Batched op failed: {}", .{err});
                // Continue with remaining ops
            };
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
