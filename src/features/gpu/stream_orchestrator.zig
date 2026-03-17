//! Stream/queue management and execution orchestration.
//!
//! Extracted from `unified.zig` to separate execution and synchronization concerns.
//! Handles high-level GPU operations (vectorAdd, matrixMultiply, etc.),
//! custom kernel compilation/launch, stream creation, and synchronization.

const std = @import("std");
const time = @import("../../services/shared/mod.zig").time;
const backend_mod = @import("backend.zig");
const device_mod = @import("device.zig");
const stream_mod = @import("stream.zig");
const buffer_mod = @import("unified_buffer.zig");
const dsl = @import("dsl/mod.zig");
const metrics_mod = @import("metrics.zig");
const dispatcher_mod = @import("dispatch/coordinator.zig");
const adaptive_tiling_mod = @import("adaptive_tiling.zig");

pub const Backend = backend_mod.Backend;
pub const Device = device_mod.Device;
pub const Stream = stream_mod.Stream;
pub const StreamOptions = stream_mod.StreamOptions;
pub const StreamPriority = stream_mod.StreamPriority;
pub const Event = stream_mod.Event;
pub const EventOptions = stream_mod.EventOptions;
pub const StreamManager = stream_mod.StreamManager;
pub const Buffer = buffer_mod.Buffer;
pub const KernelBuilder = dsl.KernelBuilder;
pub const KernelIR = dsl.KernelIR;
pub const PortableKernelSource = dsl.PortableKernelSource;
pub const MetricsCollector = metrics_mod.MetricsCollector;
pub const MetricsSummary = metrics_mod.Summary;
pub const KernelMetrics = metrics_mod.KernelMetrics;
pub const KernelDispatcher = dispatcher_mod.KernelDispatcher;
pub const DispatchError = dispatcher_mod.DispatchError;
pub const LaunchConfig = dispatcher_mod.LaunchConfig;
pub const ExecutionResult = dispatcher_mod.ExecutionResult;
pub const KernelArgs = dispatcher_mod.KernelArgs;

/// Matrix dimensions for matrix operations.
pub const MatrixDims = struct {
    m: usize,
    n: usize,
    k: usize,
};

/// Compiled kernel handle.
pub const CompiledKernel = struct {
    allocator: std.mem.Allocator,
    name: []const u8,
    backend: Backend,
    source: []const u8,
    entry_point: []const u8,
    /// Backend-specific handle.
    handle: ?*anyopaque,

    pub fn deinit(self: *CompiledKernel) void {
        self.allocator.free(self.name);
        self.allocator.free(self.source);
        self.allocator.free(self.entry_point);
        self.* = undefined;
    }
};

/// GPU statistics.
pub const GpuStats = struct {
    kernels_launched: u64,
    buffers_created: u64,
    bytes_allocated: u64,
    host_to_device_transfers: u64,
    device_to_host_transfers: u64,
    total_execution_time_ns: u64,
};

/// GPU configuration (execution-relevant subset).
pub const GpuConfig = struct {
    preferred_backend: ?Backend = null,
    allow_fallback: bool = true,
    memory_mode: buffer_mod.MemoryMode = .automatic,
    max_memory_bytes: usize = 0,
    enable_profiling: bool = false,
    multi_gpu: bool = false,
    load_balance_strategy: @import("device_manager.zig").LoadBalanceStrategy = .memory_aware,
};

// ============================================================================
// High-Level Operations
// ============================================================================

/// Record a kernel metric if profiling is enabled.
fn recordMetric(metrics: *?MetricsCollector, name: []const u8, elapsed: u64) void {
    if (metrics.*) |*m| {
        m.recordKernel(name, elapsed) catch |err| {
            std.log.debug("Failed to record {s} metrics: {t}", .{ name, err });
        };
    }
}

/// Vector addition: result = a + b
pub fn vectorAdd(
    dispatcher: *?KernelDispatcher,
    active_device: ?*const Device,
    metrics: *?MetricsCollector,
    stats: *GpuStats,
    a: *Buffer,
    b: *Buffer,
    result: *Buffer,
) !ExecutionResult {
    const device = active_device orelse return error.NoActiveDevice;

    var timer = time.Timer.start() catch return error.TimerFailed;
    var gpu_executed = false;

    if (dispatcher.*) |*disp| {
        const kernel = disp.getBuiltinKernel(.vector_add) catch null;
        if (kernel) |k| {
            const config = LaunchConfig.for1D(a.elementCount(), 256);
            const exec_result = disp.execute(k, config, .{
                .buffers = &.{ a, b, result },
            }) catch null;
            if (exec_result) |res| {
                gpu_executed = res.gpu_executed;
            }
        }
    }

    if (!gpu_executed) {
        std.log.debug("GPU dispatch failed, falling back to CPU for {s}", .{"vectorAdd"});
        if (a.host_data != null and b.host_data != null and result.host_data != null) {
            const a_data = std.mem.bytesAsSlice(f32, a.host_data.?);
            const b_data = std.mem.bytesAsSlice(f32, b.host_data.?);
            var r_data = std.mem.bytesAsSlice(f32, result.host_data.?);

            const len = @min(a_data.len, @min(b_data.len, r_data.len));
            for (0..len) |i| {
                r_data[i] = a_data[i] + b_data[i];
            }

            result.markHostDirty();
        }
    }

    const elapsed = timer.read();
    stats.kernels_launched += 1;
    stats.total_execution_time_ns += elapsed;
    recordMetric(metrics, "vectorAdd", elapsed);

    return ExecutionResult{
        .execution_time_ns = elapsed,
        .elements_processed = a.elementCount(),
        .bytes_transferred = a.getSize() + b.getSize() + result.getSize(),
        .backend = device.backend,
        .device_id = device.id,
        .gpu_executed = gpu_executed,
    };
}

/// Matrix multiplication: result = a * b
pub fn matrixMultiply(
    dispatcher: *?KernelDispatcher,
    active_device: ?*const Device,
    metrics: *?MetricsCollector,
    stats: *GpuStats,
    a: *Buffer,
    b: *Buffer,
    result: *Buffer,
    dims: MatrixDims,
) !ExecutionResult {
    const device = active_device orelse return error.NoActiveDevice;

    var timer = time.Timer.start() catch return error.TimerFailed;
    var gpu_executed = false;

    if (dispatcher.*) |*disp| {
        const kernel = disp.getBuiltinKernel(.matrix_multiply) catch null;
        if (kernel) |k| {
            const warp_size: u32 = switch (device.backend) {
                .cuda => 32,
                .vulkan => 32,
                .metal => 32,
                else => 32,
            };

            const cc_major: u32 = switch (device.backend) {
                .cuda => 7,
                .vulkan, .metal => 7,
                else => 6,
            };

            const tiling = adaptive_tiling_mod.AdaptiveTiling.init(.{
                .max_threads_per_block = device.maxWorkgroupSize(),
                .max_shared_memory = device.maxSharedMemory(),
                .warp_size = warp_size,
                .compute_capability = .{ .major = cc_major, .minor = 0 },
            });

            const tile = tiling.selectTile(
                @intCast(dims.m),
                @intCast(dims.n),
                @intCast(dims.k),
                .f32,
            );

            const config = LaunchConfig.for2D(dims.n, dims.m, tile.n, tile.m);
            const exec_result = disp.execute(k, config, .{
                .buffers = &.{ a, b, result },
            }) catch null;
            if (exec_result) |res| {
                gpu_executed = res.gpu_executed;
            }
        }
    }

    if (!gpu_executed) {
        std.log.debug("GPU dispatch failed, falling back to CPU for {s}", .{"matrixMultiply"});
        if (a.host_data != null and b.host_data != null and result.host_data != null) {
            const a_data = std.mem.bytesAsSlice(f32, a.host_data.?);
            const b_data = std.mem.bytesAsSlice(f32, b.host_data.?);
            var r_data = std.mem.bytesAsSlice(f32, result.host_data.?);

            for (0..dims.m) |i| {
                for (0..dims.n) |j| {
                    var sum: f32 = 0;
                    for (0..dims.k) |kk| {
                        sum += a_data[i * dims.k + kk] * b_data[kk * dims.n + j];
                    }
                    r_data[i * dims.n + j] = sum;
                }
            }

            result.markHostDirty();
        }
    }

    const elapsed = timer.read();
    stats.kernels_launched += 1;
    stats.total_execution_time_ns += elapsed;
    recordMetric(metrics, "matrixMultiply", elapsed);

    return ExecutionResult{
        .execution_time_ns = elapsed,
        .elements_processed = dims.m * dims.n,
        .bytes_transferred = a.getSize() + b.getSize() + result.getSize(),
        .backend = device.backend,
        .device_id = device.id,
        .gpu_executed = gpu_executed,
    };
}

pub const ReduceResult = struct { value: f32, stats: ExecutionResult };

/// Context passed to reduceSum/dotProduct to allow buffer creation/destruction.
pub const BufferContext = struct {
    allocator: std.mem.Allocator,
    buffers: *std.ArrayListUnmanaged(*Buffer),
    buffer_mutex: *@import("../../services/shared/mod.zig").sync.Mutex,
    active_device: ?*const Device,
    memory_mode: buffer_mod.MemoryMode,
    stats: *GpuStats,
};

/// Reduce sum: returns sum of all elements.
pub fn reduceSum(
    dispatcher: *?KernelDispatcher,
    active_device: ?*const Device,
    metrics: *?MetricsCollector,
    stats: *GpuStats,
    buf_ctx: BufferContext,
    input: *Buffer,
) !ReduceResult {
    const device = active_device orelse return error.NoActiveDevice;

    var timer = time.Timer.start() catch return error.TimerFailed;
    var sum: f32 = 0;
    var gpu_executed = false;

    if (dispatcher.*) |*disp| {
        const kernel = disp.getBuiltinKernel(.reduce_sum) catch null;
        if (kernel) |k| {
            const buffer_pool = @import("buffer_pool.zig");
            const result_buf = buffer_pool.createBuffer(
                buf_ctx.allocator,
                buf_ctx.buffers,
                buf_ctx.buffer_mutex,
                device,
                @sizeOf(f32),
                .{ .mode = .explicit },
                buf_ctx.memory_mode,
            ) catch null;
            if (result_buf) |rbuf| {
                defer buffer_pool.destroyBuffer(buf_ctx.allocator, buf_ctx.buffers, buf_ctx.buffer_mutex, rbuf);
                buf_ctx.stats.buffers_created += 1;
                buf_ctx.stats.bytes_allocated += @sizeOf(f32);

                const config = LaunchConfig.for1D(input.elementCount(), 256);
                const exec_result = disp.execute(k, config, .{
                    .buffers = &.{ input, rbuf },
                }) catch null;
                if (exec_result) |res| {
                    gpu_executed = res.gpu_executed;
                    if (gpu_executed) {
                        var result_val: [1]f32 = undefined;
                        rbuf.read(f32, &result_val) catch |err| {
                            std.log.warn("Failed to read GPU result: {t}", .{err});
                            gpu_executed = false;
                        };
                        sum = result_val[0];
                    }
                }
            }
        }
    }

    if (!gpu_executed) {
        std.log.debug("GPU dispatch failed, falling back to CPU for {s}", .{"reduceSum"});
        if (input.host_data) |host| {
            const data = std.mem.bytesAsSlice(f32, host);
            for (data) |v| {
                sum += v;
            }
        }
    }

    const elapsed = timer.read();
    stats.kernels_launched += 1;
    stats.total_execution_time_ns += elapsed;
    recordMetric(metrics, "reduceSum", elapsed);

    return .{
        .value = sum,
        .stats = ExecutionResult{
            .execution_time_ns = elapsed,
            .elements_processed = input.elementCount(),
            .bytes_transferred = input.getSize(),
            .backend = device.backend,
            .device_id = device.id,
            .gpu_executed = gpu_executed,
        },
    };
}

/// Dot product: returns a . b
pub fn dotProduct(
    dispatcher: *?KernelDispatcher,
    active_device: ?*const Device,
    metrics: *?MetricsCollector,
    stats: *GpuStats,
    buf_ctx: BufferContext,
    a: *Buffer,
    b: *Buffer,
) !ReduceResult {
    const device = active_device orelse return error.NoActiveDevice;

    var timer = time.Timer.start() catch return error.TimerFailed;
    var sum: f32 = 0;
    var gpu_executed = false;

    if (dispatcher.*) |*disp| {
        const kernel = disp.getBuiltinKernel(.dot_product) catch null;
        if (kernel) |k| {
            const buffer_pool = @import("buffer_pool.zig");
            const result_buf = buffer_pool.createBuffer(
                buf_ctx.allocator,
                buf_ctx.buffers,
                buf_ctx.buffer_mutex,
                device,
                @sizeOf(f32),
                .{ .mode = .explicit },
                buf_ctx.memory_mode,
            ) catch null;
            if (result_buf) |rbuf| {
                defer buffer_pool.destroyBuffer(buf_ctx.allocator, buf_ctx.buffers, buf_ctx.buffer_mutex, rbuf);
                buf_ctx.stats.buffers_created += 1;
                buf_ctx.stats.bytes_allocated += @sizeOf(f32);

                const config = LaunchConfig.for1D(a.elementCount(), 256);
                const exec_result = disp.execute(k, config, .{
                    .buffers = &.{ a, b, rbuf },
                }) catch null;
                if (exec_result) |res| {
                    gpu_executed = res.gpu_executed;
                    if (gpu_executed) {
                        var result_val: [1]f32 = undefined;
                        rbuf.read(f32, &result_val) catch |err| {
                            std.log.warn("Failed to read GPU result: {t}", .{err});
                            gpu_executed = false;
                        };
                        sum = result_val[0];
                    }
                }
            }
        }
    }

    if (!gpu_executed) {
        std.log.debug("GPU dispatch failed, falling back to CPU for {s}", .{"dotProduct"});
        if (a.host_data != null and b.host_data != null) {
            const a_data = std.mem.bytesAsSlice(f32, a.host_data.?);
            const b_data = std.mem.bytesAsSlice(f32, b.host_data.?);

            const len = @min(a_data.len, b_data.len);
            for (0..len) |i| {
                sum += a_data[i] * b_data[i];
            }
        }
    }

    const elapsed = timer.read();
    stats.kernels_launched += 1;
    stats.total_execution_time_ns += elapsed;
    recordMetric(metrics, "dotProduct", elapsed);

    return .{
        .value = sum,
        .stats = ExecutionResult{
            .execution_time_ns = elapsed,
            .elements_processed = a.elementCount(),
            .bytes_transferred = a.getSize() + b.getSize(),
            .backend = device.backend,
            .device_id = device.id,
            .gpu_executed = gpu_executed,
        },
    };
}

/// Softmax: output = softmax(input)
pub fn softmax(
    dispatcher: *?KernelDispatcher,
    active_device: ?*const Device,
    metrics: *?MetricsCollector,
    stats: *GpuStats,
    input: *Buffer,
    output: *Buffer,
) !ExecutionResult {
    const device = active_device orelse return error.NoActiveDevice;

    var timer = time.Timer.start() catch return error.TimerFailed;
    var gpu_executed = false;

    if (dispatcher.*) |*disp| {
        const kernel = disp.getBuiltinKernel(.softmax) catch null;
        if (kernel) |k| {
            const config = LaunchConfig.for1D(input.elementCount(), 256);
            const exec_result = disp.execute(k, config, .{
                .buffers = &.{ input, output },
            }) catch null;
            if (exec_result) |res| {
                gpu_executed = res.gpu_executed;
            }
        }
    }

    if (!gpu_executed) {
        std.log.debug("GPU dispatch failed, falling back to CPU for {s}", .{"softmax"});
        if (input.host_data != null and output.host_data != null) {
            const in_data = std.mem.bytesAsSlice(f32, input.host_data.?);
            var out_data = std.mem.bytesAsSlice(f32, output.host_data.?);

            const len = @min(in_data.len, out_data.len);

            var max_val: f32 = in_data[0];
            for (in_data[1..]) |v| {
                if (v > max_val) max_val = v;
            }

            var fsum: f32 = 0;
            for (0..len) |i| {
                out_data[i] = @exp(in_data[i] - max_val);
                fsum += out_data[i];
            }

            for (0..len) |i| {
                out_data[i] /= fsum;
            }

            output.markHostDirty();
        }
    }

    const elapsed = timer.read();
    stats.kernels_launched += 1;
    stats.total_execution_time_ns += elapsed;
    recordMetric(metrics, "softmax", elapsed);

    return ExecutionResult{
        .execution_time_ns = elapsed,
        .elements_processed = input.elementCount(),
        .bytes_transferred = input.getSize() + output.getSize(),
        .backend = device.backend,
        .device_id = device.id,
        .gpu_executed = gpu_executed,
    };
}

// ============================================================================
// Custom Kernel Support
// ============================================================================

/// Compile a kernel from portable source.
pub fn compileKernel(
    allocator: std.mem.Allocator,
    dispatcher: *?KernelDispatcher,
    active_device: ?*const Device,
    source: PortableKernelSource,
) !CompiledKernel {
    const device = active_device orelse return error.NoActiveDevice;
    const ir = source.ir orelse return error.InvalidKernelIR;

    if (dispatcher.*) |*disp| {
        const handle = disp.compileKernel(ir) catch |err| {
            std.log.debug("Dispatcher compilation failed for {s}: {t}, falling back to source-only", .{
                ir.name,
                err,
            });
            return compileKernelSourceOnly(allocator, ir, device);
        };

        var generated = dsl.compile(allocator, ir, device.backend, .{}) catch {
            return CompiledKernel{
                .allocator = allocator,
                .name = try allocator.dupe(u8, ir.name),
                .backend = device.backend,
                .source = try allocator.dupe(u8, ""),
                .entry_point = try allocator.dupe(u8, handle.name),
                .handle = handle.handle,
            };
        };
        defer generated.deinit(allocator);

        return CompiledKernel{
            .allocator = allocator,
            .name = try allocator.dupe(u8, ir.name),
            .backend = device.backend,
            .source = try allocator.dupe(u8, generated.code),
            .entry_point = try allocator.dupe(u8, generated.entry_point),
            .handle = handle.handle,
        };
    }

    return compileKernelSourceOnly(allocator, ir, device);
}

/// Compile kernel IR directly (convenience overload).
pub fn compileKernelIR(
    allocator: std.mem.Allocator,
    dispatcher: *?KernelDispatcher,
    active_device: ?*const Device,
    ir: *const KernelIR,
) !CompiledKernel {
    return compileKernel(allocator, dispatcher, active_device, .{ .name = ir.name, .ir = ir });
}

/// Compile kernel to source code only (no backend handle).
fn compileKernelSourceOnly(allocator: std.mem.Allocator, ir: *const KernelIR, device: *const Device) !CompiledKernel {
    var generated = try dsl.compile(allocator, ir, device.backend, .{});
    defer generated.deinit(allocator);

    return CompiledKernel{
        .allocator = allocator,
        .name = try allocator.dupe(u8, ir.name),
        .backend = device.backend,
        .source = try allocator.dupe(u8, generated.code),
        .entry_point = try allocator.dupe(u8, generated.entry_point),
        .handle = null,
    };
}

/// Launch a compiled kernel.
pub fn launchKernel(
    dispatcher: *?KernelDispatcher,
    active_device: ?*const Device,
    metrics: *?MetricsCollector,
    stats: *GpuStats,
    kernel: *const CompiledKernel,
    config: LaunchConfig,
    args: KernelArgs,
) !ExecutionResult {
    const device = active_device orelse return error.NoActiveDevice;

    var timer = time.Timer.start() catch return error.TimerFailed;
    var gpu_executed = false;
    var bytes_transferred: usize = 0;

    // Sync host-dirty buffers to device before launch
    for (args.buffers) |buf| {
        bytes_transferred += buf.getSize();
        if (buf.isHostDirty()) {
            buf.toDevice() catch |err| {
                std.log.debug("Failed to sync buffer to device: {}", .{err});
            };
            stats.host_to_device_transfers += 1;
        }
    }

    if (dispatcher.*) |*disp| {
        if (kernel.handle != null) {
            const dispatch_handle = dispatcher_mod.CompiledKernelHandle{
                .handle = kernel.handle,
                .name = kernel.name,
                .backend = kernel.backend,
                .workgroup_size = config.local_size orelse .{ 256, 1, 1 },
                .buffer_count = @intCast(args.buffers.len),
                .uniform_count = @intCast(args.uniforms.len),
            };

            const exec_result = disp.execute(dispatch_handle, config, args) catch null;
            if (exec_result) |res| {
                gpu_executed = res.gpu_executed;
            }
        }
    }

    if (gpu_executed) {
        for (args.buffers) |buf| {
            buf.markDeviceDirty();
        }
    }

    const elapsed = timer.read();
    stats.kernels_launched += 1;
    stats.total_execution_time_ns += elapsed;
    recordMetric(metrics, kernel.name, elapsed);

    const total_threads = @as(usize, config.global_size[0]) *
        @as(usize, config.global_size[1]) *
        @as(usize, config.global_size[2]);

    return ExecutionResult{
        .execution_time_ns = elapsed,
        .elements_processed = total_threads,
        .bytes_transferred = bytes_transferred,
        .backend = device.backend,
        .device_id = device.id,
        .gpu_executed = gpu_executed,
    };
}

// ============================================================================
// Stream/Synchronization
// ============================================================================

/// Synchronize all pending operations.
pub fn synchronize(stream_manager: *StreamManager) !void {
    try stream_manager.synchronizeAll();
}

/// Create a new stream.
pub fn createStream(
    stream_manager: *StreamManager,
    active_device: ?*const Device,
    options: StreamOptions,
) !*Stream {
    const device = active_device orelse return error.NoActiveDevice;
    return stream_manager.createStream(device, options);
}

/// Create a new event.
pub fn createEvent(
    stream_manager: *StreamManager,
    active_device: ?*const Device,
    options: EventOptions,
) !*Event {
    const device = active_device orelse return error.NoActiveDevice;
    return stream_manager.createEvent(device, options);
}
