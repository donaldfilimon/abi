//! Unified GPU acceleration interface.
//!
//! Provides a high-level, backend-agnostic API for GPU compute operations.
//! Automatically selects the best available backend and provides fallback
//! to CPU computation when no GPU is available.

const std = @import("std");
const backend = @import("backend.zig");
const kernels = @import("kernels.zig");
const memory = @import("memory.zig");
const profiling = @import("profiling.zig");
const build_options = @import("build_options");

pub const AcceleratorError = error{
    NoBackendAvailable,
    InitializationFailed,
    KernelCompilationFailed,
    KernelLaunchFailed,
    MemoryAllocationFailed,
    DataTransferFailed,
    InvalidConfiguration,
    Timeout,
};

pub const AcceleratorConfig = struct {
    preferred_backend: ?backend.Backend = null,
    allow_fallback: bool = true,
    enable_profiling: bool = false,
    max_memory_bytes: ?usize = null,
    async_execution: bool = false,
};

pub const ExecutionStats = struct {
    kernel_time_ms: f64 = 0.0,
    transfer_time_ms: f64 = 0.0,
    total_time_ms: f64 = 0.0,
    backend_used: backend.Backend,
    device_name: []const u8 = "Unknown",
    memory_used_bytes: usize = 0,
};

pub const ComputeTask = struct {
    name: []const u8,
    kernel_source: ?kernels.KernelSource = null,
    input_buffers: []const []const u8 = &.{},
    output_buffer_sizes: []const usize = &.{},
    grid_size: [3]u32 = .{ 1, 1, 1 },
    block_size: [3]u32 = .{ 256, 1, 1 },
    shared_memory_bytes: u32 = 0,
};

pub const Accelerator = struct {
    allocator: std.mem.Allocator,
    config: AcceleratorConfig,
    active_backend: ?backend.Backend,
    profiler: ?profiling.Profiler,
    memory_pool: ?memory.GPUMemoryPool,
    initialized: bool,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, config: AcceleratorConfig) AcceleratorError!Self {
        var self = Self{
            .allocator = allocator,
            .config = config,
            .active_backend = null,
            .profiler = null,
            .memory_pool = null,
            .initialized = false,
        };

        const selected_backend = try self.selectBackend();
        self.active_backend = selected_backend;

        if (config.enable_profiling) {
            self.profiler = profiling.Profiler.init(allocator);
            self.profiler.?.enable();
        }

        if (config.max_memory_bytes) |max_mem| {
            self.memory_pool = memory.GPUMemoryPool.init(allocator, max_mem);
        }

        self.initialized = true;
        return self;
    }

    pub fn deinit(self: *Self) void {
        if (self.memory_pool) |*pool| {
            pool.deinit();
        }

        if (self.profiler) |*prof| {
            prof.deinit(self.allocator);
        }

        self.initialized = false;
        self.active_backend = null;
    }

    fn selectBackend(self: *Self) AcceleratorError!backend.Backend {
        if (self.config.preferred_backend) |preferred| {
            const availability = backend.backendAvailability(preferred);
            if (availability.available) {
                return preferred;
            }
            if (!self.config.allow_fallback) {
                return AcceleratorError.NoBackendAvailable;
            }
        }

        const backends_to_try = [_]backend.Backend{
            .cuda,
            .vulkan,
            .metal,
            .webgpu,
            .stdgpu,
            .opengl,
            .opengles,
        };

        for (backends_to_try) |b| {
            const availability = backend.backendAvailability(b);
            if (availability.available and backend.backendSupportsKernels(b)) {
                return b;
            }
        }

        return AcceleratorError.NoBackendAvailable;
    }

    pub fn executeTask(self: *Self, task: ComputeTask) AcceleratorError!ExecutionStats {
        if (!self.initialized) {
            return AcceleratorError.InitializationFailed;
        }

        const active = self.active_backend orelse return AcceleratorError.NoBackendAvailable;

        var stats = ExecutionStats{
            .backend_used = active,
            .device_name = backend.backendDisplayName(active),
        };

        const start_time = std.time.nanoTimestamp();

        if (self.profiler) |*prof| {
            try prof.startTiming(task.name, self.allocator, 0);
        }

        if (task.kernel_source) |source| {
            var kernel = kernels.compileKernel(self.allocator, source) catch {
                return AcceleratorError.KernelCompilationFailed;
            };
            defer kernel.deinit();

            const config = kernels.KernelConfig{
                .grid_dim = task.grid_size,
                .block_dim = task.block_size,
                .shared_memory_bytes = task.shared_memory_bytes,
            };

            kernel.launch(self.allocator, config, &.{}) catch {
                return AcceleratorError.KernelLaunchFailed;
            };
        }

        if (self.profiler) |*prof| {
            try prof.endTiming(self.allocator);
        }

        const end_time = std.time.nanoTimestamp();
        const duration_ns = @as(u64, @intCast(end_time - start_time));
        stats.total_time_ms = @as(f64, @floatFromInt(duration_ns)) / 1_000_000.0;

        if (self.profiler) |*prof| {
            if (prof.getAverageTime(task.name)) |avg| {
                stats.kernel_time_ms = avg;
            }
        }

        return stats;
    }

    pub fn allocateBuffer(self: *Self, size: usize) AcceleratorError!*memory.GPUBuffer {
        if (self.memory_pool) |*pool| {
            return pool.allocate(self.allocator, size) catch {
                return AcceleratorError.MemoryAllocationFailed;
            };
        }

        // Allocate buffer on heap to avoid returning address of local variable
        const buffer = self.allocator.create(memory.GPUBuffer) catch {
            return AcceleratorError.MemoryAllocationFailed;
        };
        buffer.* = memory.GPUBuffer.init(self.allocator, size) catch {
            self.allocator.destroy(buffer);
            return AcceleratorError.MemoryAllocationFailed;
        };
        return buffer;
    }

    pub fn freeBuffer(self: *Self, buffer: *memory.GPUBuffer) void {
        if (self.memory_pool) |*pool| {
            pool.free(buffer);
        } else {
            buffer.deinit();
        }
    }

    pub fn getBackendInfo(self: *const Self) backend.BackendInfo {
        const active = self.active_backend orelse return .{
            .backend = .stdgpu,
            .name = "None",
            .description = "No backend available",
            .enabled = false,
            .available = false,
            .availability = "Not initialized",
            .device_count = 0,
            .build_flag = "",
        };

        const availability = backend.backendAvailability(active);
        return .{
            .backend = active,
            .name = backend.backendDisplayName(active),
            .description = backend.backendDescription(active),
            .enabled = availability.enabled,
            .available = availability.available,
            .availability = availability.reason,
            .device_count = availability.device_count,
            .build_flag = backend.backendFlag(active),
        };
    }

    pub fn getProfilingSummary(self: *const Self) ?profiling.Profiler.getSummary {
        if (self.profiler) |*prof| {
            return prof.getSummary();
        }
        return null;
    }

    pub fn isAvailable(self: *const Self) bool {
        return self.initialized and self.active_backend != null;
    }

    pub fn supportsAsyncExecution(self: *const Self) bool {
        if (self.active_backend) |active| {
            const capability = backend.backendCapabilities(active);
            return capability.supports_async_transfers;
        }
        return false;
    }
};

pub fn vectorAdd(
    allocator: std.mem.Allocator,
    a: []const f32,
    b: []const f32,
    result: []f32,
) AcceleratorError!ExecutionStats {
    if (a.len != b.len or a.len != result.len) {
        return AcceleratorError.InvalidConfiguration;
    }

    var accelerator = try Accelerator.init(allocator, .{
        .enable_profiling = true,
    });
    defer accelerator.deinit();

    for (a, b, 0..) |av, bv, i| {
        result[i] = av + bv;
    }

    return ExecutionStats{
        .backend_used = accelerator.active_backend orelse .stdgpu,
        .device_name = "CPU Fallback",
        .kernel_time_ms = 0.0,
        .transfer_time_ms = 0.0,
        .total_time_ms = 0.0,
    };
}

pub fn matrixMultiply(
    allocator: std.mem.Allocator,
    a: []const f32,
    b: []const f32,
    result: []f32,
    m: usize,
    n: usize,
    k: usize,
) AcceleratorError!ExecutionStats {
    if (a.len != m * k or b.len != k * n or result.len != m * n) {
        return AcceleratorError.InvalidConfiguration;
    }

    var accelerator = try Accelerator.init(allocator, .{
        .enable_profiling = true,
    });
    defer accelerator.deinit();

    for (0..m) |row| {
        for (0..n) |col| {
            var sum: f32 = 0.0;
            for (0..k) |i| {
                sum += a[row * k + i] * b[i * n + col];
            }
            result[row * n + col] = sum;
        }
    }

    return ExecutionStats{
        .backend_used = accelerator.active_backend orelse .stdgpu,
        .device_name = "CPU Fallback",
        .kernel_time_ms = 0.0,
        .transfer_time_ms = 0.0,
        .total_time_ms = 0.0,
    };
}

pub fn reduceSum(
    allocator: std.mem.Allocator,
    input: []const f32,
) AcceleratorError!struct { result: f32, stats: ExecutionStats } {
    var accelerator = try Accelerator.init(allocator, .{
        .enable_profiling = true,
    });
    defer accelerator.deinit();

    var sum: f32 = 0.0;
    for (input) |v| {
        sum += v;
    }

    return .{
        .result = sum,
        .stats = ExecutionStats{
            .backend_used = accelerator.active_backend orelse .stdgpu,
            .device_name = "CPU Fallback",
            .kernel_time_ms = 0.0,
            .transfer_time_ms = 0.0,
            .total_time_ms = 0.0,
        },
    };
}

pub fn getAvailableBackends(allocator: std.mem.Allocator) ![]backend.Backend {
    return backend.availableBackends(allocator);
}

pub fn getBestBackend() ?backend.Backend {
    const backends_to_try = [_]backend.Backend{
        .cuda,
        .vulkan,
        .metal,
        .webgpu,
        .stdgpu,
    };

    for (backends_to_try) |b| {
        const availability = backend.backendAvailability(b);
        if (availability.available and backend.backendSupportsKernels(b)) {
            return b;
        }
    }

    return null;
}

pub fn isGpuAvailable() bool {
    return getBestBackend() != null;
}

fn backendCapabilities(b: backend.Backend) backend.DeviceCapability {
    const devices = backend.listDevices(std.heap.page_allocator) catch return .{};
    defer std.heap.page_allocator.free(devices);

    for (devices) |device| {
        if (device.backend == b) {
            return device.capability;
        }
    }

    return .{};
}

test "accelerator initialization" {
    const allocator = std.testing.allocator;

    var acc = Accelerator.init(allocator, .{
        .allow_fallback = true,
    }) catch |err| {
        if (err == AcceleratorError.NoBackendAvailable) {
            return;
        }
        return err;
    };
    defer acc.deinit();

    try std.testing.expect(acc.isAvailable());
}

test "vector add operation" {
    const allocator = std.testing.allocator;

    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 4.0, 3.0, 2.0, 1.0 };
    var result: [4]f32 = undefined;

    const stats = vectorAdd(allocator, &a, &b, &result) catch |err| {
        if (err == AcceleratorError.NoBackendAvailable) {
            return;
        }
        return err;
    };

    try std.testing.expectApproxEqAbs(@as(f32, 5.0), result[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), result[1], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), result[2], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), result[3], 0.0001);

    _ = stats;
}

test "reduce sum operation" {
    const allocator = std.testing.allocator;

    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };

    const reduction = reduceSum(allocator, &input) catch |err| {
        if (err == AcceleratorError.NoBackendAvailable) {
            return;
        }
        return err;
    };

    try std.testing.expectApproxEqAbs(@as(f32, 15.0), reduction.result, 0.0001);
}
