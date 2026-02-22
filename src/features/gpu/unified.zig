//! Unified GPU API
//!
//! Main entry point for the unified GPU API.
//! Provides a single interface for all GPU backends with:
//! - High-level operations (vectorAdd, matrixMultiply, etc.)
//! - Custom kernel compilation and execution
//! - Smart buffer management
//! - Device discovery and selection
//! - Stream/event synchronization
//!
//! ## Quick Start
//!
//! ```zig
//! var gpu = try Gpu.init(allocator, .{});
//! defer gpu.deinit();
//!
//! // Create buffers
//! var a = try gpu.createBufferFromSlice(f32, &[_]f32{ 1, 2, 3, 4 }, .{});
//! var b = try gpu.createBufferFromSlice(f32, &[_]f32{ 5, 6, 7, 8 }, .{});
//! var result = try gpu.createBuffer(4 * @sizeOf(f32), .{});
//! defer { gpu.destroyBuffer(&a); gpu.destroyBuffer(&b); gpu.destroyBuffer(&result); }
//!
//! // Run operation
//! _ = try gpu.vectorAdd(&a, &b, &result);
//!
//! // Read results
//! var output: [4]f32 = undefined;
//! try result.read(f32, &output);
//! ```

const std = @import("std");
const time = @import("../../services/shared/time.zig");
const sync = @import("../../services/shared/sync.zig");
const backend_mod = @import("backend.zig");
const device_mod = @import("device.zig");
const stream_mod = @import("stream.zig");
const buffer_mod = @import("unified_buffer.zig");
const dsl = @import("dsl/mod.zig");
const multi_device = @import("multi_device.zig");
const metrics_mod = @import("metrics.zig");
const dispatcher_mod = @import("dispatch/coordinator.zig");
const adaptive_tiling_mod = @import("adaptive_tiling.zig");
const policy_mod = @import("policy/mod.zig");
const interface_mod = @import("interface.zig");

const Mutex = sync.Mutex;

// Re-export key types
pub const Backend = backend_mod.Backend;
pub const Device = device_mod.Device;
pub const DeviceSelector = device_mod.DeviceSelector;
pub const DeviceType = device_mod.DeviceType;
pub const DeviceFeature = device_mod.DeviceFeature;
pub const Stream = stream_mod.Stream;
pub const StreamOptions = stream_mod.StreamOptions;
pub const StreamPriority = stream_mod.StreamPriority;
pub const Event = stream_mod.Event;
pub const EventOptions = stream_mod.EventOptions;
pub const Buffer = buffer_mod.Buffer;
pub const BufferOptions = buffer_mod.BufferOptions;
pub const MemoryMode = buffer_mod.MemoryMode;
pub const MemoryLocation = buffer_mod.MemoryLocation;
pub const AccessHint = buffer_mod.AccessHint;
pub const ElementType = buffer_mod.ElementType;

// Re-export DSL types
pub const KernelBuilder = dsl.KernelBuilder;
pub const KernelIR = dsl.KernelIR;
pub const PortableKernelSource = dsl.PortableKernelSource;

// Re-export multi-device types
pub const DeviceGroup = multi_device.DeviceGroup;
pub const WorkDistribution = multi_device.WorkDistribution;
pub const DeviceBarrier = multi_device.DeviceBarrier;
pub const PeerTransfer = multi_device.PeerTransfer;

// Re-export metrics types
pub const MetricsCollector = metrics_mod.MetricsCollector;
pub const MetricsSummary = metrics_mod.Summary;
pub const KernelMetrics = metrics_mod.KernelMetrics;

// Re-export interface types
pub const DeviceCaps = interface_mod.DeviceCaps;

// Re-export dispatcher types (canonical definitions)
pub const KernelDispatcher = dispatcher_mod.KernelDispatcher;
pub const DispatchError = dispatcher_mod.DispatchError;
pub const LaunchConfig = dispatcher_mod.LaunchConfig;
pub const ExecutionResult = dispatcher_mod.ExecutionResult;
pub const KernelArgs = dispatcher_mod.KernelArgs;

// Aliases for backward compatibility
pub const DispatcherLaunchConfig = LaunchConfig;
pub const DispatcherExecutionResult = ExecutionResult;

/// Load balance strategy for multi-GPU.
pub const LoadBalanceStrategy = enum {
    /// Round-robin distribution.
    round_robin,
    /// Memory-aware distribution.
    memory_aware,
    /// Compute-aware distribution.
    compute_aware,
    /// Manual assignment.
    manual,
};

/// GPU configuration.
pub const GpuConfig = struct {
    /// Preferred backend (null = auto-select best).
    preferred_backend: ?Backend = null,
    /// Allow fallback to other backends if preferred unavailable.
    allow_fallback: bool = true,
    /// Memory management mode.
    memory_mode: MemoryMode = .automatic,
    /// Maximum memory to use (0 = unlimited).
    max_memory_bytes: usize = 0,
    /// Enable profiling.
    enable_profiling: bool = false,
    /// Enable multi-GPU support.
    multi_gpu: bool = false,
    /// Load balance strategy for multi-GPU.
    load_balance_strategy: LoadBalanceStrategy = .memory_aware,
};

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
        // Backend would free handle here
        self.* = undefined;
    }
};

/// GPU memory information.
pub const MemoryInfo = struct {
    total_bytes: u64,
    used_bytes: u64,
    free_bytes: u64,
    peak_used_bytes: u64,
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

/// Health status.
pub const HealthStatus = enum {
    healthy,
    degraded,
    unhealthy,
    unknown,
};

/// Multi-GPU configuration.
pub const MultiGpuConfig = struct {
    /// Devices to use (empty = use all).
    devices: []const u32 = &.{},
    /// Load balance strategy.
    strategy: LoadBalanceStrategy = .memory_aware,
};

/// Main unified GPU API.
pub const Gpu = struct {
    allocator: std.mem.Allocator,
    config: GpuConfig,

    // Managers
    device_manager: device_mod.DeviceManager,
    stream_manager: stream_mod.StreamManager,

    // Kernel dispatcher for unified backend execution
    dispatcher: ?KernelDispatcher,

    // Multi-GPU support
    device_group: ?DeviceGroup,

    // Metrics and profiling
    metrics: ?MetricsCollector,

    // Active state
    active_device: ?*const Device,
    default_stream: ?*Stream,

    // Buffer tracking
    buffers: std.ArrayListUnmanaged(*Buffer),
    buffer_mutex: Mutex,

    // Statistics
    stats: GpuStats,

    /// Initialize the unified GPU API.
    pub fn init(allocator: std.mem.Allocator, config: GpuConfig) !Gpu {
        var effective_config = config;
        if (effective_config.memory_mode == .automatic) {
            const hints = policy_mod.optimizationHintsForPlatform(policy_mod.classifyBuiltin());
            if (hints.prefer_unified_memory) {
                effective_config.memory_mode = .unified;
            }
        }

        var device_manager = try device_mod.DeviceManager.init(allocator);
        errdefer device_manager.deinit();

        var stream_manager = stream_mod.StreamManager.init(allocator);
        errdefer stream_manager.deinit();

        // Initialize multi-GPU if enabled
        var device_group: ?DeviceGroup = null;
        if (effective_config.multi_gpu) {
            const multi_config = multi_device.MultiDeviceConfig{
                .strategy = switch (effective_config.load_balance_strategy) {
                    .round_robin => .round_robin,
                    .memory_aware => .memory_aware,
                    .compute_aware => .capability_weighted,
                    .manual => .pinned,
                },
            };
            device_group = DeviceGroup.init(allocator, multi_config) catch null;
        }
        errdefer if (device_group) |*dg| dg.deinit();

        // Initialize metrics if profiling enabled
        var metrics: ?MetricsCollector = null;
        if (effective_config.enable_profiling) {
            metrics = MetricsCollector.init(allocator);
        }
        errdefer if (metrics) |*m| m.deinit();

        // Select initial device
        var active_device: ?*const Device = null;
        var default_stream: ?*Stream = null;

        if (device_manager.hasDevices()) {
            if (effective_config.preferred_backend) |backend| {
                active_device = device_manager.selectDevice(.{ .by_backend = backend }) catch null;
            }

            if (active_device == null and effective_config.allow_fallback) {
                active_device = device_manager.selectBestDevice() catch null;
            }

            // Create default stream for active device
            if (active_device) |device| {
                default_stream = stream_manager.createStream(device, .{}) catch null;
            }
        }

        // Initialize kernel dispatcher for active device
        var disp: ?KernelDispatcher = null;
        if (active_device) |dev| {
            disp = KernelDispatcher.init(allocator, dev.backend, dev) catch null;
        }
        errdefer if (disp) |*d| d.deinit();

        return .{
            .allocator = allocator,
            .config = effective_config,
            .device_manager = device_manager,
            .stream_manager = stream_manager,
            .dispatcher = disp,
            .device_group = device_group,
            .metrics = metrics,
            .active_device = active_device,
            .default_stream = default_stream,
            .buffers = .empty,
            .buffer_mutex = .{},
            .stats = std.mem.zeroes(GpuStats),
        };
    }

    /// Deinitialize and cleanup.
    pub fn deinit(self: *Gpu) void {
        // Destroy all buffers
        self.buffer_mutex.lock();
        for (self.buffers.items) |buf| {
            buf.deinit();
            self.allocator.destroy(buf);
        }
        self.buffers.deinit(self.allocator);
        self.buffer_mutex.unlock();

        // Clean up dispatcher
        if (self.dispatcher) |*d| {
            d.deinit();
        }

        // Clean up metrics
        if (self.metrics) |*m| {
            m.deinit();
        }

        // Clean up device group
        if (self.device_group) |*dg| {
            dg.deinit();
        }

        self.stream_manager.deinit();
        self.device_manager.deinit();
        self.* = undefined;
    }

    // ========================================================================
    // Device Management
    // ========================================================================

    /// Select a device based on criteria.
    pub fn selectDevice(self: *Gpu, selector: DeviceSelector) !void {
        const device = try self.device_manager.selectDevice(selector);
        self.active_device = device;

        // Create stream for new device if needed
        if (self.default_stream == null) {
            self.default_stream = try self.stream_manager.createStream(device, .{});
        }
    }

    /// Get the currently active device.
    pub fn getActiveDevice(self: *const Gpu) ?*const Device {
        return self.active_device;
    }

    /// List all available devices.
    pub fn listDevices(self: *const Gpu) []const Device {
        return self.device_manager.listDevices();
    }

    /// Enable multi-GPU mode.
    pub fn enableMultiGpu(self: *Gpu, config: MultiGpuConfig) !void {
        // Initialize device group if not already
        if (self.device_group == null) {
            const multi_config = multi_device.MultiDeviceConfig{
                .strategy = switch (config.strategy) {
                    .round_robin => .round_robin,
                    .memory_aware => .memory_aware,
                    .compute_aware => .capability_weighted,
                    .manual => .pinned,
                },
                .preferred_devices = config.devices,
            };
            self.device_group = try DeviceGroup.init(self.allocator, multi_config);
        }

        // Configure active devices
        if (self.device_group) |*dg| {
            if (config.devices.len > 0) {
                // First disable all, then enable specified
                for (dg.getAllDevices()) |device| {
                    dg.disableDevice(device.id);
                }
                for (config.devices) |device_id| {
                    try dg.enableDevice(device_id);
                }
            }
        }
    }

    /// Get multi-GPU device group (if enabled).
    pub fn getDeviceGroup(self: *Gpu) ?*DeviceGroup {
        if (self.device_group) |*dg| {
            return dg;
        }
        return null;
    }

    /// Distribute work across multiple GPUs.
    pub fn distributeWork(self: *Gpu, total_work: usize) ![]WorkDistribution {
        if (self.device_group) |*dg| {
            return dg.distributeWork(total_work);
        }
        // Single device fallback
        var result = try self.allocator.alloc(WorkDistribution, 1);
        result[0] = .{
            .device_id = if (self.active_device) |d| d.id else 0,
            .offset = 0,
            .size = total_work,
        };
        return result;
    }

    // ========================================================================
    // Buffer Management
    // ========================================================================

    /// Create a new buffer.
    pub fn createBuffer(self: *Gpu, size: usize, options: BufferOptions) !*Buffer {
        const device = self.active_device orelse return error.NoActiveDevice;

        var opts = options;
        if (opts.mode == .automatic and self.config.memory_mode != .automatic) {
            opts.mode = self.config.memory_mode;
        }

        const buffer = try self.allocator.create(Buffer);
        errdefer self.allocator.destroy(buffer);

        buffer.* = try Buffer.init(self.allocator, size, device, opts);

        self.buffer_mutex.lock();
        defer self.buffer_mutex.unlock();
        try self.buffers.append(self.allocator, buffer);

        self.stats.buffers_created += 1;
        self.stats.bytes_allocated += size;

        return buffer;
    }

    /// Create a buffer from a typed slice.
    pub fn createBufferFromSlice(
        self: *Gpu,
        comptime T: type,
        data: []const T,
        options: BufferOptions,
    ) !*Buffer {
        const device = self.active_device orelse return error.NoActiveDevice;

        var opts = options;
        if (opts.mode == .automatic and self.config.memory_mode != .automatic) {
            opts.mode = self.config.memory_mode;
        }

        const buffer = try self.allocator.create(Buffer);
        errdefer self.allocator.destroy(buffer);

        buffer.* = try buffer_mod.createFromSlice(self.allocator, T, data, device, opts);

        self.buffer_mutex.lock();
        defer self.buffer_mutex.unlock();
        try self.buffers.append(self.allocator, buffer);

        self.stats.buffers_created += 1;
        self.stats.bytes_allocated += data.len * @sizeOf(T);

        return buffer;
    }

    /// Destroy a buffer.
    pub fn destroyBuffer(self: *Gpu, buffer: *Buffer) void {
        self.buffer_mutex.lock();
        defer self.buffer_mutex.unlock();

        // Remove from tracking list
        for (self.buffers.items, 0..) |b, i| {
            if (b == buffer) {
                _ = self.buffers.swapRemove(i);
                break;
            }
        }

        buffer.deinit();
        self.allocator.destroy(buffer);
    }

    // ========================================================================
    // High-Level Operations
    // ========================================================================

    /// Vector addition: result = a + b
    pub fn vectorAdd(self: *Gpu, a: *Buffer, b: *Buffer, result: *Buffer) !ExecutionResult {
        const device = self.active_device orelse return error.NoActiveDevice;

        var timer = time.Timer.start() catch return error.TimerFailed;
        var gpu_executed = false;

        // Try dispatcher-based execution first
        if (self.dispatcher) |*disp| {
            const kernel = disp.getBuiltinKernel(.vector_add) catch null;
            if (kernel) |k| {
                const config = dispatcher_mod.LaunchConfig.for1D(a.elementCount(), 256);
                const exec_result = disp.execute(k, config, .{
                    .buffers = &.{ a, b, result },
                }) catch null;
                if (exec_result) |res| {
                    gpu_executed = res.gpu_executed;
                }
            }
        }

        // Fallback to host computation if dispatcher not available or failed
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
        self.stats.kernels_launched += 1;
        self.stats.total_execution_time_ns += elapsed;

        // Record metrics if profiling enabled
        if (self.metrics) |*m| {
            m.recordKernel("vectorAdd", elapsed) catch |err| {
                std.log.debug("Failed to record vectorAdd metrics: {t}", .{err});
            };
        }

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
        self: *Gpu,
        a: *Buffer,
        b: *Buffer,
        result: *Buffer,
        dims: MatrixDims,
    ) !ExecutionResult {
        const device = self.active_device orelse return error.NoActiveDevice;

        var timer = time.Timer.start() catch return error.TimerFailed;
        var gpu_executed = false;

        // Try dispatcher-based execution first
        if (self.dispatcher) |*disp| {
            const kernel = disp.getBuiltinKernel(.matrix_multiply) catch null;
            if (kernel) |k| {
                // Use adaptive tiling based on matrix dimensions and device capabilities
                // Default warp size: 32 for NVIDIA, 64 for AMD, 32 for others
                const warp_size: u32 = switch (device.backend) {
                    .cuda => 32,
                    .vulkan => 32, // Varies, but 32 is common
                    .metal => 32,
                    else => 32,
                };

                // Default compute capability based on backend
                // CUDA: assume SM 7.0 (Volta) as baseline
                // Others: use conservative defaults
                const cc_major: u32 = switch (device.backend) {
                    .cuda => 7,
                    .vulkan, .metal => 7, // Similar tier
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

                const config = dispatcher_mod.LaunchConfig.for2D(dims.n, dims.m, tile.n, tile.m);
                const exec_result = disp.execute(k, config, .{
                    .buffers = &.{ a, b, result },
                }) catch null;
                if (exec_result) |res| {
                    gpu_executed = res.gpu_executed;
                }
            }
        }

        // Fallback to host computation if dispatcher not available or failed
        if (!gpu_executed) {
            std.log.debug("GPU dispatch failed, falling back to CPU for {s}", .{"matrixMultiply"});
            if (a.host_data != null and b.host_data != null and result.host_data != null) {
                const a_data = std.mem.bytesAsSlice(f32, a.host_data.?);
                const b_data = std.mem.bytesAsSlice(f32, b.host_data.?);
                var r_data = std.mem.bytesAsSlice(f32, result.host_data.?);

                // C[i,j] = sum(A[i,k] * B[k,j])
                for (0..dims.m) |i| {
                    for (0..dims.n) |j| {
                        var sum: f32 = 0;
                        for (0..dims.k) |k| {
                            sum += a_data[i * dims.k + k] * b_data[k * dims.n + j];
                        }
                        r_data[i * dims.n + j] = sum;
                    }
                }

                result.markHostDirty();
            }
        }

        const elapsed = timer.read();
        self.stats.kernels_launched += 1;
        self.stats.total_execution_time_ns += elapsed;

        // Record metrics if profiling enabled
        if (self.metrics) |*m| {
            m.recordKernel("matrixMultiply", elapsed) catch |err| {
                std.log.debug("Failed to record matrixMultiply metrics: {t}", .{err});
            };
        }

        return ExecutionResult{
            .execution_time_ns = elapsed,
            .elements_processed = dims.m * dims.n,
            .bytes_transferred = a.getSize() + b.getSize() + result.getSize(),
            .backend = device.backend,
            .device_id = device.id,
            .gpu_executed = gpu_executed,
        };
    }

    /// Reduce sum: returns sum of all elements.
    pub fn reduceSum(self: *Gpu, input: *Buffer) !struct { value: f32, stats: ExecutionResult } {
        const device = self.active_device orelse return error.NoActiveDevice;

        var timer = time.Timer.start() catch return error.TimerFailed;
        var sum: f32 = 0;
        var gpu_executed = false;

        // Try dispatcher-based execution first
        if (self.dispatcher) |*disp| {
            const kernel = disp.getBuiltinKernel(.reduce_sum) catch null;
            if (kernel) |k| {
                // Create a temporary result buffer for the reduction
                const result_buf = self.createBuffer(@sizeOf(f32), .{ .mode = .explicit }) catch null;
                if (result_buf) |rbuf| {
                    defer self.destroyBuffer(rbuf);

                    const config = dispatcher_mod.LaunchConfig.for1D(input.elementCount(), 256);
                    const exec_result = disp.execute(k, config, .{
                        .buffers = &.{ input, rbuf },
                    }) catch null;
                    if (exec_result) |res| {
                        gpu_executed = res.gpu_executed;
                        if (gpu_executed) {
                            var result_val: [1]f32 = undefined;
                            rbuf.read(f32, &result_val) catch |err| {
                                std.log.warn("Failed to read GPU result: {t}", .{err});
                                gpu_executed = false; // Fall back to CPU
                            };
                            sum = result_val[0];
                        }
                    }
                }
            }
        }

        // Fallback to host computation
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
        self.stats.kernels_launched += 1;
        self.stats.total_execution_time_ns += elapsed;

        // Record metrics if profiling enabled
        if (self.metrics) |*m| {
            m.recordKernel("reduceSum", elapsed) catch |err| {
                std.log.debug("Failed to record reduceSum metrics: {t}", .{err});
            };
        }

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

    /// Dot product: returns a · b
    pub fn dotProduct(self: *Gpu, a: *Buffer, b: *Buffer) !struct { value: f32, stats: ExecutionResult } {
        const device = self.active_device orelse return error.NoActiveDevice;

        var timer = time.Timer.start() catch return error.TimerFailed;
        var sum: f32 = 0;
        var gpu_executed = false;

        // Try dispatcher-based execution first
        if (self.dispatcher) |*disp| {
            const kernel = disp.getBuiltinKernel(.dot_product) catch null;
            if (kernel) |k| {
                // Create a temporary result buffer for the dot product
                const result_buf = self.createBuffer(@sizeOf(f32), .{ .mode = .explicit }) catch null;
                if (result_buf) |rbuf| {
                    defer self.destroyBuffer(rbuf);

                    const config = dispatcher_mod.LaunchConfig.for1D(a.elementCount(), 256);
                    const exec_result = disp.execute(k, config, .{
                        .buffers = &.{ a, b, rbuf },
                    }) catch null;
                    if (exec_result) |res| {
                        gpu_executed = res.gpu_executed;
                        if (gpu_executed) {
                            var result_val: [1]f32 = undefined;
                            rbuf.read(f32, &result_val) catch |err| {
                                std.log.warn("Failed to read GPU result: {t}", .{err});
                                gpu_executed = false; // Fall back to CPU
                            };
                            sum = result_val[0];
                        }
                    }
                }
            }
        }

        // Fallback to host computation
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
        self.stats.kernels_launched += 1;
        self.stats.total_execution_time_ns += elapsed;

        // Record metrics if profiling enabled
        if (self.metrics) |*m| {
            m.recordKernel("dotProduct", elapsed) catch |err| {
                std.log.debug("Failed to record dotProduct metrics: {t}", .{err});
            };
        }

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
    pub fn softmax(self: *Gpu, input: *Buffer, output: *Buffer) !ExecutionResult {
        const device = self.active_device orelse return error.NoActiveDevice;

        var timer = time.Timer.start() catch return error.TimerFailed;
        var gpu_executed = false;

        // Try dispatcher-based execution first
        if (self.dispatcher) |*disp| {
            const kernel = disp.getBuiltinKernel(.softmax) catch null;
            if (kernel) |k| {
                const config = dispatcher_mod.LaunchConfig.for1D(input.elementCount(), 256);
                const exec_result = disp.execute(k, config, .{
                    .buffers = &.{ input, output },
                }) catch null;
                if (exec_result) |res| {
                    gpu_executed = res.gpu_executed;
                }
            }
        }

        // Fallback to host computation
        if (!gpu_executed) {
            std.log.debug("GPU dispatch failed, falling back to CPU for {s}", .{"softmax"});
            if (input.host_data != null and output.host_data != null) {
                const in_data = std.mem.bytesAsSlice(f32, input.host_data.?);
                var out_data = std.mem.bytesAsSlice(f32, output.host_data.?);

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

                output.markHostDirty();
            }
        }

        const elapsed = timer.read();
        self.stats.kernels_launched += 1;
        self.stats.total_execution_time_ns += elapsed;

        // Record metrics if profiling enabled
        if (self.metrics) |*m| {
            m.recordKernel("softmax", elapsed) catch |err| {
                std.log.debug("Failed to record softmax metrics: {t}", .{err});
            };
        }

        return ExecutionResult{
            .execution_time_ns = elapsed,
            .elements_processed = input.elementCount(),
            .bytes_transferred = input.getSize() + output.getSize(),
            .backend = device.backend,
            .device_id = device.id,
            .gpu_executed = gpu_executed,
        };
    }

    // ========================================================================
    // Custom Kernel Support
    // ========================================================================

    /// Compile a kernel from portable source.
    ///
    /// Generates backend-specific code from the portable kernel IR, then
    /// delegates to the dispatcher's backend VTable to obtain a compiled
    /// handle that can be launched later with `launchKernel()`.
    ///
    /// The `source.ir` field must be set (non-null) for compilation to succeed.
    pub fn compileKernel(self: *Gpu, source: PortableKernelSource) !CompiledKernel {
        const device = self.active_device orelse return error.NoActiveDevice;
        const ir = source.ir orelse return error.InvalidKernelIR;

        // If we have a dispatcher, delegate to it — this compiles IR through
        // the DSL compiler *and* sends the result to the backend VTable.
        if (self.dispatcher) |*disp| {
            const handle = disp.compileKernel(ir) catch |err| {
                std.log.debug("Dispatcher compilation failed for {s}: {t}, falling back to source-only", .{
                    ir.name,
                    err,
                });
                // Fall through to source-only path below
                return self.compileKernelSourceOnly(ir, device);
            };

            // Duplicate the generated code for the CompiledKernel (the
            // dispatcher keeps its own copy via the cache).
            var generated = dsl.compile(self.allocator, ir, device.backend, .{}) catch {
                // Even if re-generation fails, we have the handle
                return CompiledKernel{
                    .allocator = self.allocator,
                    .name = try self.allocator.dupe(u8, ir.name),
                    .backend = device.backend,
                    .source = try self.allocator.dupe(u8, ""),
                    .entry_point = try self.allocator.dupe(u8, handle.name),
                    .handle = handle.handle,
                };
            };
            defer generated.deinit(self.allocator);

            return CompiledKernel{
                .allocator = self.allocator,
                .name = try self.allocator.dupe(u8, ir.name),
                .backend = device.backend,
                .source = try self.allocator.dupe(u8, generated.code),
                .entry_point = try self.allocator.dupe(u8, generated.entry_point),
                .handle = handle.handle,
            };
        }

        // No dispatcher available — compile to source only (no backend handle).
        return self.compileKernelSourceOnly(ir, device);
    }

    /// Compile kernel IR directly (convenience overload).
    ///
    /// Wraps the IR pointer in a `PortableKernelSource` and delegates to
    /// `compileKernel`.
    pub fn compileKernelIR(self: *Gpu, ir: *const KernelIR) !CompiledKernel {
        return self.compileKernel(.{ .name = ir.name, .ir = ir });
    }

    /// Compile kernel to source code only (no backend handle).
    /// Used as fallback when the dispatcher is unavailable.
    fn compileKernelSourceOnly(self: *Gpu, ir: *const KernelIR, device: *const Device) !CompiledKernel {
        var generated = try dsl.compile(self.allocator, ir, device.backend, .{});
        defer generated.deinit(self.allocator);

        return CompiledKernel{
            .allocator = self.allocator,
            .name = try self.allocator.dupe(u8, ir.name),
            .backend = device.backend,
            .source = try self.allocator.dupe(u8, generated.code),
            .entry_point = try self.allocator.dupe(u8, generated.entry_point),
            .handle = null,
        };
    }

    /// Launch a compiled kernel.
    ///
    /// Dispatches the compiled kernel to the backend for execution. The `args`
    /// parameter accepts a `KernelArgs` struct containing buffer and uniform
    /// arguments. Host-dirty buffers are automatically synced to the device
    /// before launch.
    pub fn launchKernel(
        self: *Gpu,
        kernel: *const CompiledKernel,
        config: LaunchConfig,
        args: KernelArgs,
    ) !ExecutionResult {
        const device = self.active_device orelse return error.NoActiveDevice;

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
                self.stats.host_to_device_transfers += 1;
            }
        }

        // Try dispatcher-based execution if we have a compiled handle
        if (self.dispatcher) |*disp| {
            if (kernel.handle != null) {
                // Build a CompiledKernelHandle from the CompiledKernel
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

        // Mark output buffers as device-dirty after execution
        if (gpu_executed) {
            for (args.buffers) |buf| {
                buf.markDeviceDirty();
            }
        }

        const elapsed = timer.read();
        self.stats.kernels_launched += 1;
        self.stats.total_execution_time_ns += elapsed;

        // Record metrics if profiling enabled
        if (self.metrics) |*m| {
            m.recordKernel(kernel.name, elapsed) catch |err| {
                std.log.debug("Failed to record {s} metrics: {t}", .{ kernel.name, err });
            };
        }

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

    // ========================================================================
    // Synchronization
    // ========================================================================

    /// Synchronize all pending operations.
    pub fn synchronize(self: *Gpu) !void {
        try self.stream_manager.synchronizeAll();
    }

    /// Create a new stream.
    pub fn createStream(self: *Gpu, options: StreamOptions) !*Stream {
        const device = self.active_device orelse return error.NoActiveDevice;
        return self.stream_manager.createStream(device, options);
    }

    /// Create a new event.
    pub fn createEvent(self: *Gpu, options: EventOptions) !*Event {
        const device = self.active_device orelse return error.NoActiveDevice;
        return self.stream_manager.createEvent(device, options);
    }

    // ========================================================================
    // Diagnostics
    // ========================================================================

    /// Get GPU statistics.
    pub fn getStats(self: *const Gpu) GpuStats {
        return self.stats;
    }

    /// Get memory information.
    pub fn getMemoryInfo(self: *Gpu) MemoryInfo {
        var total: u64 = 0;
        var used: u64 = 0;

        if (self.active_device) |device| {
            total = device.total_memory orelse 0;
        }

        // Sum up allocated buffer sizes
        self.buffer_mutex.lock();
        defer self.buffer_mutex.unlock();
        for (self.buffers.items) |buf| {
            used += buf.getSize();
        }

        return .{
            .total_bytes = total,
            .used_bytes = used,
            .free_bytes = if (total > used) total - used else 0,
            .peak_used_bytes = self.stats.bytes_allocated,
        };
    }

    /// Check GPU health.
    pub fn checkHealth(self: *const Gpu) HealthStatus {
        if (self.active_device == null) {
            return .unhealthy;
        }

        if (!self.device_manager.hasDevices()) {
            return .unhealthy;
        }

        return .healthy;
    }

    /// Check if GPU is available.
    pub fn isAvailable(self: *const Gpu) bool {
        return self.active_device != null;
    }

    /// Get the active backend.
    pub fn getBackend(self: *const Gpu) ?Backend {
        if (self.active_device) |device| {
            return device.backend;
        }
        return null;
    }

    /// Get the kernel dispatcher (for advanced usage).
    pub fn getDispatcher(self: *Gpu) ?*KernelDispatcher {
        if (self.dispatcher) |*d| {
            return d;
        }
        return null;
    }

    /// Get dispatcher statistics.
    pub fn getDispatcherStats(self: *const Gpu) ?struct {
        kernels_compiled: u64,
        kernels_executed: u64,
        cache_hits: u64,
        cache_misses: u64,
        cache_hit_rate: f64,
    } {
        if (self.dispatcher) |*d| {
            return d.getStats();
        }
        return null;
    }

    // ========================================================================
    // Profiling
    // ========================================================================

    /// Check if profiling is enabled.
    pub fn isProfilingEnabled(self: *const Gpu) bool {
        return self.metrics != null;
    }

    /// Enable profiling (creates metrics collector if not exists).
    pub fn enableProfiling(self: *Gpu) void {
        if (self.metrics == null) {
            self.metrics = MetricsCollector.init(self.allocator);
        }
    }

    /// Disable profiling.
    pub fn disableProfiling(self: *Gpu) void {
        if (self.metrics) |*m| {
            m.deinit();
            self.metrics = null;
        }
    }

    /// Get metrics summary (if profiling enabled).
    pub fn getMetricsSummary(self: *Gpu) ?MetricsSummary {
        if (self.metrics) |*m| {
            return m.getSummary();
        }
        return null;
    }

    /// Get kernel-specific metrics (if profiling enabled).
    pub fn getKernelMetrics(self: *Gpu, name: []const u8) ?KernelMetrics {
        if (self.metrics) |*m| {
            return m.getKernelMetrics(name);
        }
        return null;
    }

    /// Get the metrics collector directly (for advanced usage).
    pub fn getMetricsCollector(self: *Gpu) ?*MetricsCollector {
        if (self.metrics) |*m| {
            return m;
        }
        return null;
    }

    /// Reset all profiling metrics.
    pub fn resetMetrics(self: *Gpu) void {
        if (self.metrics) |*m| {
            m.reset();
        }
    }

    // ========================================================================
    // Multi-GPU Helpers
    // ========================================================================

    /// Check if multi-GPU is enabled.
    pub fn isMultiGpuEnabled(self: *const Gpu) bool {
        return self.device_group != null;
    }

    /// Get multi-GPU statistics (if enabled).
    pub fn getMultiGpuStats(self: *const Gpu) ?multi_device.GroupStats {
        if (self.device_group) |*dg| {
            return dg.getStats();
        }
        return null;
    }

    /// Get the number of active devices.
    pub fn activeDeviceCount(self: *const Gpu) usize {
        if (self.device_group) |*dg| {
            return dg.activeDeviceCount();
        }
        return if (self.active_device != null) 1 else 0;
    }
};

// ============================================================================
// GpuDevice — Simplified Ergonomic Wrapper
// ============================================================================

/// A simplified, ergonomic wrapper around the full `Gpu` API.
///
/// `GpuDevice` provides a thin convenience layer that owns its `Gpu` instance
/// and exposes the most common operations. It is *not* a replacement for `Gpu`;
/// advanced use-cases (multi-GPU, custom streams, profiling) should use `Gpu`
/// directly via the `gpu` field.
///
/// ## Quick Start
///
/// ```zig
/// var dev = try GpuDevice.init(allocator, .{});
/// defer dev.deinit();
///
/// var a = try dev.createBuffer(f32, 4, .{});
/// var b = try dev.createBuffer(f32, 4, .{});
/// var out = try dev.createBuffer(f32, 4, .{});
/// defer { dev.gpu.destroyBuffer(a); dev.gpu.destroyBuffer(b); dev.gpu.destroyBuffer(out); }
///
/// _ = try dev.vectorAdd(a, b, out);
/// ```
pub const GpuDevice = struct {
    gpu: Gpu,

    /// Initialize a new `GpuDevice` with the given configuration.
    pub fn init(allocator: std.mem.Allocator, config: GpuConfig) !GpuDevice {
        return .{ .gpu = try Gpu.init(allocator, config) };
    }

    /// Release all resources.
    pub fn deinit(self: *GpuDevice) void {
        self.gpu.deinit();
    }

    /// Get the active backend name (e.g. "cuda", "metal", "simulated").
    pub fn backendName(self: *const GpuDevice) []const u8 {
        if (self.gpu.active_device) |dev| {
            return @tagName(dev.backend);
        }
        return "none";
    }

    /// Query device capabilities via the backend VTable.
    ///
    /// Returns a `DeviceCaps` populated by the backend, or a zeroed default
    /// if the backend is unavailable.
    pub fn capabilities(self: *const GpuDevice) DeviceCaps {
        if (self.gpu.active_device) |dev| {
            const name_bytes = dev.name;
            var caps = DeviceCaps{
                .total_memory = if (dev.total_memory) |m| @intCast(m) else 0,
                .max_threads_per_block = dev.maxWorkgroupSize(),
                .max_shared_memory = dev.maxSharedMemory(),
                .supports_fp16 = dev.capability.supports_fp16,
                .unified_memory = dev.capability.unified_memory,
            };
            const len = @min(name_bytes.len, caps.name.len);
            @memcpy(caps.name[0..len], name_bytes[0..len]);
            caps.name_len = len;
            return caps;
        }
        return DeviceCaps{};
    }

    /// Create a typed buffer (size in elements, not bytes).
    pub fn createBuffer(self: *GpuDevice, comptime T: type, count: usize, opts: BufferOptions) !*Buffer {
        return self.gpu.createBuffer(count * @sizeOf(T), opts);
    }

    /// Create a buffer from a typed slice.
    pub fn createBufferFromSlice(self: *GpuDevice, comptime T: type, data: []const T, opts: BufferOptions) !*Buffer {
        return self.gpu.createBufferFromSlice(T, data, opts);
    }

    /// Destroy a buffer.
    pub fn destroyBuffer(self: *GpuDevice, buffer: *Buffer) void {
        self.gpu.destroyBuffer(buffer);
    }

    /// Vector addition: out = a + b.
    pub fn vectorAdd(self: *GpuDevice, a: *Buffer, b: *Buffer, out: *Buffer) !ExecutionResult {
        return self.gpu.vectorAdd(a, b, out);
    }

    /// Matrix multiplication: out = a * b.
    pub fn matrixMultiply(self: *GpuDevice, a: *Buffer, b: *Buffer, out: *Buffer, dims: MatrixDims) !ExecutionResult {
        return self.gpu.matrixMultiply(a, b, out, dims);
    }

    /// Compile and run a custom kernel from IR.
    pub fn compileAndRun(
        self: *GpuDevice,
        ir: *const KernelIR,
        config: LaunchConfig,
        args: KernelArgs,
    ) !ExecutionResult {
        var compiled = try self.gpu.compileKernelIR(ir);
        defer compiled.deinit();

        return self.gpu.launchKernel(&compiled, config, args);
    }

    /// Query memory information for the active device.
    pub fn memoryInfo(self: *GpuDevice) MemoryInfo {
        return self.gpu.getMemoryInfo();
    }

    /// Query accumulated GPU statistics.
    pub fn stats(self: *const GpuDevice) GpuStats {
        return self.gpu.getStats();
    }

    /// Synchronize all pending GPU operations.
    pub fn sync(self: *GpuDevice) !void {
        return self.gpu.synchronize();
    }

    /// Check whether a usable device is available.
    pub fn isAvailable(self: *const GpuDevice) bool {
        return self.gpu.isAvailable();
    }

    /// Get health status.
    pub fn checkHealth(self: *const GpuDevice) HealthStatus {
        return self.gpu.checkHealth();
    }
};

// ============================================================================
// Tests
// ============================================================================

test "Gpu init and deinit" {
    var gpu = try Gpu.init(std.testing.allocator, .{});
    defer gpu.deinit();

    // Should at least not crash
    _ = gpu.listDevices();
    _ = gpu.getStats();
    _ = gpu.checkHealth();
}

test "Gpu buffer creation" {
    var gpu = try Gpu.init(std.testing.allocator, .{});
    defer gpu.deinit();

    if (!gpu.isAvailable()) {
        return; // Skip if no GPU
    }

    var buffer = try gpu.createBuffer(1024, .{});
    defer gpu.destroyBuffer(buffer);

    try std.testing.expect(buffer.getSize() == 1024);
    try std.testing.expect(gpu.getStats().buffers_created == 1);
}

test "Gpu createBufferFromSlice" {
    var gpu = try Gpu.init(std.testing.allocator, .{});
    defer gpu.deinit();

    if (!gpu.isAvailable()) {
        return; // Skip if no GPU
    }

    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var buffer = try gpu.createBufferFromSlice(f32, &data, .{ .mode = .explicit });
    defer gpu.destroyBuffer(buffer);

    var output: [4]f32 = undefined;
    try buffer.read(f32, &output);

    try std.testing.expectEqualSlices(f32, &data, &output);
}

test "Gpu vectorAdd" {
    var gpu = try Gpu.init(std.testing.allocator, .{});
    defer gpu.deinit();

    if (!gpu.isAvailable()) {
        return; // Skip if no GPU
    }

    const a_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b_data = [_]f32{ 5.0, 6.0, 7.0, 8.0 };

    const a = try gpu.createBufferFromSlice(f32, &a_data, .{ .mode = .explicit });
    defer gpu.destroyBuffer(a);

    const b = try gpu.createBufferFromSlice(f32, &b_data, .{ .mode = .explicit });
    defer gpu.destroyBuffer(b);

    const result = try gpu.createBuffer(4 * @sizeOf(f32), .{ .mode = .explicit });
    defer gpu.destroyBuffer(result);

    const exec_result = try gpu.vectorAdd(a, b, result);
    try std.testing.expect(exec_result.elements_processed == 4);

    var output: [4]f32 = undefined;
    try result.read(f32, &output);

    try std.testing.expectApproxEqAbs(@as(f32, 6.0), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), output[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), output[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), output[3], 0.001);
}

test "Gpu reduceSum" {
    var gpu = try Gpu.init(std.testing.allocator, .{});
    defer gpu.deinit();

    if (!gpu.isAvailable()) {
        return; // Skip if no GPU
    }

    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const buffer = try gpu.createBufferFromSlice(f32, &data, .{ .mode = .explicit });
    defer gpu.destroyBuffer(buffer);

    const result = try gpu.reduceSum(buffer);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), result.value, 0.001);
}

test "Gpu dotProduct" {
    var gpu = try Gpu.init(std.testing.allocator, .{});
    defer gpu.deinit();

    if (!gpu.isAvailable()) {
        return; // Skip if no GPU
    }

    const a_data = [_]f32{ 1.0, 2.0, 3.0 };
    const b_data = [_]f32{ 4.0, 5.0, 6.0 };

    const a = try gpu.createBufferFromSlice(f32, &a_data, .{ .mode = .explicit });
    defer gpu.destroyBuffer(a);

    const b = try gpu.createBufferFromSlice(f32, &b_data, .{ .mode = .explicit });
    defer gpu.destroyBuffer(b);

    const result = try gpu.dotProduct(a, b);
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    try std.testing.expectApproxEqAbs(@as(f32, 32.0), result.value, 0.001);
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
