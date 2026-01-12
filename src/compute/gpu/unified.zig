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
const backend_mod = @import("backend.zig");
const device_mod = @import("device.zig");
const stream_mod = @import("stream.zig");
const buffer_mod = @import("unified_buffer.zig");
const dsl = @import("dsl/mod.zig");
const multi_device = @import("multi_device.zig");
const metrics_mod = @import("metrics.zig");

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

/// Execution result with timing and statistics.
pub const ExecutionResult = struct {
    /// Execution time in nanoseconds.
    execution_time_ns: u64,
    /// Number of elements processed.
    elements_processed: usize,
    /// Bytes transferred.
    bytes_transferred: usize,
    /// Backend used for execution.
    backend: Backend,
    /// Device used for execution.
    device_id: u32,

    /// Get throughput in GB/s.
    pub fn throughputGBps(self: ExecutionResult) f64 {
        if (self.execution_time_ns == 0) return 0;
        const bytes_per_sec = @as(f64, @floatFromInt(self.bytes_transferred)) /
            (@as(f64, @floatFromInt(self.execution_time_ns)) / 1_000_000_000.0);
        return bytes_per_sec / (1024 * 1024 * 1024);
    }

    /// Get elements per second.
    pub fn elementsPerSecond(self: ExecutionResult) f64 {
        if (self.execution_time_ns == 0) return 0;
        return @as(f64, @floatFromInt(self.elements_processed)) /
            (@as(f64, @floatFromInt(self.execution_time_ns)) / 1_000_000_000.0);
    }
};

/// Matrix dimensions for matrix operations.
pub const MatrixDims = struct {
    m: usize,
    n: usize,
    k: usize,
};

/// Kernel launch configuration.
pub const LaunchConfig = struct {
    /// Global work size (total threads).
    global_size: [3]u32 = .{ 1, 1, 1 },
    /// Local work size (workgroup/block size).
    local_size: ?[3]u32 = null,
    /// Stream to launch on (null = default).
    stream: ?*Stream = null,
    /// Shared memory size in bytes.
    shared_memory: u32 = 0,
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

    // Multi-GPU support
    device_group: ?DeviceGroup,

    // Metrics and profiling
    metrics: ?MetricsCollector,

    // Active state
    active_device: ?*const Device,
    default_stream: ?*Stream,

    // Buffer tracking
    buffers: std.ArrayListUnmanaged(*Buffer),
    buffer_mutex: std.Thread.Mutex,

    // Statistics
    stats: GpuStats,

    /// Initialize the unified GPU API.
    pub fn init(allocator: std.mem.Allocator, config: GpuConfig) !Gpu {
        var device_manager = try device_mod.DeviceManager.init(allocator);
        errdefer device_manager.deinit();

        var stream_manager = stream_mod.StreamManager.init(allocator);
        errdefer stream_manager.deinit();

        // Initialize multi-GPU if enabled
        var device_group: ?DeviceGroup = null;
        if (config.multi_gpu) {
            const multi_config = multi_device.MultiDeviceConfig{
                .strategy = switch (config.load_balance_strategy) {
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
        if (config.enable_profiling) {
            metrics = MetricsCollector.init(allocator);
        }
        errdefer if (metrics) |*m| m.deinit();

        // Select initial device
        var active_device: ?*const Device = null;
        var default_stream: ?*Stream = null;

        if (device_manager.hasDevices()) {
            if (config.preferred_backend) |backend| {
                active_device = device_manager.selectDevice(.{ .by_backend = backend }) catch null;
            }

            if (active_device == null and config.allow_fallback) {
                active_device = device_manager.selectBestDevice() catch null;
            }

            // Create default stream for active device
            if (active_device) |device| {
                default_stream = stream_manager.createStream(device, .{}) catch null;
            }
        }

        return .{
            .allocator = allocator,
            .config = config,
            .device_manager = device_manager,
            .stream_manager = stream_manager,
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

        var timer = std.time.Timer.start() catch return error.TimerFailed;

        // In a real implementation, this would dispatch to the appropriate backend
        // For now, simulate with host computation
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

        const elapsed = timer.read();
        self.stats.kernels_launched += 1;
        self.stats.total_execution_time_ns += elapsed;

        // Record metrics if profiling enabled
        if (self.metrics) |*m| {
            m.recordKernel("vectorAdd", elapsed) catch {};
        }

        return ExecutionResult{
            .execution_time_ns = elapsed,
            .elements_processed = a.elementCount(),
            .bytes_transferred = a.getSize() + b.getSize() + result.getSize(),
            .backend = device.backend,
            .device_id = device.id,
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

        var timer = std.time.Timer.start() catch return error.TimerFailed;

        // Simplified host-side matrix multiply for demonstration
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

        const elapsed = timer.read();
        self.stats.kernels_launched += 1;
        self.stats.total_execution_time_ns += elapsed;

        // Record metrics if profiling enabled
        if (self.metrics) |*m| {
            m.recordKernel("matrixMultiply", elapsed) catch {};
        }

        return ExecutionResult{
            .execution_time_ns = elapsed,
            .elements_processed = dims.m * dims.n,
            .bytes_transferred = a.getSize() + b.getSize() + result.getSize(),
            .backend = device.backend,
            .device_id = device.id,
        };
    }

    /// Reduce sum: returns sum of all elements.
    pub fn reduceSum(self: *Gpu, input: *Buffer) !struct { value: f32, stats: ExecutionResult } {
        const device = self.active_device orelse return error.NoActiveDevice;

        var timer = std.time.Timer.start() catch return error.TimerFailed;
        var sum: f32 = 0;

        if (input.host_data) |host| {
            const data = std.mem.bytesAsSlice(f32, host);
            for (data) |v| {
                sum += v;
            }
        }

        const elapsed = timer.read();
        self.stats.kernels_launched += 1;
        self.stats.total_execution_time_ns += elapsed;

        // Record metrics if profiling enabled
        if (self.metrics) |*m| {
            m.recordKernel("reduceSum", elapsed) catch {};
        }

        return .{
            .value = sum,
            .stats = ExecutionResult{
                .execution_time_ns = elapsed,
                .elements_processed = input.elementCount(),
                .bytes_transferred = input.getSize(),
                .backend = device.backend,
                .device_id = device.id,
            },
        };
    }

    /// Dot product: returns a · b
    pub fn dotProduct(self: *Gpu, a: *Buffer, b: *Buffer) !struct { value: f32, stats: ExecutionResult } {
        const device = self.active_device orelse return error.NoActiveDevice;

        var timer = std.time.Timer.start() catch return error.TimerFailed;
        var sum: f32 = 0;

        if (a.host_data != null and b.host_data != null) {
            const a_data = std.mem.bytesAsSlice(f32, a.host_data.?);
            const b_data = std.mem.bytesAsSlice(f32, b.host_data.?);

            const len = @min(a_data.len, b_data.len);
            for (0..len) |i| {
                sum += a_data[i] * b_data[i];
            }
        }

        const elapsed = timer.read();
        self.stats.kernels_launched += 1;
        self.stats.total_execution_time_ns += elapsed;

        // Record metrics if profiling enabled
        if (self.metrics) |*m| {
            m.recordKernel("dotProduct", elapsed) catch {};
        }

        return .{
            .value = sum,
            .stats = ExecutionResult{
                .execution_time_ns = elapsed,
                .elements_processed = a.elementCount(),
                .bytes_transferred = a.getSize() + b.getSize(),
                .backend = device.backend,
                .device_id = device.id,
            },
        };
    }

    /// Softmax: output = softmax(input)
    pub fn softmax(self: *Gpu, input: *Buffer, output: *Buffer) !ExecutionResult {
        const device = self.active_device orelse return error.NoActiveDevice;

        var timer = std.time.Timer.start() catch return error.TimerFailed;

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

        const elapsed = timer.read();
        self.stats.kernels_launched += 1;
        self.stats.total_execution_time_ns += elapsed;

        // Record metrics if profiling enabled
        if (self.metrics) |*m| {
            m.recordKernel("softmax", elapsed) catch {};
        }

        return ExecutionResult{
            .execution_time_ns = elapsed,
            .elements_processed = input.elementCount(),
            .bytes_transferred = input.getSize() + output.getSize(),
            .backend = device.backend,
            .device_id = device.id,
        };
    }

    // ========================================================================
    // Custom Kernel Support
    // ========================================================================

    /// Compile a kernel from portable source.
    pub fn compileKernel(self: *Gpu, source: PortableKernelSource) !CompiledKernel {
        const device = self.active_device orelse return error.NoActiveDevice;

        // Compile the kernel IR to the target backend
        var generated = try dsl.compile(self.allocator, &source.ir, device.backend, .{});

        return CompiledKernel{
            .allocator = self.allocator,
            .name = try self.allocator.dupe(u8, source.ir.name),
            .backend = device.backend,
            .source = generated.code,
            .entry_point = generated.entry_point,
            .handle = null, // Backend would create handle here
        };
    }

    /// Launch a compiled kernel.
    pub fn launchKernel(
        self: *Gpu,
        kernel: *const CompiledKernel,
        config: LaunchConfig,
        args: anytype,
    ) !ExecutionResult {
        const device = self.active_device orelse return error.NoActiveDevice;
        _ = args;

        var timer = std.time.Timer.start() catch return error.TimerFailed;

        // In a real implementation, this would dispatch to the backend
        // For now, just record the launch

        const elapsed = timer.read();
        self.stats.kernels_launched += 1;
        self.stats.total_execution_time_ns += elapsed;

        // Record metrics if profiling enabled
        if (self.metrics) |*m| {
            m.recordKernel(kernel.name, elapsed) catch {};
        }

        const total_threads = @as(usize, config.global_size[0]) *
            @as(usize, config.global_size[1]) *
            @as(usize, config.global_size[2]);

        return ExecutionResult{
            .execution_time_ns = elapsed,
            .elements_processed = total_threads,
            .bytes_transferred = 0,
            .backend = device.backend,
            .device_id = device.id,
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
    };

    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result.throughputGBps(), 0.01);
    try std.testing.expectApproxEqAbs(@as(f64, 1_000_000.0), result.elementsPerSecond(), 1.0);
}
