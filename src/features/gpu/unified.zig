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
const time = @import("../../services/shared/mod.zig").time;
const sync = @import("../../services/shared/mod.zig").sync;
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

// Submodule imports
const dev_mgr = @import("device_manager.zig");
const buf_pool = @import("buffer_pool.zig");
const stream_orch = @import("stream_orchestrator.zig");

const Mutex = sync.Mutex;

// Re-export key types (preserving full public API)
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

// Re-export types from submodules
pub const LoadBalanceStrategy = dev_mgr.LoadBalanceStrategy;
pub const MultiGpuConfig = dev_mgr.MultiGpuConfig;
pub const HealthStatus = dev_mgr.HealthStatus;
pub const MemoryInfo = buf_pool.MemoryInfo;
pub const MatrixDims = stream_orch.MatrixDims;
pub const CompiledKernel = stream_orch.CompiledKernel;
pub const GpuStats = stream_orch.GpuStats;
pub const GpuConfig = stream_orch.GpuConfig;

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
                .strategy = dev_mgr.toMultiDeviceStrategy(effective_config.load_balance_strategy),
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

        // Discover devices and select initial
        const init_result = dev_mgr.initDevices(
            allocator,
            &device_manager,
            &stream_manager,
            effective_config,
        );
        errdefer if (init_result.dispatcher) |*d| {
            var dd = d.*;
            dd.deinit();
        };

        return .{
            .allocator = allocator,
            .config = effective_config,
            .device_manager = device_manager,
            .stream_manager = stream_manager,
            .dispatcher = init_result.dispatcher,
            .device_group = device_group,
            .metrics = metrics,
            .active_device = init_result.active_device,
            .default_stream = init_result.default_stream,
            .buffers = .empty,
            .buffer_mutex = .{},
            .stats = std.mem.zeroes(GpuStats),
        };
    }

    /// Deinitialize and cleanup.
    pub fn deinit(self: *Gpu) void {
        // Destroy all buffers
        buf_pool.destroyAllBuffers(self.allocator, &self.buffers, &self.buffer_mutex);

        // Clean up dispatcher
        if (self.dispatcher) |*d| d.deinit();

        // Clean up metrics
        if (self.metrics) |*m| m.deinit();

        // Clean up device group
        if (self.device_group) |*dg| dg.deinit();

        self.stream_manager.deinit();
        self.device_manager.deinit();
        self.* = undefined;
    }

    // ========================================================================
    // Device Management (delegates to device_manager.zig)
    // ========================================================================

    /// Select a device based on criteria.
    pub fn selectDevice(self: *Gpu, selector: DeviceSelector) !void {
        return dev_mgr.selectDevice(
            &self.device_manager,
            &self.stream_manager,
            &self.active_device,
            &self.default_stream,
            selector,
        );
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
        return dev_mgr.enableMultiGpu(self.allocator, &self.device_group, config);
    }

    /// Get multi-GPU device group (if enabled).
    pub fn getDeviceGroup(self: *Gpu) ?*DeviceGroup {
        if (self.device_group) |*dg| return dg;
        return null;
    }

    /// Distribute work across multiple GPUs.
    pub fn distributeWork(self: *Gpu, total_work: usize) ![]WorkDistribution {
        return dev_mgr.distributeWork(self.allocator, &self.device_group, self.active_device, total_work);
    }

    // ========================================================================
    // Buffer Management (delegates to buffer_pool.zig)
    // ========================================================================

    /// Create a new buffer.
    pub fn createBuffer(self: *Gpu, size: usize, options: BufferOptions) !*Buffer {
        const device = self.active_device orelse return error.NoActiveDevice;
        const buffer = try buf_pool.createBuffer(
            self.allocator,
            &self.buffers,
            &self.buffer_mutex,
            device,
            size,
            options,
            self.config.memory_mode,
        );
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
        const buffer = try buf_pool.createBufferFromSlice(
            self.allocator,
            &self.buffers,
            &self.buffer_mutex,
            device,
            T,
            data,
            options,
            self.config.memory_mode,
        );
        self.stats.buffers_created += 1;
        self.stats.bytes_allocated += data.len * @sizeOf(T);
        return buffer;
    }

    /// Destroy a buffer.
    pub fn destroyBuffer(self: *Gpu, buffer: *Buffer) void {
        buf_pool.destroyBuffer(self.allocator, &self.buffers, &self.buffer_mutex, buffer);
    }

    // ========================================================================
    // High-Level Operations (delegates to stream_orchestrator.zig)
    // ========================================================================

    /// Vector addition: result = a + b
    pub fn vectorAdd(self: *Gpu, a: *Buffer, b: *Buffer, result: *Buffer) !ExecutionResult {
        return stream_orch.vectorAdd(
            &self.dispatcher,
            self.active_device,
            &self.metrics,
            &self.stats,
            a,
            b,
            result,
        );
    }

    /// Matrix multiplication: result = a * b
    pub fn matrixMultiply(
        self: *Gpu,
        a: *Buffer,
        b: *Buffer,
        result: *Buffer,
        dims: MatrixDims,
    ) !ExecutionResult {
        return stream_orch.matrixMultiply(
            &self.dispatcher,
            self.active_device,
            &self.metrics,
            &self.stats,
            a,
            b,
            result,
            dims,
        );
    }

    /// Reduce sum: returns sum of all elements.
    pub fn reduceSum(self: *Gpu, input: *Buffer) !stream_orch.ReduceResult {
        return stream_orch.reduceSum(
            &self.dispatcher,
            self.active_device,
            &self.metrics,
            &self.stats,
            .{
                .allocator = self.allocator,
                .buffers = &self.buffers,
                .buffer_mutex = &self.buffer_mutex,
                .active_device = self.active_device,
                .memory_mode = self.config.memory_mode,
                .stats = &self.stats,
            },
            input,
        );
    }

    /// Dot product: returns a · b
    pub fn dotProduct(self: *Gpu, a: *Buffer, b: *Buffer) !stream_orch.ReduceResult {
        return stream_orch.dotProduct(
            &self.dispatcher,
            self.active_device,
            &self.metrics,
            &self.stats,
            .{
                .allocator = self.allocator,
                .buffers = &self.buffers,
                .buffer_mutex = &self.buffer_mutex,
                .active_device = self.active_device,
                .memory_mode = self.config.memory_mode,
                .stats = &self.stats,
            },
            a,
            b,
        );
    }

    /// Softmax: output = softmax(input)
    pub fn softmax(self: *Gpu, input: *Buffer, output: *Buffer) !ExecutionResult {
        return stream_orch.softmax(
            &self.dispatcher,
            self.active_device,
            &self.metrics,
            &self.stats,
            input,
            output,
        );
    }

    // ========================================================================
    // Custom Kernel Support (delegates to stream_orchestrator.zig)
    // ========================================================================

    /// Compile a kernel from portable source.
    pub fn compileKernel(self: *Gpu, source: PortableKernelSource) !CompiledKernel {
        return stream_orch.compileKernel(self.allocator, &self.dispatcher, self.active_device, source);
    }

    /// Compile kernel IR directly (convenience overload).
    pub fn compileKernelIR(self: *Gpu, ir: *const KernelIR) !CompiledKernel {
        return stream_orch.compileKernelIR(self.allocator, &self.dispatcher, self.active_device, ir);
    }

    /// Launch a compiled kernel.
    pub fn launchKernel(
        self: *Gpu,
        kernel: *const CompiledKernel,
        config: LaunchConfig,
        args: KernelArgs,
    ) !ExecutionResult {
        return stream_orch.launchKernel(
            &self.dispatcher,
            self.active_device,
            &self.metrics,
            &self.stats,
            kernel,
            config,
            args,
        );
    }

    // ========================================================================
    // Synchronization (delegates to stream_orchestrator.zig)
    // ========================================================================

    /// Synchronize all pending operations.
    pub fn synchronize(self: *Gpu) !void {
        try stream_orch.synchronize(&self.stream_manager);
    }

    /// Create a new stream.
    pub fn createStream(self: *Gpu, options: StreamOptions) !*Stream {
        return stream_orch.createStream(&self.stream_manager, self.active_device, options);
    }

    /// Create a new event.
    pub fn createEvent(self: *Gpu, options: EventOptions) !*Event {
        return stream_orch.createEvent(&self.stream_manager, self.active_device, options);
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
        return buf_pool.getMemoryInfo(
            self.active_device,
            &self.buffers,
            &self.buffer_mutex,
            self.stats.bytes_allocated,
        );
    }

    /// Check GPU health.
    pub fn checkHealth(self: *const Gpu) HealthStatus {
        return dev_mgr.checkHealth(self.active_device, &self.device_manager);
    }

    /// Check if GPU is available.
    pub fn isAvailable(self: *const Gpu) bool {
        return self.active_device != null;
    }

    /// Get the active backend.
    pub fn getBackend(self: *const Gpu) ?Backend {
        if (self.active_device) |device| return device.backend;
        return null;
    }

    /// Get the kernel dispatcher (for advanced usage).
    pub fn getDispatcher(self: *Gpu) ?*KernelDispatcher {
        if (self.dispatcher) |*d| return d;
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
        if (self.dispatcher) |*d| return d.getStats();
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
        if (self.metrics) |*m| return m.getSummary();
        return null;
    }

    /// Get kernel-specific metrics (if profiling enabled).
    pub fn getKernelMetrics(self: *Gpu, name: []const u8) ?KernelMetrics {
        if (self.metrics) |*m| return m.getKernelMetrics(name);
        return null;
    }

    /// Get the metrics collector directly (for advanced usage).
    pub fn getMetricsCollector(self: *Gpu) ?*MetricsCollector {
        if (self.metrics) |*m| return m;
        return null;
    }

    /// Reset all profiling metrics.
    pub fn resetMetrics(self: *Gpu) void {
        if (self.metrics) |*m| m.reset();
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
        if (self.device_group) |*dg| return dg.getStats();
        return null;
    }

    /// Get the number of active devices.
    pub fn activeDeviceCount(self: *const Gpu) usize {
        if (self.device_group) |*dg| return dg.activeDeviceCount();
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

    var output: [4]f32 = [_]f32{0} ** 4;
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

    var output: [4]f32 = [_]f32{0} ** 4;
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

test {
    std.testing.refAllDecls(@This());
}
