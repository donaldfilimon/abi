# gpu API Reference
> **Codebase Status:** Synced with repository as of 2026-01-18.

**Source:** `src/gpu/unified.zig`

 Unified GPU API
 Main entry point for the unified GPU API.
 Provides a single interface for all GPU backends with:
 - High-level operations (vectorAdd, matrixMultiply, etc.)
 - Custom kernel compilation and execution
 - Smart buffer management
 - Device discovery and selection
 - Stream/event synchronization
 ## Quick Start
 ```zig
 var gpu = try Gpu.init(allocator, .{});
 defer gpu.deinit();
 // Create buffers
 var a = try gpu.createBufferFromSlice(f32, &[_]f32{ 1, 2, 3, 4 }, .{});
 var b = try gpu.createBufferFromSlice(f32, &[_]f32{ 5, 6, 7, 8 }, .{});
 var result = try gpu.createBuffer(4 * @sizeOf(f32), .{});
 defer { gpu.destroyBuffer(&a); gpu.destroyBuffer(&b); gpu.destroyBuffer(&result); }
 // Run operation
 _ = try gpu.vectorAdd(&a, &b, &result);
 // Read results
 var output: [4]f32 = undefined;
 try result.read(f32, &output);
 ```
### `pub const LoadBalanceStrategy`

 Load balance strategy for multi-GPU.

### `pub const GpuConfig`

 GPU configuration.

### `pub const ExecutionResult`

 Execution result with timing and statistics.

### `pub fn throughputGBps(self: ExecutionResult) f64`

 Get throughput in GB/s.

### `pub fn elementsPerSecond(self: ExecutionResult) f64`

 Get elements per second.

### `pub const MatrixDims`

 Matrix dimensions for matrix operations.

### `pub const LaunchConfig`

 Kernel launch configuration.

### `pub const CompiledKernel`

 Compiled kernel handle.

### `pub const MemoryInfo`

 GPU memory information.

### `pub const GpuStats`

 GPU statistics.

### `pub const HealthStatus`

 Health status.

### `pub const MultiGpuConfig`

 Multi-GPU configuration.

### `pub const Gpu`

 Main unified GPU API.

### `pub fn init(allocator: std.mem.Allocator, config: GpuConfig) !Gpu`

 Initialize the unified GPU API.

### `pub fn deinit(self: *Gpu) void`

 Deinitialize and cleanup.

### `pub fn selectDevice(self: *Gpu, selector: DeviceSelector) !void`

 Select a device based on criteria.

### `pub fn getActiveDevice(self: *const Gpu) ?*const Device`

 Get the currently active device.

### `pub fn listDevices(self: *const Gpu) []const Device`

 List all available devices.

### `pub fn enableMultiGpu(self: *Gpu, config: MultiGpuConfig) !void`

 Enable multi-GPU mode.

### `pub fn getDeviceGroup(self: *Gpu) ?*DeviceGroup`

 Get multi-GPU device group (if enabled).

### `pub fn distributeWork(self: *Gpu, total_work: usize) ![]WorkDistribution`

 Distribute work across multiple GPUs.

### `pub fn createBuffer(self: *Gpu, size: usize, options: BufferOptions) !*Buffer`

 Create a new buffer.

### `pub fn createBufferFromSlice(`

 Create a buffer from a typed slice.

### `pub fn destroyBuffer(self: *Gpu, buffer: *Buffer) void`

 Destroy a buffer.

### `pub fn vectorAdd(self: *Gpu, a: *Buffer, b: *Buffer, result: *Buffer) !ExecutionResult`

 Vector addition: result = a + b

### `pub fn matrixMultiply(`

 Matrix multiplication: result = a * b

### `pub fn reduceSum(self: *Gpu, input: *Buffer) !struct`

 Reduce sum: returns sum of all elements.

### `pub fn dotProduct(self: *Gpu, a: *Buffer, b: *Buffer) !struct`

 Dot product: returns a · b

### `pub fn softmax(self: *Gpu, input: *Buffer, output: *Buffer) !ExecutionResult`

 Softmax: output = softmax(input)

### `pub fn compileKernel(self: *Gpu, source: PortableKernelSource) !CompiledKernel`

 Compile a kernel from portable source.

### `pub fn launchKernel(`

 Launch a compiled kernel.

### `pub fn synchronize(self: *Gpu) !void`

 Synchronize all pending operations.

### `pub fn createStream(self: *Gpu, options: StreamOptions) !*Stream`

 Create a new stream.

### `pub fn createEvent(self: *Gpu, options: EventOptions) !*Event`

 Create a new event.

### `pub fn getStats(self: *const Gpu) GpuStats`

 Get GPU statistics.

### `pub fn getMemoryInfo(self: *Gpu) MemoryInfo`

 Get memory information.

### `pub fn checkHealth(self: *const Gpu) HealthStatus`

 Check GPU health.

### `pub fn isAvailable(self: *const Gpu) bool`

 Check if GPU is available.

### `pub fn getBackend(self: *const Gpu) ?Backend`

 Get the active backend.

### `pub fn getDispatcher(self: *Gpu) ?*KernelDispatcher`

 Get the kernel dispatcher (for advanced usage).

### `pub fn getDispatcherStats(self: *const Gpu) ?struct`

 Get dispatcher statistics.

### `pub fn isProfilingEnabled(self: *const Gpu) bool`

 Check if profiling is enabled.

### `pub fn enableProfiling(self: *Gpu) void`

 Enable profiling (creates metrics collector if not exists).

### `pub fn disableProfiling(self: *Gpu) void`

 Disable profiling.

### `pub fn getMetricsSummary(self: *Gpu) ?MetricsSummary`

 Get metrics summary (if profiling enabled).

### `pub fn getKernelMetrics(self: *Gpu, name: []const u8) ?KernelMetrics`

 Get kernel-specific metrics (if profiling enabled).

### `pub fn getMetricsCollector(self: *Gpu) ?*MetricsCollector`

 Get the metrics collector directly (for advanced usage).

### `pub fn resetMetrics(self: *Gpu) void`

 Reset all profiling metrics.

### `pub fn isMultiGpuEnabled(self: *const Gpu) bool`

 Check if multi-GPU is enabled.

### `pub fn getMultiGpuStats(self: *const Gpu) ?multi_device.GroupStats`

 Get multi-GPU statistics (if enabled).

### `pub fn activeDeviceCount(self: *const Gpu) usize`

 Get the number of active devices.

