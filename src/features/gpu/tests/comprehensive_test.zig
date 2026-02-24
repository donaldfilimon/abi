//! Comprehensive GPU Testing Suite
//!
//! This module provides extensive testing capabilities for the GPU framework,
//! including performance benchmarks, stress tests, and correctness verification.

const std = @import("std");
const time = @import("../../../services/shared/time.zig");
const sync = @import("../../../services/shared/sync.zig");
const gpu = @import("../mod.zig");
const unified = @import("../unified.zig");
const device_mod = @import("../device.zig");
const profiling = @import("../profiling.zig");

/// Test configuration for comprehensive testing
pub const TestConfig = struct {
    /// Number of iterations for performance tests
    iterations: usize = 100,
    /// Size of test data
    data_size: usize = 1024 * 1024, // 1M elements
    /// Enable performance benchmarking
    benchmark: bool = true,
    /// Enable stress testing
    stress_test: bool = false,
    /// Stress test duration in seconds
    stress_duration_seconds: usize = 30,
    /// Memory pressure test (fraction of available memory)
    memory_pressure: f64 = 0.8,
    /// Enable multi-device testing
    multi_device: bool = true,
};

/// Performance benchmark results
pub const BenchmarkResult = struct {
    operation: []const u8,
    backend: gpu.Backend,
    data_size: usize,
    iterations: usize,
    total_time_ns: u64,
    avg_time_ns: u64,
    throughput_elements_per_sec: f64,
    memory_bandwidth_gb_per_sec: f64,
    peak_memory_usage: usize,
};

/// Comprehensive test suite runner
pub const TestSuite = struct {
    allocator: std.mem.Allocator,
    config: TestConfig,
    results: std.ArrayListUnmanaged(BenchmarkResult),

    pub fn init(allocator: std.mem.Allocator, config: TestConfig) TestSuite {
        return .{
            .allocator = allocator,
            .config = config,
            .results = .empty,
        };
    }

    pub fn deinit(self: *TestSuite) void {
        self.results.deinit(self.allocator);
        self.* = undefined;
    }

    /// Run all tests in the suite
    pub fn runAll(self: *TestSuite) !void {
        try self.runCorrectnessTests();
        if (self.config.benchmark) {
            try self.runPerformanceBenchmarks();
        }
        if (self.config.stress_test) {
            try self.runStressTests();
        }
        if (self.config.multi_device) {
            try self.runMultiDeviceTests();
        }
    }

    /// Run correctness verification tests
    pub fn runCorrectnessTests(self: *TestSuite) !void {
        std.log.info("Running GPU correctness tests...", .{});

        // Test basic vector operations
        try self.testVectorOperations();
        try self.testMatrixOperations();
        try self.testMemoryOperations();

        // Test kernel DSL compilation
        try self.testKernelCompilation();

        std.log.info("Correctness tests passed!", .{});
    }

    /// Run performance benchmarks
    pub fn runPerformanceBenchmarks(self: *TestSuite) !void {
        std.log.info("Running GPU performance benchmarks...", .{});

        try self.benchmarkVectorAdd();
        try self.benchmarkMatrixMultiply();
        try self.benchmarkMemoryBandwidth();

        std.log.info("Performance benchmarks completed!", .{});
    }

    /// Run stress tests
    pub fn runStressTests(self: *TestSuite) !void {
        std.log.info("Running GPU stress tests...", .{});

        try self.stressMemoryPressure();
        try self.stressConcurrentOperations();

        std.log.info("Stress tests completed!", .{});
    }

    /// Run multi-device tests
    pub fn runMultiDeviceTests(self: *TestSuite) !void {
        std.log.info("Running multi-device tests...", .{});

        const devices = try device_mod.discoverDevices(self.allocator);
        defer self.allocator.free(devices);

        if (devices.len > 1) {
            try self.testDevicePeerTransfer(devices);
            try self.testLoadBalancing(devices);
        } else {
            std.log.info("Skipping multi-device tests (only {} devices available)", .{devices.len});
        }
    }

    fn testVectorOperations(self: *TestSuite) !void {
        var g = try gpu.Gpu.init(self.allocator, .{});
        defer g.deinit();

        // Create test data
        const size = 1024;
        const a_data = try self.allocator.alloc(f32, size);
        defer self.allocator.free(a_data);
        const b_data = try self.allocator.alloc(f32, size);
        defer self.allocator.free(b_data);
        const result_data = try self.allocator.alloc(f32, size);
        defer self.allocator.free(result_data);

        // Initialize with known values
        for (a_data, 0..) |*val, i| val.* = @floatFromInt(i);
        for (b_data, 0..) |*val, i| val.* = @floatFromInt(i * 2);

        // Create buffers
        var a_buf = try g.createBufferFromSlice(f32, a_data, .{});
        defer g.destroyBuffer(&a_buf);
        var b_buf = try g.createBufferFromSlice(f32, b_data, .{});
        defer g.destroyBuffer(&b_buf);
        var result_buf = try g.createBuffer(f32, size, .{});
        defer g.destroyBuffer(&result_buf);

        // Perform vector addition
        _ = try g.vectorAdd(&a_buf, &b_buf, &result_buf);

        // Verify results
        try result_buf.read(f32, result_data);
        for (result_data, 0..) |val, i| {
            const expected = @as(f32, @floatFromInt(i + i * 2));
            try std.testing.expectApproxEqAbs(expected, val, 1e-6);
        }
    }

    fn testMatrixOperations(self: *TestSuite) !void {
        var g = try gpu.Gpu.init(self.allocator, .{});
        defer g.deinit();

        // 4x4 matrices for simplicity
        const dim = 4;
        const a_data = try self.allocator.alloc(f32, dim * dim);
        defer self.allocator.free(a_data);
        const b_data = try self.allocator.alloc(f32, dim * dim);
        defer self.allocator.free(b_data);
        const result_data = try self.allocator.alloc(f32, dim * dim);
        defer self.allocator.free(result_data);

        // Initialize identity matrices
        @memset(a_data, 0);
        @memset(b_data, 0);
        for (0..dim) |i| {
            a_data[i * dim + i] = 1.0;
            b_data[i * dim + i] = 1.0;
        }

        var a_buf = try g.createBufferFromSlice(f32, a_data, .{});
        defer g.destroyBuffer(&a_buf);
        var b_buf = try g.createBufferFromSlice(f32, b_data, .{});
        defer g.destroyBuffer(&b_buf);
        var result_buf = try g.createBuffer(f32, dim * dim, .{});
        defer g.destroyBuffer(&result_buf);

        const dims = gpu.unified.MatrixDims{ .m = dim, .n = dim, .k = dim };
        _ = try g.matrixMultiply(&a_buf, &b_buf, &result_buf, dims);

        try result_buf.read(f32, result_data);
        // Identity * Identity = Identity
        for (result_data, 0..) |val, i| {
            const expected: f32 = if (i % (dim + 1) == 0) 1.0 else 0.0;
            try std.testing.expectApproxEqAbs(expected, val, 1e-6);
        }
    }

    fn testMemoryOperations(self: *TestSuite) !void {
        var g = try gpu.Gpu.init(self.allocator, .{});
        defer g.deinit();

        const size = 1024;
        var buffer = try g.createBuffer(f32, size, .{});
        defer g.destroyBuffer(&buffer);

        // Test write and read
        const test_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
        try buffer.write(f32, &test_data);

        var read_data: [4]f32 = undefined;
        try buffer.read(f32, &read_data);

        for (test_data, read_data) |expected, actual| {
            try std.testing.expectApproxEqAbs(expected, actual, 1e-6);
        }
    }

    fn testKernelCompilation(self: *TestSuite) !void {
        const kernel_src = "kernel void test_kernel(global float* output, uint n) { for(uint i = 0; i < n; i++) output[i] = i * 2.0f; }";
        const compiled = try gpu.compileKernel(self.allocator, kernel_src, .opencl, "test_kernel", .{});
        defer self.allocator.free(compiled);
        try std.testing.expect(compiled.len > 0);
    }

    fn benchmarkVectorAdd(self: *TestSuite) !void {
        var g = try gpu.Gpu.init(self.allocator, .{});
        defer g.deinit();

        const size = self.config.data_size;
        const a_data = try self.allocator.alloc(f32, size);
        defer self.allocator.free(a_data);
        const b_data = try self.allocator.alloc(f32, size);
        defer self.allocator.free(b_data);

        for (a_data, 0..) |*val, i| val.* = @floatFromInt(i % 100);
        for (b_data, 0..) |*val, i| val.* = @floatFromInt((i * 2) % 100);

        var a_buf = try g.createBufferFromSlice(f32, a_data, .{});
        defer g.destroyBuffer(&a_buf);
        var b_buf = try g.createBufferFromSlice(f32, b_data, .{});
        defer g.destroyBuffer(&b_buf);
        var result_buf = try g.createBuffer(f32, size, .{});
        defer g.destroyBuffer(&result_buf);

        var timer = try time.Timer.start();
        var total_time: u64 = 0;

        for (0..self.config.iterations) |_| {
            timer.reset();
            _ = try g.vectorAdd(&a_buf, &b_buf, &result_buf);
            total_time += timer.read();
        }

        const avg_time = total_time / self.config.iterations;
        const throughput = @as(f64, @floatFromInt(size)) / (@as(f64, @floatFromInt(avg_time)) / 1_000_000_000.0);
        const bandwidth = throughput * @sizeOf(f32) * 3 / (1024 * 1024 * 1024); // Read 2, write 1

        try self.recordBenchmark(.{
            .operation = "vector_add",
            .backend = g.getHealth().backend,
            .data_size = size,
            .iterations = self.config.iterations,
            .total_time_ns = total_time,
            .avg_time_ns = avg_time,
            .throughput_elements_per_sec = throughput,
            .memory_bandwidth_gb_per_sec = bandwidth,
            .peak_memory_usage = size * @sizeOf(f32) * 3,
        });
    }

    fn benchmarkMatrixMultiply(self: *TestSuite) !void {
        var g = try gpu.Gpu.init(self.allocator, .{});
        defer g.deinit();

        const dim = 512; // 512x512 matrices
        const size = dim * dim;
        const a_data = try self.allocator.alloc(f32, size);
        defer self.allocator.free(a_data);
        const b_data = try self.allocator.alloc(f32, size);
        defer self.allocator.free(b_data);

        // Initialize with small values to avoid overflow
        for (a_data) |*val| val.* = 0.01;
        for (b_data) |*val| val.* = 0.01;

        var a_buf = try g.createBufferFromSlice(f32, a_data, .{});
        defer g.destroyBuffer(&a_buf);
        var b_buf = try g.createBufferFromSlice(f32, b_data, .{});
        defer g.destroyBuffer(&b_buf);
        var result_buf = try g.createBuffer(f32, size, .{});
        defer g.destroyBuffer(&result_buf);

        const dims = gpu.unified.MatrixDims{ .m = dim, .n = dim, .k = dim };

        var timer = try time.Timer.start();
        var total_time: u64 = 0;

        for (0..self.config.iterations) |_| {
            timer.reset();
            _ = try g.matrixMultiply(&a_buf, &b_buf, &result_buf, dims);
            total_time += timer.read();
        }

        const avg_time = total_time / self.config.iterations;
        const operations = @as(f64, @floatFromInt(dim * dim * dim * 2)); // 2 operations per element
        const throughput = operations / (@as(f64, @floatFromInt(avg_time)) / 1_000_000_000.0);
        const bandwidth = @as(f64, @floatFromInt(size * @sizeOf(f32) * 3)) / (@as(f64, @floatFromInt(avg_time)) / 1_000_000_000.0) / (1024 * 1024 * 1024);

        try self.recordBenchmark(.{
            .operation = "matrix_multiply",
            .backend = g.getHealth().backend,
            .data_size = size,
            .iterations = self.config.iterations,
            .total_time_ns = total_time,
            .avg_time_ns = avg_time,
            .throughput_elements_per_sec = throughput,
            .memory_bandwidth_gb_per_sec = bandwidth,
            .peak_memory_usage = size * @sizeOf(f32) * 3,
        });
    }

    fn benchmarkMemoryBandwidth(self: *TestSuite) !void {
        var g = try gpu.Gpu.init(self.allocator, .{});
        defer g.deinit();

        const size = self.config.data_size;
        const data = try self.allocator.alloc(f32, size);
        defer self.allocator.free(data);

        for (data, 0..) |*val, i| val.* = @floatFromInt(i % 100);

        var buffer = try g.createBuffer(f32, size, .{});
        defer g.destroyBuffer(&buffer);

        var timer = try time.Timer.start();
        var total_time: u64 = 0;

        // Benchmark host-to-device transfer
        for (0..self.config.iterations) |_| {
            timer.reset();
            try buffer.write(f32, data);
            total_time += timer.read();
        }

        const avg_time = total_time / self.config.iterations;
        const bandwidth = @as(f64, @floatFromInt(size * @sizeOf(f32))) / (@as(f64, @floatFromInt(avg_time)) / 1_000_000_000.0) / (1024 * 1024 * 1024);

        try self.recordBenchmark(.{
            .operation = "memory_upload",
            .backend = g.getHealth().backend,
            .data_size = size,
            .iterations = self.config.iterations,
            .total_time_ns = total_time,
            .avg_time_ns = avg_time,
            .throughput_elements_per_sec = @as(f64, @floatFromInt(size)) / (@as(f64, @floatFromInt(avg_time)) / 1_000_000_000.0),
            .memory_bandwidth_gb_per_sec = bandwidth,
            .peak_memory_usage = size * @sizeOf(f32),
        });
    }

    fn stressMemoryPressure(self: *TestSuite) !void {
        var g = try gpu.Gpu.init(self.allocator, .{});
        defer g.deinit();

        const health = try g.getHealth();
        const max_allocation = @as(usize, @intFromFloat(@as(f64, @floatFromInt(health.memory_total)) * self.config.memory_pressure));

        var buffers = std.ArrayListUnmanaged(*gpu.UnifiedBuffer).initCapacity(self.allocator, 100) catch |err| {
            std.log.warn("Failed to allocate buffer list: {t}", .{err});
            return err;
        };
        defer {
            for (buffers.items) |buf| {
                g.destroyBuffer(buf);
                self.allocator.destroy(buf);
            }
            buffers.deinit(self.allocator);
        }

        var allocated: usize = 0;
        var allocation_count: usize = 0;

        // Allocate until we hit memory pressure
        while (allocated < max_allocation and allocation_count < 1000) {
            const alloc_size = @min(1024 * 1024, max_allocation - allocated); // 1MB chunks
            if (alloc_size == 0) break;

            const buf = self.allocator.create(gpu.UnifiedBuffer) catch break;
            buf.* = g.createBuffer(f32, alloc_size / @sizeOf(f32), .{}) catch {
                self.allocator.destroy(buf);
                break;
            };

            buffers.append(self.allocator, buf) catch {
                g.destroyBuffer(buf);
                self.allocator.destroy(buf);
                break;
            };

            allocated += alloc_size;
            allocation_count += 1;
        }

        std.log.info("Allocated {} MB across {} buffers", .{ allocated / (1024 * 1024), allocation_count });

        // Perform operations under memory pressure
        if (buffers.items.len >= 2) {
            const a = buffers.items[0];
            const b = buffers.items[1];
            const result = buffers.items[2];

            for (0..10) |_| {
                _ = try g.vectorAdd(a, b, result);
            }
        }
    }

    fn stressConcurrentOperations(self: *TestSuite) !void {
        var g = try gpu.Gpu.init(self.allocator, .{});
        defer g.deinit();

        const num_threads = 4;
        const operations_per_thread = 100;

        const threads = try self.allocator.alloc(std.Thread, num_threads);
        defer self.allocator.free(threads);

        const contexts = try self.allocator.alloc(ThreadContext, num_threads);
        defer self.allocator.free(contexts);

        // Initialize thread contexts
        for (contexts, 0..) |*ctx, i| {
            ctx.gpu = &g;
            ctx.thread_id = i;
            ctx.operations = operations_per_thread;
        }

        // Start threads
        for (threads, contexts) |*thread, *ctx| {
            thread.* = try std.Thread.spawn(.{}, concurrentOperations, .{ctx});
        }

        // Wait for completion
        for (threads) |thread| {
            thread.join();
        }

        std.log.info("Completed {} concurrent operations across {} threads", .{ operations_per_thread * num_threads, num_threads });
    }

    const ThreadContext = struct {
        gpu: *gpu.Gpu,
        thread_id: usize,
        operations: usize,
    };

    fn concurrentOperations(ctx: *const ThreadContext) void {
        const size = 1024;
        const a_data = [_]f32{1.0} ** size;
        const b_data = [_]f32{2.0} ** size;

        for (0..ctx.operations) |_| {
            var a_buf = ctx.gpu.createBufferFromSlice(f32, &a_data, .{}) catch continue;
            defer ctx.gpu.destroyBuffer(&a_buf);
            var b_buf = ctx.gpu.createBufferFromSlice(f32, &b_data, .{}) catch continue;
            defer ctx.gpu.destroyBuffer(&b_buf);
            var result_buf = ctx.gpu.createBuffer(f32, size, .{}) catch continue;
            defer ctx.gpu.destroyBuffer(&result_buf);

            _ = ctx.gpu.vectorAdd(&a_buf, &b_buf, &result_buf) catch continue;
        }
    }

    fn testDevicePeerTransfer(_: *TestSuite, devices: []const device_mod.Device) !void {
        if (devices.len < 2) return;

        // Test peer-to-peer transfer capabilities
        const dev0 = &devices[0];
        const dev1 = &devices[1];

        const capabilities = try gpu.peer_transfer.getPeerTransferCapabilities(dev0, dev1);
        std.log.info("Peer transfer between {} and {}: {}", .{
            dev0.name, dev1.name, capabilities,
        });
    }

    fn testLoadBalancing(self: *TestSuite, devices: []const device_mod.Device) !void {
        if (devices.len < 2) return;

        // Create a multi-device group
        var device_group = try gpu.multi_device.DeviceGroup.init(self.allocator, devices);
        defer device_group.deinit();

        // Test load balancing across devices
        const workload_size = 1024 * 1024;
        const distribution = try device_group.computeWorkloadDistribution(workload_size);
        defer self.allocator.free(distribution);

        std.log.info("Workload distribution across {} devices:", .{devices.len});
        for (distribution, 0..) |size, i| {
            std.log.info("  Device {}: {} elements", .{ i, size });
        }
    }

    fn recordBenchmark(self: *TestSuite, result: BenchmarkResult) !void {
        try self.results.append(self.allocator, result);

        // Use {t} format for enums (Zig 0.16)
        std.log.info("Benchmark: {s} on {t}", .{ result.operation, result.backend });
        std.log.info("  Data size: {} elements", .{result.data_size});
        std.log.info("  Iterations: {}", .{result.iterations});
        std.log.info("  Avg time: {} ns", .{result.avg_time_ns});
        std.log.info("  Throughput: {:.2} elements/sec", .{result.throughput_elements_per_sec});
        std.log.info("  Memory bandwidth: {:.2} GB/s", .{result.memory_bandwidth_gb_per_sec});
        std.log.info("  Peak memory: {} MB", .{result.peak_memory_usage / (1024 * 1024)});
    }

    /// Get benchmark results
    pub fn getResults(self: *const TestSuite) []const BenchmarkResult {
        return self.results.items;
    }

    /// Generate performance report
    pub fn generateReport(self: *const TestSuite) void {
        std.log.info("GPU Performance Test Report", .{});
        std.log.info("==========================", .{});

        std.log.info("Test Configuration:", .{});
        std.log.info("  Iterations: {}", .{self.config.iterations});
        std.log.info("  Data size: {} elements", .{self.config.data_size});
        std.log.info("  Benchmark: {}", .{self.config.benchmark});
        std.log.info("  Stress test: {}", .{self.config.stress_test});
        std.log.info("  Multi-device: {}", .{self.config.multi_device});

        if (self.results.items.len > 0) {
            std.log.info("Benchmark Results:", .{});
            for (self.results.items) |result| {
                // Use {t} format for enums (Zig 0.16)
                std.log.info("  {s} ({t}):", .{ result.operation, result.backend });
                std.log.info("    Throughput: {:.2} elements/sec", .{result.throughput_elements_per_sec});
                std.log.info("    Memory BW: {:.2} GB/s", .{result.memory_bandwidth_gb_per_sec});
                std.log.info("    Avg latency: {} ns", .{result.avg_time_ns});
            }
        }
    }
};

test "comprehensive GPU test suite" {
    const allocator = std.testing.allocator;

    var suite = TestSuite.init(allocator, .{
        .iterations = 10, // Reduced for test
        .data_size = 1024, // Reduced for test
        .benchmark = true,
        .stress_test = false, // Skip stress tests in unit tests
        .multi_device = false, // Skip multi-device in unit tests
    });
    defer suite.deinit();

    try suite.runAll();

    // Verify we got some results
    try std.testing.expect(suite.getResults().len > 0);
}

test {
    std.testing.refAllDecls(@This());
}
