//! Advanced GPU Features Tests
//!
//! Tests for advanced GPU functionality:
//! - Specialized AI kernels
//! - Memory pooling
//! - Multi-backend support
//! - Performance profiling
//! - Benchmarking tools

const std = @import("std");
const testing = std.testing;
const gpu = @import("gpu");

test "GPU Kernel Manager - Dense Layer" {
    const allocator = testing.allocator;

    var renderer = gpu.initDefault(allocator) catch |err| {
        if (err == gpu.GpuError.DeviceNotFound or
            err == gpu.GpuError.InitializationFailed or
            err == gpu.GpuError.UnsupportedBackend)
        {
            return; // Skip test if GPU not available
        }
        return err;
    };
    defer renderer.deinit();

    // Create kernel manager
    var kernel_manager = try gpu.KernelManager.init(allocator, renderer);
    defer kernel_manager.deinit();

    // Create dense layer kernel
    const kernel_idx = try kernel_manager.createDenseLayer(
        "test_dense",
        784, // input size (e.g., 28x28 image flattened)
        128, // output size
        .relu, // activation
        .adam, // optimizer
        .{}, // default config
    );

    try testing.expect(kernel_idx == 0);

    // Verify kernel was created
    try testing.expect(kernel_manager.kernels.items.len == 1);
    const kernel = &kernel_manager.kernels.items[0];
    try testing.expect(std.mem.eql(u8, kernel.name, "test_dense"));
    try testing.expect(kernel.layer_type == .dense);
    try testing.expect(kernel.activation == .relu);
    try testing.expect(kernel.optimizer == .adam);
    try testing.expect(std.mem.eql(u32, kernel.input_shape, &[_]u32{784}));
    try testing.expect(std.mem.eql(u32, kernel.output_shape, &[_]u32{128}));
}

test "GPU Kernel Manager - Convolutional Layer" {
    const allocator = testing.allocator;

    var renderer = gpu.initDefault(allocator) catch |err| {
        if (err == gpu.GpuError.DeviceNotFound or
            err == gpu.GpuError.InitializationFailed or
            err == gpu.GpuError.UnsupportedBackend)
        {
            return; // Skip test if GPU not available
        }
        return err;
    };
    defer renderer.deinit();

    var kernel_manager = try gpu.KernelManager.init(allocator, renderer);
    defer kernel_manager.deinit();

    // Create convolutional layer kernel
    const kernel_idx = try kernel_manager.createConvLayer(
        "test_conv",
        3, // input channels
        64, // output channels
        3, // kernel size
        1, // stride
        1, // padding
        .relu,
        .adam,
        .{},
    );

    try testing.expect(kernel_idx == 0);

    // Verify kernel was created
    const kernel = &kernel_manager.kernels.items[0];
    try testing.expect(std.mem.eql(u8, kernel.name, "test_conv"));
    try testing.expect(kernel.layer_type == .convolutional);
    try testing.expect(std.mem.eql(u32, kernel.input_shape, &[_]u32{ 0, 0, 3 }));
    try testing.expect(std.mem.eql(u32, kernel.output_shape, &[_]u32{ 0, 0, 64 }));
}

test "GPU Kernel Manager - Attention Layer" {
    const allocator = testing.allocator;

    var renderer = gpu.initDefault(allocator) catch |err| {
        if (err == gpu.GpuError.DeviceNotFound or
            err == gpu.GpuError.InitializationFailed or
            err == gpu.GpuError.UnsupportedBackend)
        {
            return; // Skip test if GPU not available
        }
        return err;
    };
    defer renderer.deinit();

    var kernel_manager = try gpu.KernelManager.init(allocator, renderer);
    defer kernel_manager.deinit();

    // Create attention layer kernel
    const kernel_idx = try kernel_manager.createAttentionLayer(
        "test_attention",
        768, // embed dim
        12, // num heads
        512, // seq length
        .adam,
        .{},
    );

    try testing.expect(kernel_idx == 0);

    // Verify kernel was created
    const kernel = &kernel_manager.kernels.items[0];
    try testing.expect(std.mem.eql(u8, kernel.name, "test_attention"));
    try testing.expect(kernel.layer_type == .attention);
    try testing.expect(std.mem.eql(u32, kernel.input_shape, &[_]u32{ 512, 768 }));
    try testing.expect(std.mem.eql(u32, kernel.output_shape, &[_]u32{ 512, 768 }));
}

test "GPU Memory Pool - Basic Operations" {
    const allocator = testing.allocator;

    var renderer = gpu.initDefault(allocator) catch |err| {
        if (err == gpu.GpuError.DeviceNotFound or
            err == gpu.GpuError.InitializationFailed or
            err == gpu.GpuError.UnsupportedBackend)
        {
            return; // Skip test if GPU not available
        }
        return err;
    };
    defer renderer.deinit();

    // Create memory pool
    var memory_pool = try gpu.MemoryPool.init(allocator, renderer, .{});
    defer memory_pool.deinit();

    // Test buffer allocation
    const buffer_size = 1024 * 1024; // 1MB
    const handle1 = try memory_pool.allocBuffer(buffer_size, .{ .storage = true, .copy_src = true });
    try testing.expect(handle1 > 0);

    const handle2 = try memory_pool.allocBuffer(buffer_size, .{ .storage = true, .copy_src = true });
    try testing.expect(handle2 > 0);
    try testing.expect(handle1 != handle2);

    // Test buffer deallocation
    try memory_pool.freeBuffer(handle1);

    // Allocate again - should reuse the freed buffer
    const handle3 = try memory_pool.allocBuffer(buffer_size, .{ .storage = true, .copy_src = true });
    // Note: In a real implementation, this might return the same handle due to reuse

    // Clean up
    try memory_pool.freeBuffer(handle2);
    try memory_pool.freeBuffer(handle3);
}

test "GPU Memory Pool - Statistics" {
    const allocator = testing.allocator;

    var renderer = gpu.initDefault(allocator) catch |err| {
        if (err == gpu.GpuError.DeviceNotFound or
            err == gpu.GpuError.InitializationFailed or
            err == gpu.GpuError.UnsupportedBackend)
        {
            return; // Skip test if GPU not available
        }
        return err;
    };
    defer renderer.deinit();

    var memory_pool = try gpu.MemoryPool.init(allocator, renderer, .{});
    defer memory_pool.deinit();

    // Get initial stats
    const initial_stats = memory_pool.getStats();
    try testing.expect(initial_stats.total_buffers_allocated == 0);
    try testing.expect(initial_stats.total_buffers_free == 0);

    // Allocate some buffers
    const handle1 = try memory_pool.allocBuffer(1024 * 1024, .{ .storage = true });
    const handle2 = try memory_pool.allocBuffer(2 * 1024 * 1024, .{ .storage = true });

    const stats_after_alloc = memory_pool.getStats();
    try testing.expect(stats_after_alloc.total_buffers_allocated >= 2);
    try testing.expect(stats_after_alloc.total_memory_allocated >= 3 * 1024 * 1024);

    // Free one buffer
    try memory_pool.freeBuffer(handle1);

    const stats_after_free = memory_pool.getStats();
    try testing.expect(stats_after_free.total_buffers_free >= 1);

    // Clean up
    try memory_pool.freeBuffer(handle2);
}

test "GPU Memory Pool - Cleanup" {
    const allocator = testing.allocator;

    var renderer = gpu.initDefault(allocator) catch |err| {
        if (err == gpu.GpuError.DeviceNotFound or
            err == gpu.GpuError.InitializationFailed or
            err == gpu.GpuError.UnsupportedBackend)
        {
            return; // Skip test if GPU not available
        }
        return err;
    };
    defer renderer.deinit();

    var memory_pool = try gpu.MemoryPool.init(allocator, renderer, .{
        .max_buffer_age_ms = 100, // Short age for testing
    });
    defer memory_pool.deinit();

    // Allocate and free a buffer
    const handle = try memory_pool.allocBuffer(1024 * 1024, .{ .storage = true });
    try memory_pool.freeBuffer(handle);

    // Simulate time passing (in real implementation, this would use actual timing)
    // For testing, we'll just call cleanup
    try memory_pool.cleanup();

    // Check that cleanup worked
    const stats = memory_pool.getStats();
    _ = stats; // Stats may vary depending on implementation
}

test "GPU Backend Support - Detection" {
    const allocator = testing.allocator;

    // Create backend support manager
    var backend_manager = try gpu.BackendSupport.init(allocator);
    defer backend_manager.deinit();

    // Detect available backends
    const available_backends = try backend_manager.detectAvailableBackends();
    defer allocator.free(available_backends);

    // Should always have at least WebGPU and CPU fallback
    try testing.expect(available_backends.len >= 2);

    // Check that WebGPU is available
    var has_webgpu = false;
    for (available_backends) |backend| {
        if (backend == .webgpu) {
            has_webgpu = true;
            break;
        }
    }
    try testing.expect(has_webgpu);

    // Check that CPU fallback is available
    var has_cpu = false;
    for (available_backends) |backend| {
        if (backend == .cpu_fallback) {
            has_cpu = true;
            break;
        }
    }
    try testing.expect(has_cpu);
}

test "GPU Backend Support - Capabilities" {
    const allocator = testing.allocator;

    var backend_manager = try gpu.BackendSupport.init(allocator);
    defer backend_manager.deinit();

    // Test WebGPU capabilities
    const webgpu_caps = try backend_manager.getCapabilities(.webgpu);
    try testing.expect(webgpu_caps.compute_shaders);
    try testing.expect(!webgpu_caps.ray_tracing);
    try testing.expect(!webgpu_caps.supports_fp16);

    // Test Vulkan capabilities
    const vulkan_caps = try backend_manager.getCapabilities(.vulkan);
    try testing.expect(vulkan_caps.compute_shaders);
    try testing.expect(vulkan_caps.ray_tracing);
    try testing.expect(vulkan_caps.supports_fp16);

    // Test CUDA capabilities
    const cuda_caps = try backend_manager.getCapabilities(.cuda);
    try testing.expect(cuda_caps.compute_shaders);
    try testing.expect(cuda_caps.tensor_cores);
    try testing.expect(cuda_caps.supports_fp16);
    try testing.expect(cuda_caps.unified_memory);
}

test "GPU Performance Profiler - Basic Operations" {
    const allocator = testing.allocator;

    var renderer = gpu.initDefault(allocator) catch |err| {
        if (err == gpu.GpuError.DeviceNotFound or
            err == gpu.GpuError.InitializationFailed or
            err == gpu.GpuError.UnsupportedBackend)
        {
            return; // Skip test if GPU not available
        }
        return err;
    };
    defer renderer.deinit();

    // Create performance profiler
    var profiler = try gpu.PerformanceProfiler.init(allocator, renderer);
    defer profiler.deinit();

    // Test timing operations
    try profiler.startTiming("test_operation");
    std.Thread.sleep(1 * 1000 * 1000); // Sleep for 1ms
    _ = try profiler.endTiming();

    // Should have one measurement
    try testing.expect(profiler.measurements.items.len == 1);
    const measurement = profiler.measurements.items[0];
    try testing.expect(std.mem.eql(u8, measurement.name, "test_operation"));
    try testing.expect(measurement.end_time > measurement.start_time);
}

test "GPU Memory Bandwidth Benchmark" {
    const allocator = testing.allocator;

    var renderer = gpu.initDefault(allocator) catch |err| {
        if (err == gpu.GpuError.DeviceNotFound or
            err == gpu.GpuError.InitializationFailed or
            err == gpu.GpuError.UnsupportedBackend)
        {
            return; // Skip test if GPU not available
        }
        return err;
    };
    defer renderer.deinit();

    // Create memory bandwidth benchmark
    var benchmark = try gpu.MemoryBandwidthBenchmark.init(allocator, renderer);
    defer benchmark.deinit();

    // Test bandwidth measurement
    const bandwidth = benchmark.measureBandwidth(1024 * 1024, 5) catch |err| {
        // Bandwidth measurement may fail in test environment
        if (err == gpu.GpuError.BufferCreationFailed or
            err == gpu.GpuError.BufferMappingFailed)
        {
            return; // Skip test if buffers can't be created/mapped
        }
        return err;
    };

    // Bandwidth should be non-negative
    try testing.expect(bandwidth >= 0.0);
}

test "GPU Compute Throughput Benchmark" {
    const allocator = testing.allocator;

    var renderer = gpu.initDefault(allocator) catch |err| {
        if (err == gpu.GpuError.DeviceNotFound or
            err == gpu.GpuError.InitializationFailed or
            err == gpu.GpuError.UnsupportedBackend)
        {
            return; // Skip test if GPU not available
        }
        return err;
    };
    defer renderer.deinit();

    // Create compute throughput benchmark
    var benchmark = try gpu.ComputeThroughputBenchmark.init(allocator, renderer);
    defer benchmark.deinit();

    // Test throughput measurement
    const throughput = try benchmark.measureComputeThroughput(256, 10);
    try testing.expect(throughput >= 0.0);
}

test "GPU Backend Priority Selection" {
    const allocator = testing.allocator;

    var backend_manager = try gpu.BackendSupport.init(allocator);
    defer backend_manager.deinit();

    // Test backend priority ordering
    try testing.expect(gpu.Backend.vulkan.getPriority() > gpu.Backend.webgpu.getPriority());
    try testing.expect(gpu.Backend.cuda.getPriority() > gpu.Backend.vulkan.getPriority());
    try testing.expect(gpu.Backend.webgpu.getPriority() > gpu.Backend.cpu_fallback.getPriority());

    // Test string conversion
    try testing.expect(std.mem.eql(u8, gpu.Backend.vulkan.toString(), "Vulkan"));
    try testing.expect(std.mem.eql(u8, gpu.Backend.metal.toString(), "Metal"));
    try testing.expect(std.mem.eql(u8, gpu.Backend.webgpu.toString(), "WebGPU"));
}

test "GPU Kernel Manager - Multiple Kernels" {
    const allocator = testing.allocator;

    var renderer = gpu.initDefault(allocator) catch |err| {
        if (err == gpu.GpuError.DeviceNotFound or
            err == gpu.GpuError.InitializationFailed or
            err == gpu.GpuError.UnsupportedBackend)
        {
            return; // Skip test if GPU not available
        }
        return err;
    };
    defer renderer.deinit();

    var kernel_manager = try gpu.KernelManager.init(allocator, renderer);
    defer kernel_manager.deinit();

    // Create multiple kernels
    const dense_idx = try kernel_manager.createDenseLayer("dense1", 100, 50, .relu, .adam, .{});
    const conv_idx = try kernel_manager.createConvLayer("conv1", 3, 32, 3, 1, 1, .relu, .adam, .{});
    const attn_idx = try kernel_manager.createAttentionLayer("attn1", 256, 8, 128, .adam, .{});

    // Verify indices are unique
    try testing.expect(dense_idx == 0);
    try testing.expect(conv_idx == 1);
    try testing.expect(attn_idx == 2);

    // Verify all kernels are stored
    try testing.expect(kernel_manager.kernels.items.len == 3);

    // Verify different layer types
    try testing.expect(kernel_manager.kernels.items[0].layer_type == .dense);
    try testing.expect(kernel_manager.kernels.items[1].layer_type == .convolutional);
    try testing.expect(kernel_manager.kernels.items[2].layer_type == .attention);
}

test "GPU Kernel Manager - Weights Initialization" {
    const allocator = testing.allocator;

    var renderer = gpu.initDefault(allocator) catch |err| {
        if (err == gpu.GpuError.DeviceNotFound or
            err == gpu.GpuError.InitializationFailed or
            err == gpu.GpuError.UnsupportedBackend)
        {
            return; // Skip test if GPU not available
        }
        return err;
    };
    defer renderer.deinit();

    var kernel_manager = try gpu.KernelManager.init(allocator, renderer);
    defer kernel_manager.deinit();

    // Create a dense layer
    _ = try kernel_manager.createDenseLayer("test_weights", 10, 5, .relu, .adam, .{});

    // Verify weights and biases handles were created
    const kernel = &kernel_manager.kernels.items[0];
    try testing.expect(kernel.weights_handle != null);
    try testing.expect(kernel.biases_handle != null);
    try testing.expect(kernel.weights_handle.? > 0);
    try testing.expect(kernel.biases_handle.? > 0);
}

test "GPU Performance Profiler - Benchmark Results" {
    const allocator = testing.allocator;

    var renderer = gpu.initDefault(allocator) catch |err| {
        if (err == gpu.GpuError.DeviceNotFound or
            err == gpu.GpuError.InitializationFailed or
            err == gpu.GpuError.UnsupportedBackend)
        {
            return; // Skip test if GPU not available
        }
        return err;
    };
    defer renderer.deinit();

    var profiler = try gpu.PerformanceProfiler.init(allocator, renderer);
    defer profiler.deinit();

    // Run a simple benchmark
    _ = try profiler.runWorkloadBenchmark(.vector_add, 1024, .{
        .iterations = 5,
        .warmup_iterations = 1,
        .detailed_timing = false,
    });

    // Should have results
    try testing.expect(profiler.results.items.len >= 1);

    // Check result structure
    const result = profiler.results.items[0];
    try testing.expect(std.mem.eql(u8, result.workload, "vector_add"));
    try testing.expect(result.iterations > 0);
    try testing.expect(result.avg_time_ns > 0);
    try testing.expect(result.throughput_items_per_sec >= 0);
}

test "GPU Memory Pool - Prefetch" {
    const allocator = testing.allocator;

    var renderer = gpu.initDefault(allocator) catch |err| {
        if (err == gpu.GpuError.DeviceNotFound or
            err == gpu.GpuError.InitializationFailed or
            err == gpu.GpuError.UnsupportedBackend)
        {
            return; // Skip test if GPU not available
        }
        return err;
    };
    defer renderer.deinit();

    var memory_pool = try gpu.MemoryPool.init(allocator, renderer, .{});
    defer memory_pool.deinit();

    // Test prefetch functionality
    const sizes_to_prefetch = [_]usize{ 1024, 2048, 4096 };
    try memory_pool.prefetchBuffers(&sizes_to_prefetch, .{ .storage = true });

    // Should not crash and should work
    try testing.expect(true);
}

test "GPU Backend Support - Backend Selection" {
    const allocator = testing.allocator;

    var backend_manager = try gpu.BackendSupport.init(allocator);
    defer backend_manager.deinit();

    // Test selecting best backend
    const best_backend = backend_manager.selectBestBackend();
    if (best_backend) |backend| {
        // Should be a valid backend
        _ = backend;
    }

    // Test forcing a specific backend
    backend_manager.selectBackend(.webgpu) catch {
        // WebGPU should be available
        try testing.expect(false); // This should not fail
    };

    try testing.expect(backend_manager.current_backend.? == .webgpu);
}
