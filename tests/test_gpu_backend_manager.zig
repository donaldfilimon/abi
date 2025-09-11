//! GPU Backend Manager Tests
//!
//! Tests for the GPU Backend Manager including:
//! - Backend detection and selection
//! - CUDA driver integration
//! - SPIRV compiler functionality
//! - Hardware capability querying

const std = @import("std");
const testing = std.testing;
const gpu = @import("gpu");

test "GPU Backend Manager - Initialization" {
    const allocator = testing.allocator;

    // Initialize backend manager
    var backend_manager = try gpu.GPUBackendManager.init(allocator);
    defer backend_manager.deinit();

    // Should have at least CPU fallback
    try testing.expect(backend_manager.available_backends.items.len >= 1);

    // Should have selected a backend
    try testing.expect(backend_manager.current_backend != null);
}

test "GPU Backend Manager - Backend Detection" {
    const allocator = testing.allocator;

    var backend_manager = try gpu.GPUBackendManager.init(allocator);
    defer backend_manager.deinit();

    // Check that backends are properly sorted by priority
    var highest_priority: u8 = 0;
    for (backend_manager.available_backends.items) |backend| {
        const priority = backend.priority();
        try testing.expect(priority <= highest_priority or highest_priority == 0);
        highest_priority = priority;
    }
}

test "GPU Backend Manager - Backend Selection" {
    const allocator = testing.allocator;

    var backend_manager = try gpu.GPUBackendManager.init(allocator);
    defer backend_manager.deinit();

    // Test selecting WebGPU (should always be available)
    try backend_manager.selectBackend(.webgpu);
    try testing.expect(backend_manager.current_backend.? == .webgpu);

    // Test selecting CPU fallback
    try backend_manager.selectBackend(.cpu_fallback);
    try testing.expect(backend_manager.current_backend.? == .cpu_fallback);
}

test "GPU Backend Manager - Hardware Capabilities" {
    const allocator = testing.allocator;

    var backend_manager = try gpu.GPUBackendManager.init(allocator);
    defer backend_manager.deinit();

    // Get capabilities for each available backend
    for (backend_manager.available_backends.items) |backend| {
        const caps = try backend_manager.getBackendCapabilities(backend);

        // Basic validation
        try testing.expect(caps.name.len > 0);
        try testing.expect(caps.vendor.len > 0);
        try testing.expect(caps.max_workgroup_size > 0);
    }
}

test "GPU Backend Manager - CUDA Driver" {
    const allocator = testing.allocator;

    var backend_manager = try gpu.GPUBackendManager.init(allocator);
    defer backend_manager.deinit();

    if (backend_manager.cuda_driver) |cuda| {
        // Test CUDA device count
        const device_count = cuda.getDeviceCount();
        try testing.expect(device_count >= 0);

        // If devices are available, test device properties
        if (device_count > 0) {
            const caps = try cuda.getDeviceProperties(0);
            try testing.expect(caps.name.len > 0);
            try testing.expect(caps.total_memory_mb > 0);
            try testing.expect(caps.compute_units > 0);
        }
    } else {
        // CUDA not available - this is fine
        try testing.expect(true);
    }
}

test "GPU Backend Manager - SPIRV Compiler" {
    const allocator = testing.allocator;

    var backend_manager = try gpu.GPUBackendManager.init(allocator);
    defer backend_manager.deinit();

    if (backend_manager.spirv_compiler) |spirv| {
        // Test SPIRV validation
        const test_spirv = [_]u32{
            0x07230203, // Magic number
            0x00010000, // Version
            0, // Generator
            0, // Bound
            0, // Schema
        };

        const is_valid = try spirv.validateSPIRV(&test_spirv);
        try testing.expect(is_valid);

        // Test SPIRV disassembly
        const disassembly = try spirv.disassembleSPIRV(&test_spirv);
        defer allocator.free(disassembly);
        try testing.expect(disassembly.len > 0);
    } else {
        // SPIRV compiler not available - this is fine
        try testing.expect(true);
    }
}

test "GPU Backend Manager - Backend Priority" {
    const allocator = testing.allocator;

    var backend_manager = try gpu.GPUBackendManager.init(allocator);
    defer backend_manager.deinit();

    // Verify backend priority ordering
    for (0..backend_manager.available_backends.items.len - 1) |i| {
        const current = backend_manager.available_backends.items[i];
        const next = backend_manager.available_backends.items[i + 1];
        try testing.expect(current.priority() >= next.priority());
    }
}

test "GPU Backend Manager - Backend Availability Check" {
    const allocator = testing.allocator;

    var backend_manager = try gpu.GPUBackendManager.init(allocator);
    defer backend_manager.deinit();

    // Test hasBackend function
    for (backend_manager.available_backends.items) |backend| {
        try testing.expect(backend_manager.hasBackend(backend));
    }

    // Test non-existent backend
    try testing.expect(!backend_manager.hasBackend(.cuda)); // May not be available
}

test "GPU Memory Bandwidth Benchmark - Basic" {
    const allocator = testing.allocator;

    var benchmark = try gpu.MemoryBandwidthBenchmark.init(allocator, null);
    defer benchmark.deinit();

    // Should initialize without error
    try testing.expect(true);
}

test "GPU Compute Throughput Benchmark - Basic" {
    const allocator = testing.allocator;

    var benchmark = try gpu.ComputeThroughputBenchmark.init(allocator, null);
    defer benchmark.deinit();

    // Test throughput measurement
    const throughput = try benchmark.measureComputeThroughput(256, 10);
    try testing.expect(throughput >= 0.0);
}

test "GPU Kernel Manager - Basic Operations" {
    const allocator = testing.allocator;

    var kernel_manager = try gpu.KernelManager.init(allocator, null);
    defer kernel_manager.deinit();

    // Should initialize without error
    try testing.expect(kernel_manager.kernels.items.len == 0);
}

test "GPU Backend Manager - Multiple Backend Support" {
    const allocator = testing.allocator;

    var backend_manager = try gpu.GPUBackendManager.init(allocator);
    defer backend_manager.deinit();

    // Should detect multiple backends
    const backend_count = backend_manager.available_backends.items.len;
    try testing.expect(backend_count >= 2); // At least WebGPU and CPU fallback

    // Should include WebGPU and CPU fallback
    var has_webgpu = false;
    var has_cpu_fallback = false;

    for (backend_manager.available_backends.items) |backend| {
        if (backend == .webgpu) has_webgpu = true;
        if (backend == .cpu_fallback) has_cpu_fallback = true;
    }

    try testing.expect(has_webgpu);
    try testing.expect(has_cpu_fallback);
}

test "GPU Backend Manager - Hardware Capabilities Validation" {
    const allocator = testing.allocator;

    var backend_manager = try gpu.GPUBackendManager.init(allocator);
    defer backend_manager.deinit();

    // Test that hardware capabilities are reasonable
    const caps = backend_manager.hardware_caps;

    // Basic sanity checks
    try testing.expect(caps.max_workgroup_size > 0);
    try testing.expect(caps.max_workgroup_size <= 1024 * 1024); // Reasonable upper bound

    // Memory should be reasonable
    if (caps.total_memory_mb > 0) {
        try testing.expect(caps.total_memory_mb <= 128 * 1024); // 128GB max reasonable
    }
}

test "GPU Backend Manager - Backend String Names" {
    // Test that all backends have valid string names
    inline for (@typeInfo(gpu.BackendType).Enum.fields) |field| {
        const backend = @field(gpu.BackendType, field.name);
        const name = backend.displayName();
        try testing.expect(name.len > 0);
        try testing.expect(!std.mem.eql(u8, name, ""));
    }
}

test "GPU Backend Manager - Backend Priority Values" {
    // Test that all backends have valid priority values
    inline for (@typeInfo(gpu.BackendType).Enum.fields) |field| {
        const backend = @field(gpu.BackendType, field.name);
        const priority = backend.priority();
        try testing.expect(priority >= 1 and priority <= 100);
    }
}
