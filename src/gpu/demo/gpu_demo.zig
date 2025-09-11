//! GPU Backend Manager Demo
//!
//! This demo shows how to use the GPU Backend Manager to:
//! - Detect available GPU backends
//! - Initialize CUDA and SPIRV compilers
//! - Query hardware capabilities
//! - Run performance benchmarks

const std = @import("std");
const gpu = @import("../mod.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("🚀 GPU Backend Manager Demo", .{});
    std.log.info("============================", .{});

    // Initialize GPU Backend Manager
    var backend_manager = try gpu.GPUBackendManager.init(allocator);
    defer backend_manager.deinit();

    // Print system information
    backend_manager.printSystemInfo();

    // Check for CUDA support
    if (backend_manager.cuda_driver) |cuda| {
        std.log.info("✅ CUDA Driver Available", .{});
        std.log.info("  Device Count: {}", .{cuda.getDeviceCount()});

        if (cuda.getDeviceCount() > 0) {
            const caps = try cuda.getDeviceProperties(0);
            std.log.info("  Primary GPU: {}", .{caps});
        }
    } else {
        std.log.info("❌ CUDA Driver Not Available", .{});
    }

    // Check for SPIRV compiler
    if (backend_manager.spirv_compiler) |_| {
        std.log.info("✅ SPIRV Compiler Available", .{});
    } else {
        std.log.info("❌ SPIRV Compiler Not Available", .{});
    }

    // Test backend capabilities
    if (backend_manager.current_backend) |current_backend| {
        std.log.info("🔍 Testing {} Backend Capabilities", .{current_backend.displayName()});

        const caps = try backend_manager.getBackendCapabilities(current_backend);
        std.log.info("  Capabilities: {}", .{caps});

        // Test shader compilation (if supported)
        if (backend_manager.spirv_compiler != null) {
            const test_shader = @embedFile("test_shader.glsl");
            std.log.info("  Testing shader compilation...", .{});

            const compiled = backend_manager.compileShader(test_shader, .compute) catch |err| {
                std.log.warn("  Shader compilation failed: {}", .{err});
                return;
            };

            std.log.info("  ✅ Shader compilation successful ({} bytes)", .{compiled.len});
            allocator.free(compiled);
        }
    }

    // Run memory bandwidth benchmark
    std.log.info("📊 Running Memory Bandwidth Benchmark", .{});
    var mem_benchmark = try gpu.MemoryBandwidthBenchmark.init(allocator, null); // No renderer for this demo
    defer mem_benchmark.deinit();

    // Note: This would require actual GPU buffers, so we'll skip the actual benchmark
    std.log.info("  Memory benchmark requires GPU renderer (skipped in demo)", .{});

    // Run compute throughput benchmark
    std.log.info("⚡ Running Compute Throughput Benchmark", .{});
    var compute_benchmark = try gpu.ComputeThroughputBenchmark.init(allocator, null);
    defer compute_benchmark.deinit();

    const throughput = try compute_benchmark.measureComputeThroughput(256, 100);
    std.log.info("  Simulated throughput: {d:.2} GFLOPS", .{throughput});

    // Test kernel manager
    std.log.info("🧠 Testing Kernel Manager", .{});
    var kernel_manager = try gpu.KernelManager.init(allocator, null); // No renderer for this demo
    defer kernel_manager.deinit();

    std.log.info("  ✅ Kernel manager initialized successfully", .{});

    std.log.info("🎉 GPU Backend Manager Demo Complete!", .{});
    std.log.info("=============================================", .{});
    std.log.info("Summary:", .{});
    std.log.info("  - Available backends: {}", .{backend_manager.available_backends.items.len});
    std.log.info("  - Current backend: {s}", .{if (backend_manager.current_backend) |b| b.displayName() else "None"});
    std.log.info("  - CUDA support: {}", .{backend_manager.cuda_driver != null});
    std.log.info("  - SPIRV support: {}", .{backend_manager.spirv_compiler != null});
}
