//! Enhanced GPU Demo with Advanced Library Integration
//!
//! This demo showcases the integration of advanced GPU libraries including:
//! - Vulkan bindings for advanced graphics
//! - Mach/GPU for cross-platform compatibility
//! - CUDA for GPU-accelerated computing
//! - SIMD optimizations for enhanced performance

const std = @import("std");
const gpu = @import("gpu");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("🚀 Enhanced GPU Demo with Advanced Library Integration", .{});
    std.log.info("========================================================", .{});

    // Initialize GPU Library Manager
    var library_manager = try gpu.GPULibraryManager.init(allocator);
    defer library_manager.deinit();

    // Display available libraries
    const available_libs = library_manager.getAvailableLibraries();
    std.log.info("📚 Available GPU Libraries:", .{});
    std.log.info("  - Vulkan: {}", .{available_libs.vulkan});
    std.log.info("  - Mach/GPU: {}", .{available_libs.mach_gpu});
    std.log.info("  - CUDA: {}", .{available_libs.cuda});
    std.log.info("  - SIMD: {}", .{available_libs.simd});

    // Initialize available libraries
    if (available_libs.vulkan) {
        std.log.info("🔧 Initializing Vulkan renderer...", .{});
        library_manager.initVulkan() catch |err| {
            std.log.warn("❌ Vulkan initialization failed: {}", .{err});
        };
    }

    if (available_libs.mach_gpu) {
        std.log.info("🔧 Initializing Mach/GPU renderer...", .{});
        library_manager.initMachGPU(.auto) catch |err| {
            std.log.warn("❌ Mach/GPU initialization failed: {}", .{err});
        };
    }

    if (available_libs.cuda) {
        std.log.info("🔧 Initializing CUDA renderer...", .{});
        library_manager.initCUDA() catch |err| {
            std.log.warn("❌ CUDA initialization failed: {}", .{err});
        };
    }

    // Display library status
    const status = library_manager.getLibraryStatus();
    std.log.info("📊 Library Status:", .{});
    std.log.info("  - Vulkan: {s}", .{@tagName(status.vulkan)});
    std.log.info("  - Mach/GPU: {s}", .{@tagName(status.mach_gpu)});
    std.log.info("  - CUDA: {s}", .{@tagName(status.cuda)});
    std.log.info("  - SIMD: {s}", .{@tagName(status.simd)});

    // Run SIMD benchmarks
    if (available_libs.simd) {
        std.log.info("⚡ Running SIMD Performance Benchmarks...", .{});
        library_manager.runSIMDBenchmarks() catch |err| {
            std.log.warn("❌ SIMD benchmarks failed: {}", .{err});
        };
    }

    // Demonstrate SIMD operations
    try demonstrateSIMDOperations();

    // Demonstrate advanced graphics operations
    try demonstrateAdvancedGraphics();

    // Demonstrate compute operations
    try demonstrateComputeOperations();

    std.log.info("🎉 Enhanced GPU Demo Complete!", .{});
}

fn demonstrateSIMDOperations() !void {
    std.log.info("🧮 Demonstrating SIMD Operations", .{});

    // Vector math operations
    const vec1 = gpu.VectorTypes.Vec4f{ 1.0, 2.0, 3.0, 4.0 };
    const vec2 = gpu.VectorTypes.Vec4f{ 5.0, 6.0, 7.0, 8.0 };

    const sum = gpu.SIMDMath.add(vec1, vec2);
    const dot_product = gpu.SIMDMath.dot(vec1, vec2);
    const length = gpu.SIMDMath.length(vec1);
    const normalized = gpu.SIMDMath.normalize(vec1);

    std.log.info("  - Vector Addition: {any}", .{sum});
    std.log.info("  - Dot Product: {d:.2}", .{dot_product});
    std.log.info("  - Vector Length: {d:.2}", .{length});
    std.log.info("  - Normalized Vector: {any}", .{normalized});

    // Matrix operations
    const matrix = gpu.VectorTypes.Mat4x4f{
        gpu.VectorTypes.Vec4f{ 1, 0, 0, 0 },
        gpu.VectorTypes.Vec4f{ 0, 1, 0, 0 },
        gpu.VectorTypes.Vec4f{ 0, 0, 1, 0 },
        gpu.VectorTypes.Vec4f{ 0, 0, 0, 1 },
    };
    const transformed = gpu.SIMDMath.mat4MulVec4(matrix, vec1);
    std.log.info("  - Matrix-Vector Transform: {any}", .{transformed});
}

fn demonstrateAdvancedGraphics() !void {
    std.log.info("🎨 Demonstrating Advanced Graphics Operations", .{});

    // Color space conversions
    const rgb_color = gpu.VectorTypes.Vec3f{ 0.8, 0.2, 0.4 };
    const hsv_color = gpu.SIMDGraphics.rgbToHsv(rgb_color);
    const back_to_rgb = gpu.SIMDGraphics.hsvToRgb(hsv_color);

    std.log.info("  - RGB Color: {any}", .{rgb_color});
    std.log.info("  - HSV Color: {any}", .{hsv_color});
    std.log.info("  - Back to RGB: {any}", .{back_to_rgb});

    // Color blending
    const src_color = gpu.VectorTypes.Vec4f{ 1.0, 0.0, 0.0, 0.5 }; // Red with 50% alpha
    const dst_color = gpu.VectorTypes.Vec4f{ 0.0, 0.0, 1.0, 1.0 }; // Blue
    const blended = gpu.SIMDGraphics.blendColors(src_color, dst_color);

    std.log.info("  - Source Color: {any}", .{src_color});
    std.log.info("  - Destination Color: {any}", .{dst_color});
    std.log.info("  - Blended Color: {any}", .{blended});

    // Tone mapping
    const hdr_color = gpu.VectorTypes.Vec3f{ 2.5, 1.8, 0.9 };
    const reinhard_mapped = gpu.SIMDGraphics.toneMapReinhard(hdr_color);
    const aces_mapped = gpu.SIMDGraphics.toneMapACES(hdr_color);

    std.log.info("  - HDR Color: {any}", .{hdr_color});
    std.log.info("  - Reinhard Mapped: {any}", .{reinhard_mapped});
    std.log.info("  - ACES Mapped: {any}", .{aces_mapped});
}

fn demonstrateComputeOperations() !void {
    std.log.info("⚡ Demonstrating Compute Operations", .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const array_size = 1000;
    const a = try allocator.alloc(f32, array_size);
    const b = try allocator.alloc(f32, array_size);
    const result = try allocator.alloc(f32, array_size);
    defer allocator.free(a);
    defer allocator.free(b);
    defer allocator.free(result);

    // Initialize arrays
    for (a, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i % 100)) / 100.0;
    }
    for (b, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt((i + 50) % 100)) / 100.0;
    }

    // SIMD array operations
    gpu.SIMDCompute.addArrays(a, b, result);
    const sum = gpu.SIMDCompute.sumArray(result);
    const dot = gpu.SIMDCompute.dotProduct(a, b);

    std.log.info("  - Array Size: {}", .{array_size});
    std.log.info("  - Array Sum: {d:.2}", .{sum});
    std.log.info("  - Dot Product: {d:.2}", .{dot});

    // Show first few results
    std.log.info("  - First 5 Results: {any}", .{result[0..5]});
}
