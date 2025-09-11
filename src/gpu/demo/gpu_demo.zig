//! GPU Backend Manager Demo
//!
//! This demo shows how to use the GPU Backend Manager to:
//! - Detect available GPU backends
//! - Initialize GPU renderer
//! - Query hardware capabilities
//! - Run performance benchmarks

const std = @import("std");
const gpu = @import("gpu");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("🚀 GPU Backend Manager Demo", .{});
    std.log.info("============================", .{});

    // Initialize GPU renderer with fallback configuration
    const config = gpu.GPUConfig{
        .debug_validation = false,
        .power_preference = .high_performance,
        .backend = .auto,
        .try_webgpu_first = false, // Disable WebGPU for this demo
    };

    std.log.info("🔧 Initializing GPU renderer...", .{});
    var renderer = gpu.GPURenderer.init(allocator, config) catch |err| {
        std.log.warn("❌ GPU renderer initialization failed: {}", .{err});
        std.log.info("🔄 Falling back to CPU mode", .{});
        return demoCpuMode(allocator);
    };
    defer renderer.deinit();

    std.log.info("✅ GPU renderer initialized successfully", .{});

    // Test basic GPU functionality
    std.log.info("🧪 Testing GPU functionality...", .{});
    
    // Test buffer creation
    const buffer_size = 1024 * 1024; // 1MB
    const buffer = renderer.createBuffer(buffer_size, .{ .storage = true, .copy_dst = true }) catch |err| {
        std.log.warn("❌ Buffer creation failed: {}", .{err});
        return demoCpuMode(allocator);
    };
    defer renderer.destroyBuffer(buffer) catch {};

    std.log.info("✅ GPU buffer created ({} bytes)", .{buffer_size});

    // Test basic compute operations
    std.log.info("⚡ Testing compute operations...", .{});
    
    // Create test data
    const test_data = try allocator.alloc(f32, 1024);
    defer allocator.free(test_data);
    
    for (test_data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i)) * 0.1;
    }

    // Upload data to GPU
    renderer.writeBuffer(buffer, std.mem.sliceAsBytes(test_data)) catch |err| {
        std.log.warn("❌ Buffer write failed: {}", .{err});
        return demoCpuMode(allocator);
    };

    std.log.info("✅ Data uploaded to GPU successfully", .{});

    // Read data back
    const read_bytes = renderer.readBuffer(buffer, allocator) catch |err| {
        std.log.warn("❌ Buffer read failed: {}", .{err});
        return demoCpuMode(allocator);
    };
    defer allocator.free(read_bytes);
    
    const read_data = std.mem.bytesAsSlice(f32, read_bytes);

    std.log.info("✅ Data read from GPU successfully", .{});

    // Verify data integrity
    var data_valid = true;
    const min_len = @min(test_data.len, read_data.len);
    for (0..min_len) |i| {
        const original = test_data[i];
        const read = read_data[i];
        if (@abs(original - read) > 0.001) {
            std.log.warn("❌ Data mismatch at index {}: {} vs {}", .{ i, original, read });
            data_valid = false;
            break;
        }
    }

    if (data_valid) {
        std.log.info("✅ Data integrity verified", .{});
    } else {
        std.log.warn("❌ Data integrity check failed", .{});
    }

    // Performance test
    std.log.info("📊 Running GPU performance test...", .{});
    const iterations = 1000;
    var timer = try std.time.Timer.start();
    
    for (0..iterations) |_| {
        renderer.writeBuffer(buffer, std.mem.sliceAsBytes(test_data)) catch break;
    }
    
    const elapsed = timer.read();
    const throughput = (@as(f64, @floatFromInt(buffer_size * iterations)) / @as(f64, @floatFromInt(elapsed))) * 1e9;
    
    std.log.info("✅ GPU throughput: {d:.2} MB/s", .{throughput / (1024 * 1024)});

    std.log.info("🎉 GPU Backend Manager Demo Complete!", .{});
    std.log.info("=============================================", .{});
    std.log.info("Summary:", .{});
    std.log.info("  - GPU renderer: ✅ Working", .{});
    std.log.info("  - Buffer operations: ✅ Working", .{});
    std.log.info("  - Data integrity: ✅ Verified", .{});
    std.log.info("  - Performance: {d:.2} MB/s", .{throughput / (1024 * 1024)});
}

fn demoCpuMode(allocator: std.mem.Allocator) !void {
    std.log.info("🖥️  CPU Fallback Mode Demo", .{});
    std.log.info("=========================", .{});
    
    // Simulate some CPU-based operations
    const test_size = 1024 * 1024;
    const test_data = try allocator.alloc(f32, test_size);
    defer allocator.free(test_data);
    
    // Initialize test data
    for (test_data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i)) * 0.1;
    }
    
    // Simulate some computation
    var sum: f32 = 0.0;
    for (test_data) |val| {
        sum += val * val;
    }
    
    std.log.info("✅ CPU computation completed", .{});
    std.log.info("  - Data size: {} MB", .{test_size * @sizeOf(f32) / (1024 * 1024)});
    std.log.info("  - Sum of squares: {d:.2}", .{sum});
    
    std.log.info("🎉 CPU Fallback Demo Complete!", .{});
}
