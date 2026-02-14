const std = @import("std");
const abi = @import("abi");

/// Comprehensive GPU backend benchmarking
/// Tests all supported backends: CUDA, Vulkan, Metal, OpenGL, WebGL2, WebGPU
/// Includes multi-GPU detection and per-device benchmarking
pub fn runGpuBackendBenchmarks(allocator: std.mem.Allocator) !void {
    std.debug.print("\n=== GPU Backend Detection & Benchmarking ===\n", .{});

    // Initialize framework with GPU enabled
    var framework = abi.init(allocator, abi.Config{
        .gpu = .{}, // Auto-detect best available backend
    }) catch |err| {
        std.debug.print("GPU initialization failed: {t}\n", .{err});
        std.debug.print("No GPU backends available\n", .{});
        return;
    };
    defer framework.deinit();

    if (!abi.gpu.moduleEnabled()) {
        std.debug.print("GPU module not enabled\n", .{});
        return;
    }

    // Detect available backends
    try detectAndBenchmarkBackends(allocator);

    // Multi-GPU detection
    try detectMultiGpu(allocator);

    // Per-backend performance tests
    try benchmarkAllBackends(allocator);
}

fn detectAndBenchmarkBackends(allocator: std.mem.Allocator) !void {
    std.debug.print("\n--- Backend Detection ---\n", .{});

    const backends = try abi.gpu.availableBackends(allocator);
    defer allocator.free(backends);

    if (backends.len == 0) {
        std.debug.print("No GPU backends detected\n", .{});
        return;
    }

    std.debug.print("Found {d} backend(s):\n", .{backends.len});
    for (backends) |backend| {
        std.debug.print("  - {t}\n", .{backend});
    }

    // Get current backend info
    const current_backend = abi.gpu.currentBackend();
    std.debug.print("\nActive backend: {t}\n", .{current_backend});
}

fn detectMultiGpu(allocator: std.mem.Allocator) !void {
    std.debug.print("\n--- Multi-GPU Detection ---\n", .{});

    const devices = try abi.gpu.listDevices(allocator);
    defer allocator.free(devices);

    if (devices.len == 0) {
        std.debug.print("No GPU devices found\n", .{});
        return;
    }

    std.debug.print("Found {d} GPU device(s):\n", .{devices.len});
    for (devices, 0..) |device, idx| {
        std.debug.print("\nDevice {d}:\n", .{idx});
        std.debug.print("  Name: {s}\n", .{device.name});
        std.debug.print("  Type: {t}\n", .{device.type});
        std.debug.print("  Memory: {d} MB\n", .{device.total_memory / (1024 * 1024)});
        std.debug.print("  Compute Units: {d}\n", .{device.compute_units});
    }
}

fn benchmarkAllBackends(allocator: std.mem.Allocator) !void {
    std.debug.print("\n--- Backend Performance Tests ---\n", .{});

    // Try each backend explicitly
    const backend_list = [_]abi.GpuBackend{
        .vulkan,
        .cuda,
        .metal,
        .opengl,
        .webgpu,
        .webgl2,
        .stdgpu, // CPU fallback
    };

    for (backend_list) |backend| {
        benchmarkBackend(allocator, backend) catch |err| {
            std.debug.print("{t}: Not available ({t})\n", .{ backend, err });
            continue;
        };
    }
}

fn benchmarkBackend(allocator: std.mem.Allocator, backend: abi.GpuBackend) !void {
    std.debug.print("\n{t} Backend:\n", .{backend});

    // Initialize with specific backend
    var fw = try abi.init(allocator, abi.Config{
        .gpu = .{ .backend = backend },
    });
    defer fw.deinit();

    // Vector addition benchmark
    const vec_size = 1024 * 1024;
    var vec_a = try allocator.alloc(f32, vec_size);
    defer allocator.free(vec_a);
    var vec_b = try allocator.alloc(f32, vec_size);
    defer allocator.free(vec_b);
    var result = try allocator.alloc(f32, vec_size);
    defer allocator.free(result);

    // Initialize data
    for (vec_a, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }
    for (vec_b, 0..) |*val, i| {
        val.* = @floatFromInt(i * 2);
    }

    // Warmup
    for (0..10) |_| {
        abi.simd.vectorAdd(vec_a, vec_b, result);
    }

    // Benchmark
    var timer = try abi.shared.time.Timer.start();
    const iterations: usize = 100;
    for (0..iterations) |_| {
        abi.simd.vectorAdd(vec_a, vec_b, result);
    }
    const elapsed = timer.read();

    const ops_per_sec = @as(f64, @floatFromInt(iterations * vec_size)) / (@as(f64, @floatFromInt(elapsed)) / 1_000_000_000.0);
    const gb_per_sec = (ops_per_sec * @as(f64, @sizeOf(f32))) / 1_000_000_000.0;

    std.debug.print("  Vector Add: {d:.2} GFLOPS, {d:.2} GB/s\n", .{ ops_per_sec / 1_000_000_000.0, gb_per_sec });

    // Matrix multiply benchmark (smaller for quick test)
    const mat_size = 256;
    try benchmarkMatrixMultiply(allocator, mat_size);
}

fn benchmarkMatrixMultiply(allocator: std.mem.Allocator, size: usize) !void {
    var mat_a = try allocator.alloc(f32, size * size);
    defer allocator.free(mat_a);
    var mat_b = try allocator.alloc(f32, size * size);
    defer allocator.free(mat_b);
    var result = try allocator.alloc(f32, size * size);
    defer allocator.free(result);

    // Initialize matrices
    for (mat_a, 0..) |*val, i| {
        val.* = @floatFromInt(i % 100);
    }
    for (mat_b, 0..) |*val, i| {
        val.* = @floatFromInt((i + 50) % 100);
    }

    // Warmup
    for (0..5) |_| {
        abi.simd.matrixMultiply(mat_a, mat_b, result, size, size, size);
    }

    // Benchmark
    var timer = try abi.shared.time.Timer.start();
    const iterations: usize = 10;
    for (0..iterations) |_| {
        abi.simd.matrixMultiply(mat_a, mat_b, result, size, size, size);
    }
    const elapsed = timer.read();

    const flops = 2 * size * size * size; // 2n³ for matrix multiply
    const total_flops = @as(f64, @floatFromInt(flops * iterations));
    const gflops = total_flops / (@as(f64, @floatFromInt(elapsed)) / 1_000_000_000.0) / 1_000_000_000.0;

    std.debug.print("  MatMul {d}x{d}: {d:.2} GFLOPS\n", .{ size, size, gflops });
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    try runGpuBackendBenchmarks(allocator);

    std.debug.print("\n=== GPU Backend Benchmarking Complete ===\n", .{});
}
