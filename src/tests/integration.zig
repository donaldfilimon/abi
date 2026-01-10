//! Integration tests for end-to-end functionality.
const std = @import("std");
const abi = @import("abi");

pub fn main(init: std.process.Init) !void {
    std.debug.print("Running integration tests...\n", .{});

    const allocator = init.gpa;

    var framework = try abi.init(allocator, abi.FrameworkOptions{
        .enable_ai = true,
        .enable_database = true,
        .enable_gpu = true,
    });
    defer abi.shutdown(&framework);

    std.debug.print("Framework initialized\n", .{});

    try testDatabaseOperations(allocator);
    try testVectorOperations(allocator);
    try testGPUOperations(allocator);

    std.debug.print("\n✅ All integration tests passed!\n", .{});
}

fn testDatabaseOperations(allocator: std.mem.Allocator) !void {
    std.debug.print("Test: Database operations\n", .{});

    const handle = try abi.database.createDatabase(allocator, "integration_test");
    defer abi.database.closeDatabase(&handle);

    const vec1 = &.{ 1.0, 0.0, 0.0 };
    const vec2 = &.{ 0.0, 1.0, 0.0 };
    const vec3 = &.{ 0.0, 0.0, 1.0 };

    try abi.database.insertVector(handle, 1, vec1, null);
    try abi.database.insertVector(handle, 2, vec2, null);
    try abi.database.insertVector(handle, 3, vec3, null);

    const query = &.{ 1.0, 0.0, 0.0 };
    const results = try abi.database.searchVectors(handle, query, 2);
    defer allocator.free(results);

    try std.testing.expectEqual(@as(usize, 2), results.len);
    try std.testing.expectEqual(@as(u64, 1), results[0].id);
    try std.testing.expect(results[0].score > 0.9);

    const stats = abi.database.getStats(&handle);
    try std.testing.expectEqual(@as(usize, 3), stats.count);
    try std.testing.expectEqual(@as(usize, 3), stats.dimension);

    std.debug.print("  ✅ Database operations passed\n", .{});
}

fn testVectorOperations() !void {
    std.debug.print("Test: Vector operations\n", .{});
    const simd = abi.simd;

    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var b = [_]f32{ 2.0, 3.0, 4.0, 5.0 };
    var result: [4]f32 = undefined;

    simd.vectorAdd(&a, &b, &result);
    try std.testing.expectEqual(@as(f32, 3.0), result[0]);
    try std.testing.expectEqual(@as(f32, 5.0), result[1]);
    try std.testing.expectEqual(@as(f32, 7.0), result[2]);
    try std.testing.expectEqual(@as(f32, 9.0), result[3]);

    const dot = simd.vectorDot(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 40.0), dot, 1e-6);

    const norm_a = simd.vectorL2Norm(&a);
    try std.testing.expectApproxEqAbs(@as(f32, 5.4772), norm_a, 1e-4);

    const similarity = simd.cosineSimilarity(&a, &a);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), similarity, 1e-6);

    std.debug.print("  ✅ Vector operations passed\n", .{});
}

fn testGPUOperations(allocator: std.mem.Allocator) !void {
    std.debug.print("Test: GPU operations\n", .{});

    if (!abi.gpu.moduleEnabled()) {
        std.debug.print("  ⚠️  GPU module not enabled, skipping GPU tests\n", .{});
        return;
    }

    try abi.gpu.ensureInitialized(allocator);

    const backends = try abi.gpu.availableBackends(allocator);
    defer allocator.free(backends);

    if (backends.len == 0) {
        std.debug.print("  ⚠️  No GPU backends available, skipping GPU tests\n", .{});
        return;
    }

    std.debug.print("  Available backends: {d}\n", .{backends.len});

    var kernels = try abi.gpu.createDefaultKernels(allocator);
    defer {
        for (kernels) |*kernel| {
            kernel.deinit(allocator);
        }
        allocator.free(kernels);
    }

    if (kernels.len == 0) {
        std.debug.print("  ⚠️  No kernels available, skipping kernel tests\n", .{});
        return;
    }

    for (kernels) |kernel_source| {
        std.debug.print("  Testing kernel: {s} on {s}\n", .{ kernel_source.name, abi.gpu.backendDisplayName(kernel_source.backend) });

        var compiled = try abi.gpu.compileKernel(allocator, kernel_source);
        defer compiled.deinit();

        if (std.mem.eql(u8, kernel_source.name, "vector_add")) {
            try testVectorAddKernel(allocator, &compiled);
        } else if (std.mem.eql(u8, kernel_source.name, "reduce_sum")) {
            try testReduceSumKernel(allocator, &compiled);
        }
    }

    std.debug.print("  ✅ GPU operations passed\n", .{});
}

fn testVectorAddKernel(allocator: std.mem.Allocator, compiled: *const abi.gpu.CompiledKernel) !void {
    const n: u32 = 1024;

    const input_a = try allocator.alloc(f32, n);
    defer allocator.free(input_a);
    const input_b = try allocator.alloc(f32, n);
    defer allocator.free(input_b);
    var output = try allocator.alloc(f32, n);
    defer allocator.free(output);

    for (0..n) |i| {
        input_a[i] = @floatFromInt(i);
        input_b[i] = @floatFromInt(n - i);
    }

    const config = abi.gpu.KernelConfig{
        .grid_dim = .{ (n + 255) / 256, 1, 1 },
        .block_dim = .{ 256, 1, 1 },
        .shared_memory_bytes = 0,
    };

    const args = [_]?*const anyopaque{
        input_a.ptr,
        input_b.ptr,
        output.ptr,
        &n,
    };

    try compiled.launch(allocator, config, &args);

    var all_correct = true;
    for (0..n) |i| {
        const expected = @as(f32, @floatFromInt(n));
        const actual = output[i];
        if (@abs(actual - expected) > 0.01) {
            all_correct = false;
            break;
        }
    }

    try std.testing.expect(all_correct);
}

fn testReduceSumKernel(allocator: std.mem.Allocator, compiled: *const abi.gpu.CompiledKernel) !void {
    const n: u32 = 256;

    const input = try allocator.alloc(f32, n);
    defer allocator.free(input);
    var output = try allocator.alloc(f32, 1);
    defer allocator.free(output);

    for (0..n) |i| {
        input[i] = @floatFromInt(i + 1);
    }

    const config = abi.gpu.KernelConfig{
        .grid_dim = .{ 1, 1, 1 },
        .block_dim = .{ 256, 1, 1 },
        .shared_memory_bytes = 256 * @sizeOf(f32),
    };

    const args = [_]?*const anyopaque{
        input.ptr,
        output.ptr,
        &n,
    };

    try compiled.launch(allocator, config, &args);

    const expected: f32 = @floatFromInt(n * (n + 1) / 2);
    try std.testing.expectApproxEqAbs(expected, output[0], 0.01);
}
