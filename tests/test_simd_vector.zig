const std = @import("std");
const testing = std.testing;

// Local SIMD implementations for testing (since we can't import outside module)
const SIMD_WIDTH = 4;
const F32Vector = @Vector(SIMD_WIDTH, f32);

fn distanceSquaredSIMD(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var sum: f32 = 0.0;
    var i: usize = 0;
    while (i + SIMD_WIDTH <= a.len) : (i += SIMD_WIDTH) {
        const va: F32Vector = a[i .. i + SIMD_WIDTH][0..SIMD_WIDTH].*;
        const vb: F32Vector = b[i .. i + SIMD_WIDTH][0..SIMD_WIDTH].*;
        const diff = va - vb;
        const squared = diff * diff;
        sum += @reduce(.Add, squared);
    }
    while (i < a.len) : (i += 1) {
        const diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

fn dotProductSIMD(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var sum: f32 = 0.0;
    var i: usize = 0;
    while (i + SIMD_WIDTH <= a.len) : (i += SIMD_WIDTH) {
        const va: F32Vector = a[i .. i + SIMD_WIDTH][0..SIMD_WIDTH].*;
        const vb: F32Vector = b[i .. i + SIMD_WIDTH][0..SIMD_WIDTH].*;
        const product = va * vb;
        sum += @reduce(.Add, product);
    }
    while (i < a.len) : (i += 1) {
        sum += a[i] * b[i];
    }
    return sum;
}

fn addVectorsSIMD(a: []const f32, b: []const f32, result: []f32) void {
    std.debug.assert(a.len == b.len and b.len == result.len);
    var i: usize = 0;
    while (i + SIMD_WIDTH <= a.len) : (i += SIMD_WIDTH) {
        const va: F32Vector = a[i .. i + SIMD_WIDTH][0..SIMD_WIDTH].*;
        const vb: F32Vector = b[i .. i + SIMD_WIDTH][0..SIMD_WIDTH].*;
        const sum = va + vb;
        result[i .. i + SIMD_WIDTH][0..SIMD_WIDTH].* = sum;
    }
    while (i < a.len) : (i += 1) {
        result[i] = a[i] + b[i];
    }
}

fn normalizeSIMD(vector: []f32) void {
    var sum_squares: f32 = 0.0;
    var i: usize = 0;
    while (i + SIMD_WIDTH <= vector.len) : (i += SIMD_WIDTH) {
        const v: F32Vector = vector[i .. i + SIMD_WIDTH][0..SIMD_WIDTH].*;
        const squared = v * v;
        sum_squares += @reduce(.Add, squared);
    }
    while (i < vector.len) : (i += 1) {
        sum_squares += vector[i] * vector[i];
    }
    const magnitude = @sqrt(sum_squares);
    if (magnitude == 0.0) return;
    const inv_magnitude = 1.0 / magnitude;
    const splat_inv_mag = @as(F32Vector, @splat(inv_magnitude));
    i = 0;
    while (i + SIMD_WIDTH <= vector.len) : (i += SIMD_WIDTH) {
        const v: F32Vector = vector[i .. i + SIMD_WIDTH][0..SIMD_WIDTH].*;
        const normalized = v * splat_inv_mag;
        vector[i .. i + SIMD_WIDTH][0..SIMD_WIDTH].* = normalized;
    }
    while (i < vector.len) : (i += 1) {
        vector[i] *= inv_magnitude;
    }
}

fn scaleVectorSIMD(vector: []const f32, scalar: f32, result: []f32) void {
    std.debug.assert(vector.len == result.len);
    const splat_scalar = @as(F32Vector, @splat(scalar));
    var i: usize = 0;
    while (i + SIMD_WIDTH <= vector.len) : (i += SIMD_WIDTH) {
        const v: F32Vector = vector[i .. i + SIMD_WIDTH][0..SIMD_WIDTH].*;
        const scaled = v * splat_scalar;
        result[i .. i + SIMD_WIDTH][0..SIMD_WIDTH].* = scaled;
    }
    while (i < vector.len) : (i += 1) {
        result[i] = vector[i] * scalar;
    }
}

fn cosineSimilaritySIMD(a: []const f32, b: []const f32) f32 {
    const dot = dotProductSIMD(a, b);
    const norm_a = @sqrt(dotProductSIMD(a, a));
    const norm_b = @sqrt(dotProductSIMD(b, b));
    if (norm_a == 0.0 or norm_b == 0.0) return 0.0;
    return dot / (norm_a * norm_b);
}

test "SIMD vector creation and basic operations" {
    // Test basic vector operations
    const vec1 = @Vector(4, f32){ 1.0, 2.0, 3.0, 4.0 };

    try testing.expectEqual(@as(f32, 1.0), vec1[0]);
    try testing.expectEqual(@as(f32, 4.0), vec1[3]);
}

test "SIMD vector addition" {
    const vec1 = @Vector(4, f32){ 1.0, 2.0, 3.0, 4.0 };
    const vec2 = @Vector(4, f32){ 5.0, 6.0, 7.0, 8.0 };

    const result = vec1 + vec2;

    try testing.expectEqual(@as(f32, 6.0), result[0]);
    try testing.expectEqual(@as(f32, 8.0), result[1]);
    try testing.expectEqual(@as(f32, 10.0), result[2]);
    try testing.expectEqual(@as(f32, 12.0), result[3]);
}

test "SIMD vector dot product" {
    const vec1 = @Vector(4, f32){ 1.0, 2.0, 3.0, 4.0 };
    const vec2 = @Vector(4, f32){ 2.0, 3.0, 4.0, 5.0 };

    // Calculate dot product: (1*2) + (2*3) + (3*4) + (4*5) = 2 + 6 + 12 + 20 = 40
    const product = vec1 * vec2;
    const dot_product = @reduce(.Add, product);

    try testing.expectEqual(@as(f32, 40.0), dot_product);
}

test "SIMD vector magnitude" {
    const vec = @Vector(4, f32){ 3.0, 4.0, 0.0, 0.0 };

    const squared = vec * vec;
    const sum_squares = @reduce(.Add, squared);
    const magnitude = @sqrt(sum_squares);

    try testing.expectEqual(@as(f32, 5.0), magnitude);
}

test "SIMD module: distance calculation" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const b = [_]f32{ 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };

    const dist = distanceSquaredSIMD(&a, &b);
    try testing.expectEqual(@as(f32, 8.0), dist); // Each element differs by 1
}

test "SIMD module: dot product" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 2.0, 3.0, 4.0, 5.0 };

    const dot = dotProductSIMD(&a, &b);
    try testing.expectEqual(@as(f32, 40.0), dot); // 1*2 + 2*3 + 3*4 + 4*5 = 40
}

test "SIMD module: vector normalization" {
    const allocator = testing.allocator;
    var vector = try allocator.alloc(f32, 4);
    defer allocator.free(vector);

    vector[0] = 3.0;
    vector[1] = 4.0;
    vector[2] = 0.0;
    vector[3] = 0.0;

    normalizeSIMD(vector);

    // Original magnitude was 5.0, so normalized should be [0.6, 0.8, 0.0, 0.0]
    try testing.expectApproxEqAbs(@as(f32, 0.6), vector[0], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.8), vector[1], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.0), vector[2], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.0), vector[3], 0.001);
}

test "SIMD module: vector addition" {
    const allocator = testing.allocator;
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 5.0, 6.0, 7.0, 8.0 };

    const result = try allocator.alloc(f32, 4);
    defer allocator.free(result);

    addVectorsSIMD(&a, &b, result);

    try testing.expectEqual(@as(f32, 6.0), result[0]);
    try testing.expectEqual(@as(f32, 8.0), result[1]);
    try testing.expectEqual(@as(f32, 10.0), result[2]);
    try testing.expectEqual(@as(f32, 12.0), result[3]);
}

test "SIMD module: vector scaling" {
    const allocator = testing.allocator;
    const vector = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const scalar: f32 = 2.5;

    const result = try allocator.alloc(f32, 4);
    defer allocator.free(result);

    scaleVectorSIMD(&vector, scalar, result);

    try testing.expectEqual(@as(f32, 2.5), result[0]);
    try testing.expectEqual(@as(f32, 5.0), result[1]);
    try testing.expectEqual(@as(f32, 7.5), result[2]);
    try testing.expectEqual(@as(f32, 10.0), result[3]);
}

test "SIMD module: cosine similarity" {
    // Test identical vectors (should be 1.0)
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    const sim_identical = cosineSimilaritySIMD(&a, &b);
    try testing.expectApproxEqAbs(@as(f32, 1.0), sim_identical, 0.001);

    // Test orthogonal vectors (should be 0.0)
    const c = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const d = [_]f32{ 0.0, 1.0, 0.0, 0.0 };

    const sim_orthogonal = cosineSimilaritySIMD(&c, &d);
    try testing.expectApproxEqAbs(@as(f32, 0.0), sim_orthogonal, 0.001);
}

test "SIMD performance comparison" {
    const allocator = testing.allocator;
    const size = 1000;

    // Create test data
    var vec1 = try allocator.alloc(f32, size);
    defer allocator.free(vec1);
    var vec2 = try allocator.alloc(f32, size);
    defer allocator.free(vec2);
    var result_scalar = try allocator.alloc(f32, size);
    defer allocator.free(result_scalar);
    const result_simd = try allocator.alloc(f32, size);
    defer allocator.free(result_simd);

    // Initialize vectors
    for (0..size) |i| {
        vec1[i] = @floatFromInt(i);
        vec2[i] = @floatFromInt(i * 2);
    }

    // Scalar addition
    var timer = try std.time.Timer.start();
    for (0..size) |i| {
        result_scalar[i] = vec1[i] + vec2[i];
    }
    const scalar_time = timer.read();

    // SIMD addition using our module
    timer.reset();
    addVectorsSIMD(vec1, vec2, result_simd);
    const simd_time = timer.read();

    // Verify results are identical
    for (0..size) |j| {
        try testing.expectEqual(result_scalar[j], result_simd[j]);
    }

    std.debug.print("Scalar time: {}ns, SIMD time: {}ns, Speedup: {d:.2}x\n", .{ scalar_time, simd_time, @as(f32, @floatFromInt(scalar_time)) / @as(f32, @floatFromInt(simd_time)) });
}
