//! AI/ML Kernel Benchmarks
//!
//! Core computational kernels for AI/ML inference:
//! - GEMM (General Matrix Multiplication)
//! - Activation functions (ReLU, GELU, SiLU, Softmax)
//! - Normalization (LayerNorm, RMSNorm)

const std = @import("std");
const framework = @import("../../system/framework.zig");
const core = @import("../../core/mod.zig");

// ============================================================================
// Activation Functions
// ============================================================================

/// ReLU activation: max(0, x)
pub fn relu(x: f32) f32 {
    return @max(0, x);
}

/// Apply ReLU to a vector
pub fn reluVector(input: []const f32, output: []f32) void {
    for (input, output) |x, *o| {
        o.* = relu(x);
    }
}

/// SIMD ReLU
pub fn simdRelu(comptime N: usize, input: []const f32, output: []f32) void {
    const Vec = @Vector(N, f32);
    const zero: Vec = @splat(0);
    var i: usize = 0;

    while (i + N <= input.len) : (i += N) {
        const x: Vec = input[i..][0..N].*;
        output[i..][0..N].* = @max(zero, x);
    }

    while (i < input.len) : (i += 1) {
        output[i] = relu(input[i]);
    }
}

/// GELU activation (approximate): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu(x: f32) f32 {
    const sqrt_2_over_pi: f32 = 0.7978845608;
    const coeff: f32 = 0.044715;
    const tanh_arg = sqrt_2_over_pi * (x + coeff * x * x * x);
    return 0.5 * x * (1.0 + std.math.tanh(tanh_arg));
}

/// Apply GELU to a vector
pub fn geluVector(input: []const f32, output: []f32) void {
    for (input, output) |x, *o| {
        o.* = gelu(x);
    }
}

/// SiLU (Swish) activation: x * sigmoid(x) = x / (1 + exp(-x))
pub fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

/// Apply SiLU to a vector
pub fn siluVector(input: []const f32, output: []f32) void {
    for (input, output) |x, *o| {
        o.* = silu(x);
    }
}

/// Softmax: exp(x_i) / sum(exp(x_j))
pub fn softmax(input: []const f32, output: []f32) void {
    // Find max for numerical stability
    var max_val: f32 = input[0];
    for (input[1..]) |x| {
        max_val = @max(max_val, x);
    }

    // Compute exp and sum
    var sum: f32 = 0;
    for (input, output) |x, *o| {
        o.* = @exp(x - max_val);
        sum += o.*;
    }

    // Normalize
    for (output) |*o| {
        o.* /= sum;
    }
}

// ============================================================================
// Normalization
// ============================================================================

/// Layer Normalization
pub fn layerNorm(input: []const f32, gamma: []const f32, beta: []const f32, output: []f32, eps: f32) void {
    // Compute mean
    var mean: f32 = 0;
    for (input) |x| {
        mean += x;
    }
    mean /= @floatFromInt(input.len);

    // Compute variance
    var variance: f32 = 0;
    for (input) |x| {
        const diff = x - mean;
        variance += diff * diff;
    }
    variance /= @floatFromInt(input.len);

    // Normalize and scale
    const inv_std = 1.0 / @sqrt(variance + eps);
    for (input, gamma, beta, output) |x, g, b, *o| {
        o.* = (x - mean) * inv_std * g + b;
    }
}

/// RMS Normalization (used in LLaMA, etc.)
pub fn rmsNorm(input: []const f32, gamma: []const f32, output: []f32, eps: f32) void {
    // Compute RMS
    var sum_sq: f32 = 0;
    for (input) |x| {
        sum_sq += x * x;
    }
    const rms = @sqrt(sum_sq / @as(f32, @floatFromInt(input.len)) + eps);

    // Normalize and scale
    for (input, gamma, output) |x, g, *o| {
        o.* = (x / rms) * g;
    }
}

// ============================================================================
// Matrix Operations (GEMM)
// ============================================================================

/// Naive GEMM: C = alpha * A @ B + beta * C
pub fn gemm(
    A: []const f32,
    B: []const f32,
    C: []f32,
    M: usize,
    N: usize,
    K: usize,
    alpha: f32,
    beta: f32,
) void {
    for (0..M) |i| {
        for (0..N) |j| {
            var sum: f32 = 0;
            for (0..K) |k| {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

/// Tiled GEMM for better cache utilization
pub fn tiledGemm(
    A: []const f32,
    B: []const f32,
    C: []f32,
    M: usize,
    N: usize,
    K: usize,
    tile_size: usize,
) void {
    @memset(C, 0);

    var i: usize = 0;
    while (i < M) : (i += tile_size) {
        var j: usize = 0;
        while (j < N) : (j += tile_size) {
            var k: usize = 0;
            while (k < K) : (k += tile_size) {
                const i_end = @min(i + tile_size, M);
                const j_end = @min(j + tile_size, N);
                const k_end = @min(k + tile_size, K);

                var ii = i;
                while (ii < i_end) : (ii += 1) {
                    var kk = k;
                    while (kk < k_end) : (kk += 1) {
                        const a_val = A[ii * K + kk];
                        var jj = j;
                        while (jj < j_end) : (jj += 1) {
                            C[ii * N + jj] += a_val * B[kk * N + jj];
                        }
                    }
                }
            }
        }
    }
}

/// SIMD GEMM
pub fn simdGemm(comptime VEC_SIZE: usize, A: []const f32, B: []const f32, C: []f32, M: usize, N: usize, K: usize) void {
    const Vec = @Vector(VEC_SIZE, f32);
    @memset(C, 0);

    for (0..M) |i| {
        for (0..K) |k| {
            const a_val: Vec = @splat(A[i * K + k]);
            var j: usize = 0;

            while (j + VEC_SIZE <= N) : (j += VEC_SIZE) {
                const b_vec: Vec = B[k * N + j ..][0..VEC_SIZE].*;
                const c_vec: Vec = C[i * N + j ..][0..VEC_SIZE].*;
                C[i * N + j ..][0..VEC_SIZE].* = c_vec + a_val * b_vec;
            }

            while (j < N) : (j += 1) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

// ============================================================================
// Attention Computation
// ============================================================================

/// Scaled dot-product attention (single head)
pub fn scaledDotProductAttention(
    allocator: std.mem.Allocator,
    Q: []const f32, // [seq_len, head_dim]
    K: []const f32, // [seq_len, head_dim]
    V: []const f32, // [seq_len, head_dim]
    output: []f32, // [seq_len, head_dim]
    seq_len: usize,
    head_dim: usize,
) !void {
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    // Compute attention scores: Q @ K^T
    const scores = try allocator.alloc(f32, seq_len * seq_len);
    defer allocator.free(scores);

    for (0..seq_len) |i| {
        for (0..seq_len) |j| {
            var dot: f32 = 0;
            for (0..head_dim) |d| {
                dot += Q[i * head_dim + d] * K[j * head_dim + d];
            }
            scores[i * seq_len + j] = dot * scale;
        }
    }

    // Apply softmax row-wise
    const probs = try allocator.alloc(f32, seq_len * seq_len);
    defer allocator.free(probs);

    for (0..seq_len) |i| {
        softmax(scores[i * seq_len ..][0..seq_len], probs[i * seq_len ..][0..seq_len]);
    }

    // Compute output: probs @ V
    for (0..seq_len) |i| {
        for (0..head_dim) |d| {
            var sum: f32 = 0;
            for (0..seq_len) |j| {
                sum += probs[i * seq_len + j] * V[j * head_dim + d];
            }
            output[i * head_dim + d] = sum;
        }
    }
}

// ============================================================================
// Embedding Operations
// ============================================================================

/// Embedding lookup
pub fn embeddingLookup(
    embeddings: []const f32, // [vocab_size, hidden]
    indices: []const u32,
    output: []f32, // [seq_len, hidden]
    hidden: usize,
) void {
    for (indices, 0..) |idx, i| {
        const src_offset = idx * hidden;
        const dst_offset = i * hidden;
        @memcpy(output[dst_offset..][0..hidden], embeddings[src_offset..][0..hidden]);
    }
}

// ============================================================================
// Benchmark Runner
// ============================================================================

/// Run all kernel benchmarks
pub fn runKernelBenchmarks(allocator: std.mem.Allocator, config: core.config.AIBenchConfig) !void {
    var runner = framework.BenchmarkRunner.init(allocator);
    defer runner.deinit();

    std.debug.print("\n[Activation Functions]\n", .{});
    for (config.activation_sizes[0..@min(3, config.activation_sizes.len)]) |size| {
        const input = try allocator.alloc(f32, size);
        defer allocator.free(input);
        const output = try allocator.alloc(f32, size);
        defer allocator.free(output);

        var prng = std.Random.DefaultPrng.init(42);
        for (input) |*x| {
            x.* = prng.random().float(f32) * 2 - 1;
        }

        // ReLU
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "relu_{d}", .{size}) catch "relu";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "ai/activation",
                    .bytes_per_op = size * @sizeOf(f32) * 2,
                    .warmup_iterations = config.warmup_iterations,
                    .min_time_ns = config.min_time_ns,
                },
                struct {
                    fn bench(inp: []const f32, out: []f32) void {
                        reluVector(inp, out);
                    }
                }.bench,
                .{ input, output },
            );

            std.debug.print("  {s}: {d:.0} ops/sec ({d:.2} GB/s)\n", .{
                name,
                result.stats.opsPerSecond(),
                result.stats.throughputMBps(size * @sizeOf(f32) * 2) / 1024.0,
            });
        }

        // SIMD ReLU
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "relu_simd_{d}", .{size}) catch "relu_simd";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "ai/activation",
                    .bytes_per_op = size * @sizeOf(f32) * 2,
                    .warmup_iterations = config.warmup_iterations,
                    .min_time_ns = config.min_time_ns,
                },
                struct {
                    fn bench(inp: []const f32, out: []f32) void {
                        simdRelu(8, inp, out);
                    }
                }.bench,
                .{ input, output },
            );

            std.debug.print("  {s}: {d:.0} ops/sec ({d:.2} GB/s)\n", .{
                name,
                result.stats.opsPerSecond(),
                result.stats.throughputMBps(size * @sizeOf(f32) * 2) / 1024.0,
            });
        }

        // GELU
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "gelu_{d}", .{size}) catch "gelu";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "ai/activation",
                    .warmup_iterations = config.warmup_iterations,
                    .min_time_ns = config.min_time_ns,
                },
                struct {
                    fn bench(inp: []const f32, out: []f32) void {
                        geluVector(inp, out);
                    }
                }.bench,
                .{ input, output },
            );

            std.debug.print("  {s}: {d:.0} ops/sec\n", .{ name, result.stats.opsPerSecond() });
        }

        // SiLU
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "silu_{d}", .{size}) catch "silu";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "ai/activation",
                    .warmup_iterations = config.warmup_iterations,
                    .min_time_ns = config.min_time_ns,
                },
                struct {
                    fn bench(inp: []const f32, out: []f32) void {
                        siluVector(inp, out);
                    }
                }.bench,
                .{ input, output },
            );

            std.debug.print("  {s}: {d:.0} ops/sec\n", .{ name, result.stats.opsPerSecond() });
        }

        // Softmax
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "softmax_{d}", .{size}) catch "softmax";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "ai/activation",
                    .warmup_iterations = config.warmup_iterations,
                    .min_time_ns = config.min_time_ns,
                },
                struct {
                    fn bench(inp: []const f32, out: []f32) void {
                        softmax(inp, out);
                    }
                }.bench,
                .{ input, output },
            );

            std.debug.print("  {s}: {d:.0} ops/sec\n", .{ name, result.stats.opsPerSecond() });
        }
    }

    // Normalization benchmarks
    std.debug.print("\n[Normalization]\n", .{});
    for (config.hidden_sizes[0..@min(3, config.hidden_sizes.len)]) |hidden| {
        const input = try allocator.alloc(f32, hidden);
        defer allocator.free(input);
        const gamma = try allocator.alloc(f32, hidden);
        defer allocator.free(gamma);
        const beta = try allocator.alloc(f32, hidden);
        defer allocator.free(beta);
        const output = try allocator.alloc(f32, hidden);
        defer allocator.free(output);

        @memset(gamma, 1.0);
        @memset(beta, 0.0);
        var prng = std.Random.DefaultPrng.init(42);
        for (input) |*x| x.* = prng.random().float(f32);

        // LayerNorm
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "layernorm_{d}", .{hidden}) catch "ln";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "ai/normalization",
                    .warmup_iterations = config.warmup_iterations,
                    .min_time_ns = config.min_time_ns,
                },
                struct {
                    fn bench(inp: []const f32, g: []const f32, b: []const f32, out: []f32) void {
                        layerNorm(inp, g, b, out, 1e-5);
                    }
                }.bench,
                .{ input, gamma, beta, output },
            );

            std.debug.print("  {s}: {d:.0} ops/sec\n", .{ name, result.stats.opsPerSecond() });
        }

        // RMSNorm
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "rmsnorm_{d}", .{hidden}) catch "rms";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "ai/normalization",
                    .warmup_iterations = config.warmup_iterations,
                    .min_time_ns = config.min_time_ns,
                },
                struct {
                    fn bench(inp: []const f32, g: []const f32, out: []f32) void {
                        rmsNorm(inp, g, out, 1e-5);
                    }
                }.bench,
                .{ input, gamma, output },
            );

            std.debug.print("  {s}: {d:.0} ops/sec\n", .{ name, result.stats.opsPerSecond() });
        }
    }

    // GEMM benchmarks
    std.debug.print("\n[Matrix Multiplication (GEMM)]\n", .{});
    for (config.matrix_sizes[0..@min(3, config.matrix_sizes.len)]) |size| {
        const A = try allocator.alloc(f32, size * size);
        defer allocator.free(A);
        const B = try allocator.alloc(f32, size * size);
        defer allocator.free(B);
        const C = try allocator.alloc(f32, size * size);
        defer allocator.free(C);

        var prng = std.Random.DefaultPrng.init(42);
        for (A) |*x| x.* = prng.random().float(f32);
        for (B) |*x| x.* = prng.random().float(f32);

        // Naive GEMM
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "gemm_naive_{d}", .{size}) catch "gemm_naive";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "ai/gemm",
                    .warmup_iterations = 10,
                    .min_time_ns = config.min_time_ns,
                    .max_iterations = 100,
                },
                struct {
                    fn bench(a: []const f32, b: []const f32, c: []f32, n: usize) void {
                        gemm(a, b, c, n, n, n, 1.0, 0.0);
                    }
                }.bench,
                .{ A, B, C, size },
            );

            const gflops = @as(f64, @floatFromInt(2 * size * size * size)) * result.stats.opsPerSecond() / 1e9;
            std.debug.print("  {s}: {d:.2} GFLOPS\n", .{ name, gflops });
        }

        // Tiled GEMM
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "gemm_tiled_{d}", .{size}) catch "gemm_tiled";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "ai/gemm",
                    .warmup_iterations = 10,
                    .min_time_ns = config.min_time_ns,
                    .max_iterations = 100,
                },
                struct {
                    fn bench(a: []const f32, b: []const f32, c: []f32, n: usize) void {
                        tiledGemm(a, b, c, n, n, n, 32);
                    }
                }.bench,
                .{ A, B, C, size },
            );

            const gflops = @as(f64, @floatFromInt(2 * size * size * size)) * result.stats.opsPerSecond() / 1e9;
            std.debug.print("  {s}: {d:.2} GFLOPS\n", .{ name, gflops });
        }

        // SIMD GEMM
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "gemm_simd_{d}", .{size}) catch "gemm_simd";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "ai/gemm",
                    .warmup_iterations = 10,
                    .min_time_ns = config.min_time_ns,
                    .max_iterations = 100,
                },
                struct {
                    fn bench(a: []const f32, b: []const f32, c: []f32, n: usize) void {
                        simdGemm(8, a, b, c, n, n, n);
                    }
                }.bench,
                .{ A, B, C, size },
            );

            const gflops = @as(f64, @floatFromInt(2 * size * size * size)) * result.stats.opsPerSecond() / 1e9;
            std.debug.print("  {s}: {d:.2} GFLOPS\n", .{ name, gflops });
        }
    }

    // Attention benchmarks
    std.debug.print("\n[Attention Computation]\n", .{});
    for (config.seq_lengths[0..@min(3, config.seq_lengths.len)]) |seq_len| {
        const head_dim: usize = 64;
        const Q = try allocator.alloc(f32, seq_len * head_dim);
        defer allocator.free(Q);
        const K = try allocator.alloc(f32, seq_len * head_dim);
        defer allocator.free(K);
        const V = try allocator.alloc(f32, seq_len * head_dim);
        defer allocator.free(V);
        const output = try allocator.alloc(f32, seq_len * head_dim);
        defer allocator.free(output);

        var prng = std.Random.DefaultPrng.init(42);
        for (Q) |*x| x.* = prng.random().float(f32) * 0.1;
        for (K) |*x| x.* = prng.random().float(f32) * 0.1;
        for (V) |*x| x.* = prng.random().float(f32) * 0.1;

        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "attention_s{d}_d{d}", .{ seq_len, head_dim }) catch "attn";

        const result = try runner.run(
            .{
                .name = name,
                .category = "ai/attention",
                .warmup_iterations = 10,
                .min_time_ns = config.min_time_ns,
                .max_iterations = 100,
            },
            struct {
                fn bench(a: std.mem.Allocator, q: []const f32, k: []const f32, v: []const f32, out: []f32, sl: usize, hd: usize) !void {
                    try scaledDotProductAttention(a, q, k, v, out, sl, hd);
                }
            }.bench,
            .{ allocator, Q, K, V, output, seq_len, head_dim },
        );

        std.debug.print("  {s}: {d:.0} ops/sec\n", .{ name, result.stats.opsPerSecond() });
    }

    std.debug.print("\n", .{});
    runner.printSummaryDebug();
}

// ============================================================================
// Tests
// ============================================================================

test "activation functions" {
    try std.testing.expectApproxEqAbs(@as(f32, 0), relu(-1), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1), relu(1), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0), gelu(0), 0.001);
    try std.testing.expect(silu(0) == 0);
}

test "softmax" {
    var input = [_]f32{ 1, 2, 3 };
    var output: [3]f32 = undefined;

    softmax(&input, &output);

    var sum: f32 = 0;
    for (output) |x| sum += x;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 0.001);

    try std.testing.expect(output[2] > output[1]);
    try std.testing.expect(output[1] > output[0]);
}

test "layer norm" {
    var input = [_]f32{ 1, 2, 3, 4 };
    var gamma = [_]f32{ 1, 1, 1, 1 };
    var beta = [_]f32{ 0, 0, 0, 0 };
    var output: [4]f32 = undefined;

    layerNorm(&input, &gamma, &beta, &output, 1e-5);

    // Mean should be approximately 0 after normalization
    var mean: f32 = 0;
    for (output) |x| mean += x;
    mean /= 4;
    try std.testing.expectApproxEqAbs(@as(f32, 0), mean, 0.01);
}

test "gemm" {
    const A = [_]f32{ 1, 2, 3, 4 };
    const B = [_]f32{ 1, 0, 0, 1 };
    var C: [4]f32 = undefined;

    gemm(&A, &B, &C, 2, 2, 2, 1.0, 0.0);

    // [[1,2],[3,4]] @ [[1,0],[0,1]] = [[1,2],[3,4]]
    try std.testing.expectApproxEqAbs(@as(f32, 1), C[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2), C[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 3), C[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4), C[3], 0.001);
}
