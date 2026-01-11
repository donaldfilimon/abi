//! AI/ML Inference Benchmarks
//!
//! Industry-standard benchmarks for AI/ML operations:
//! - Matrix multiplication (GEMM)
//! - Activation functions (ReLU, GELU, SiLU, Softmax)
//! - Layer normalization
//! - Attention computation
//! - Embedding lookup
//! - Tokenization throughput
//! - Quantization/dequantization
//! - Batch inference scaling
//! - Memory access patterns
//! - Transformer building blocks

const std = @import("std");
const framework = @import("framework.zig");

/// AI benchmark configuration
pub const AIBenchConfig = struct {
    /// Hidden dimensions (typical transformer sizes)
    hidden_sizes: []const usize = &.{ 256, 512, 768, 1024, 2048, 4096 },
    /// Sequence lengths
    seq_lengths: []const usize = &.{ 32, 64, 128, 256, 512, 1024 },
    /// Batch sizes
    batch_sizes: []const usize = &.{ 1, 2, 4, 8, 16, 32 },
    /// Attention head counts
    num_heads: []const usize = &.{ 8, 12, 16, 32 },
    /// Vocabulary sizes
    vocab_sizes: []const usize = &.{ 32000, 50257, 128256 },
};

// ============================================================================
// Activation Functions
// ============================================================================

/// ReLU activation
fn relu(x: f32) f32 {
    return @max(0, x);
}

fn reluVector(input: []const f32, output: []f32) void {
    for (input, output) |x, *o| {
        o.* = relu(x);
    }
}

fn simdRelu(comptime N: usize, input: []const f32, output: []f32) void {
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

/// GELU activation (approximate)
fn gelu(x: f32) f32 {
    const sqrt_2_over_pi: f32 = 0.7978845608;
    const coeff: f32 = 0.044715;
    const tanh_arg = sqrt_2_over_pi * (x + coeff * x * x * x);
    return 0.5 * x * (1.0 + std.math.tanh(tanh_arg));
}

fn geluVector(input: []const f32, output: []f32) void {
    for (input, output) |x, *o| {
        o.* = gelu(x);
    }
}

/// SiLU (Swish) activation
fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

fn siluVector(input: []const f32, output: []f32) void {
    for (input, output) |x, *o| {
        o.* = silu(x);
    }
}

/// Softmax
fn softmax(input: []const f32, output: []f32) void {
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
// Layer Normalization
// ============================================================================

fn layerNorm(input: []const f32, gamma: []const f32, beta: []const f32, output: []f32, eps: f32) void {
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

fn rmsNorm(input: []const f32, gamma: []const f32, output: []f32, eps: f32) void {
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
fn gemm(
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
fn tiledGemm(
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
                // Multiply tiles
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
fn simdGemm(comptime VEC_SIZE: usize, A: []const f32, B: []const f32, C: []f32, M: usize, N: usize, K: usize) void {
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

            // Handle remainder
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
fn scaledDotProductAttention(
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

/// Multi-head attention
fn multiHeadAttention(
    allocator: std.mem.Allocator,
    Q: []const f32, // [seq_len, hidden]
    K: []const f32,
    V: []const f32,
    output: []f32,
    seq_len: usize,
    hidden: usize,
    num_heads: usize,
) !void {
    const head_dim = hidden / num_heads;

    for (0..num_heads) |h| {
        const offset = h * head_dim;

        // Extract head slices
        const q_head = try allocator.alloc(f32, seq_len * head_dim);
        defer allocator.free(q_head);
        const k_head = try allocator.alloc(f32, seq_len * head_dim);
        defer allocator.free(k_head);
        const v_head = try allocator.alloc(f32, seq_len * head_dim);
        defer allocator.free(v_head);
        const out_head = try allocator.alloc(f32, seq_len * head_dim);
        defer allocator.free(out_head);

        for (0..seq_len) |s| {
            for (0..head_dim) |d| {
                q_head[s * head_dim + d] = Q[s * hidden + offset + d];
                k_head[s * head_dim + d] = K[s * hidden + offset + d];
                v_head[s * head_dim + d] = V[s * hidden + offset + d];
            }
        }

        try scaledDotProductAttention(allocator, q_head, k_head, v_head, out_head, seq_len, head_dim);

        // Copy to output
        for (0..seq_len) |s| {
            for (0..head_dim) |d| {
                output[s * hidden + offset + d] = out_head[s * head_dim + d];
            }
        }
    }
}

// ============================================================================
// Embedding Operations
// ============================================================================

fn embeddingLookup(
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
// Quantization
// ============================================================================

/// Quantize f32 to int8
fn quantizeToInt8(input: []const f32, output: []i8, scale: *f32) void {
    // Find absmax
    var absmax: f32 = 0;
    for (input) |x| {
        absmax = @max(absmax, @abs(x));
    }

    scale.* = if (absmax == 0) 1.0 else absmax / 127.0;

    for (input, output) |x, *o| {
        const quantized = @round(x / scale.*);
        o.* = @intFromFloat(std.math.clamp(quantized, -127, 127));
    }
}

/// Dequantize int8 to f32
fn dequantizeFromInt8(input: []const i8, output: []f32, scale: f32) void {
    for (input, output) |x, *o| {
        o.* = @as(f32, @floatFromInt(x)) * scale;
    }
}

/// Quantized GEMM (int8 with f32 accumulator)
fn quantizedGemm(
    A: []const i8,
    B: []const i8,
    C: []f32,
    M: usize,
    N: usize,
    K: usize,
    scale_a: f32,
    scale_b: f32,
) void {
    for (0..M) |i| {
        for (0..N) |j| {
            var acc: i32 = 0;
            for (0..K) |k| {
                acc += @as(i32, A[i * K + k]) * @as(i32, B[k * N + j]);
            }
            C[i * N + j] = @as(f32, @floatFromInt(acc)) * scale_a * scale_b;
        }
    }
}

// ============================================================================
// Tokenization Simulation
// ============================================================================

/// Simple BPE-like tokenization benchmark
fn tokenizeBpe(
    allocator: std.mem.Allocator,
    text: []const u8,
    vocab: []const []const u8,
) ![]u32 {
    var tokens = std.ArrayListUnmanaged(u32){};
    errdefer tokens.deinit(allocator);

    var i: usize = 0;
    while (i < text.len) {
        // Find longest matching token (greedy)
        var best_len: usize = 1;
        var best_id: u32 = 0; // Unknown token

        for (vocab, 0..) |token, id| {
            if (i + token.len <= text.len and
                std.mem.eql(u8, text[i..][0..token.len], token) and
                token.len > best_len)
            {
                best_len = token.len;
                best_id = @intCast(id);
            }
        }

        try tokens.append(allocator, best_id);
        i += best_len;
    }

    return tokens.toOwnedSlice(allocator);
}

// ============================================================================
// Feed Forward Network (MLP)
// ============================================================================

fn feedForward(
    allocator: std.mem.Allocator,
    input: []const f32, // [seq_len, hidden]
    W1: []const f32, // [hidden, intermediate]
    W2: []const f32, // [intermediate, hidden]
    output: []f32, // [seq_len, hidden]
    seq_len: usize,
    hidden: usize,
    intermediate: usize,
) !void {
    // First linear + activation
    const hidden_states = try allocator.alloc(f32, seq_len * intermediate);
    defer allocator.free(hidden_states);

    simdGemm(8, input, W1, hidden_states, seq_len, intermediate, hidden);
    geluVector(hidden_states, hidden_states);

    // Second linear
    simdGemm(8, hidden_states, W2, output, seq_len, hidden, intermediate);
}

// ============================================================================
// Main Benchmark Runner
// ============================================================================

pub fn runAIBenchmarks(allocator: std.mem.Allocator, _: AIBenchConfig) !void {
    var runner = framework.BenchmarkRunner.init(allocator);
    defer runner.deinit();

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("                    AI/ML INFERENCE BENCHMARKS\n", .{});
    std.debug.print("================================================================================\n\n", .{});

    // Activation functions
    std.debug.print("[Activation Functions]\n", .{});
    for ([_]usize{ 256, 1024, 4096 }) |size| {
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
                    .warmup_iterations = 1000,
                    .min_time_ns = 500_000_000,
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
                    .warmup_iterations = 1000,
                    .min_time_ns = 500_000_000,
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
                    .bytes_per_op = size * @sizeOf(f32) * 2,
                    .warmup_iterations = 1000,
                    .min_time_ns = 500_000_000,
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

        // Softmax
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "softmax_{d}", .{size}) catch "softmax";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "ai/activation",
                    .warmup_iterations = 1000,
                    .min_time_ns = 500_000_000,
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

    // Layer normalization
    std.debug.print("\n[Layer Normalization]\n", .{});
    for ([_]usize{ 256, 768, 1024 }) |hidden| {
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
                    .warmup_iterations = 1000,
                    .min_time_ns = 500_000_000,
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
                    .warmup_iterations = 1000,
                    .min_time_ns = 500_000_000,
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

    // Matrix multiplication (GEMM)
    std.debug.print("\n[Matrix Multiplication (GEMM)]\n", .{});
    for ([_]usize{ 128, 256, 512 }) |size| {
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
                    .min_time_ns = 500_000_000,
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
                    .min_time_ns = 500_000_000,
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
                    .min_time_ns = 500_000_000,
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

    // Attention
    std.debug.print("\n[Attention Computation]\n", .{});
    for ([_]usize{ 32, 64, 128 }) |seq_len| {
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
                .min_time_ns = 500_000_000,
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

    // Embedding lookup
    std.debug.print("\n[Embedding Lookup]\n", .{});
    for ([_]usize{ 32000, 50257 }) |vocab_size| {
        const hidden: usize = 768;
        const seq_len: usize = 128;

        const embeddings = try allocator.alloc(f32, vocab_size * hidden);
        defer allocator.free(embeddings);
        const indices = try allocator.alloc(u32, seq_len);
        defer allocator.free(indices);
        const output = try allocator.alloc(f32, seq_len * hidden);
        defer allocator.free(output);

        var prng = std.Random.DefaultPrng.init(42);
        for (embeddings) |*x| x.* = prng.random().float(f32);
        for (indices) |*idx| idx.* = prng.random().intRangeLessThan(u32, 0, @intCast(vocab_size));

        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "embed_v{d}_s{d}", .{ vocab_size, seq_len }) catch "embed";

        const result = try runner.run(
            .{
                .name = name,
                .category = "ai/embedding",
                .bytes_per_op = seq_len * hidden * @sizeOf(f32),
                .warmup_iterations = 1000,
                .min_time_ns = 500_000_000,
            },
            struct {
                fn bench(emb: []const f32, idx: []const u32, out: []f32, hid: usize) void {
                    embeddingLookup(emb, idx, out, hid);
                }
            }.bench,
            .{ embeddings, indices, output, hidden },
        );

        std.debug.print("  {s}: {d:.0} lookups/sec ({d:.2} GB/s)\n", .{
            name,
            result.stats.opsPerSecond() * @as(f64, @floatFromInt(seq_len)),
            result.stats.throughputMBps(seq_len * hidden * @sizeOf(f32)) / 1024.0,
        });
    }

    // Quantization
    std.debug.print("\n[Quantization]\n", .{});
    for ([_]usize{ 1024, 4096, 16384 }) |size| {
        const input = try allocator.alloc(f32, size);
        defer allocator.free(input);
        const quantized = try allocator.alloc(i8, size);
        defer allocator.free(quantized);
        const dequantized = try allocator.alloc(f32, size);
        defer allocator.free(dequantized);

        var prng = std.Random.DefaultPrng.init(42);
        for (input) |*x| x.* = prng.random().float(f32) * 2 - 1;

        var scale: f32 = undefined;

        // Quantize
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "quantize_int8_{d}", .{size}) catch "quant";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "ai/quantization",
                    .bytes_per_op = size * (@sizeOf(f32) + @sizeOf(i8)),
                    .warmup_iterations = 1000,
                    .min_time_ns = 500_000_000,
                },
                struct {
                    fn bench(inp: []const f32, out: []i8, s: *f32) void {
                        quantizeToInt8(inp, out, s);
                    }
                }.bench,
                .{ input, quantized, &scale },
            );

            std.debug.print("  {s}: {d:.0} ops/sec\n", .{ name, result.stats.opsPerSecond() });
        }

        // Dequantize
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "dequantize_int8_{d}", .{size}) catch "dequant";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "ai/quantization",
                    .bytes_per_op = size * (@sizeOf(i8) + @sizeOf(f32)),
                    .warmup_iterations = 1000,
                    .min_time_ns = 500_000_000,
                },
                struct {
                    fn bench(inp: []const i8, out: []f32, s: f32) void {
                        dequantizeFromInt8(inp, out, s);
                    }
                }.bench,
                .{ quantized, dequantized, scale },
            );

            std.debug.print("  {s}: {d:.0} ops/sec\n", .{ name, result.stats.opsPerSecond() });
        }
    }

    std.debug.print("\n", .{});
    runner.printSummaryDebug();
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    try runAIBenchmarks(allocator, .{});
}

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

    // Sum should be 1
    var sum: f32 = 0;
    for (output) |x| sum += x;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 0.001);

    // Larger input should have larger output
    try std.testing.expect(output[2] > output[1]);
    try std.testing.expect(output[1] > output[0]);
}

test "quantization roundtrip" {
    const input = [_]f32{ 0.5, -0.3, 0.8, -0.9 };
    var quantized: [4]i8 = undefined;
    var dequantized: [4]f32 = undefined;
    var scale: f32 = undefined;

    quantizeToInt8(&input, &quantized, &scale);
    dequantizeFromInt8(&quantized, &dequantized, scale);

    // Should be approximately equal (with quantization error)
    for (input, dequantized) |original, recovered| {
        try std.testing.expectApproxEqAbs(original, recovered, 0.02);
    }
}
