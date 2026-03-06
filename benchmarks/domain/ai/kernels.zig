//! AI/ML Kernel Benchmarks
//!
//! Measures performance of the production-grade computational kernels used in ABI.
//! Benchmarks: GEMM (Quantized & Regular), Attention, RMSNorm, and Activations.

const std = @import("std");
const abi = @import("abi");
const framework = @import("../../system/framework.zig");
const core = @import("../../core/mod.zig");
const ops = abi.personas.routing.context_engine.triad.ai.llm.ops; // Deep path to canonical ops

pub fn runKernelBenchmarks(allocator: std.mem.Allocator, config: core.config.AIBenchConfig) !void {
    var runner = framework.BenchmarkRunner.init(allocator);
    defer runner.deinit();

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("                    AI/ML KERNEL BENCHMARKS\n", .{});
    std.debug.print("================================================================================\n\n", .{});

    try benchmarkActivations(allocator, &runner, config);
    try benchmarkNormalization(allocator, &runner, config);
    try benchmarkMatMul(allocator, &runner, config);
    try benchmarkAttention(allocator, &runner, config);

    runner.printSummaryDebug();
}

fn benchmarkActivations(allocator: std.mem.Allocator, runner: *framework.BenchmarkRunner, config: core.config.AIBenchConfig) !void {
    std.debug.print("[Activation Functions (SiLU, GELU, Softmax)]\n", .{});
    
    for (config.activation_sizes[0..@min(2, config.activation_sizes.len)]) |size| {
        const input = try allocator.alloc(f32, size);
        defer allocator.free(input);
        const output = try allocator.alloc(f32, size);
        defer allocator.free(output);
        @memset(input, 1.0);

        // SiLU
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "silu_{d}", .{size}) catch "silu";
            _ = try runner.run(.{ .name = name, .category = "ai/ops" }, struct {
                fn bench(inp: []const f32, out: []f32) void {
                    for (inp, out) |x, *o| { o.* = ops.silu(x); }
                }
            }.bench, .{ input, output });
        }

        // Softmax
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "softmax_{d}", .{size}) catch "softmax";
            _ = try runner.run(.{ .name = name, .category = "ai/ops" }, ops.softmax, .{ input, output });
        }
    }
}

fn benchmarkNormalization(allocator: std.mem.Allocator, runner: *framework.BenchmarkRunner, config: core.config.AIBenchConfig) !void {
    std.debug.print("\n[Normalization (RMSNorm)]\n", .{});
    
    for (config.hidden_sizes[0..@min(2, config.hidden_sizes.len)]) |hidden| {
        const input = try allocator.alloc(f32, hidden);
        defer allocator.free(input);
        const weights = try allocator.alloc(f32, hidden);
        defer allocator.free(weights);
        const output = try allocator.alloc(f32, hidden);
        defer allocator.free(output);
        @memset(input, 0.5);
        @memset(weights, 1.0);

        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "rmsnorm_{d}", .{hidden}) catch "rmsnorm";
        _ = try runner.run(.{ .name = name, .category = "ai/ops" }, ops.rmsNorm, .{ input, weights, output, 1e-5 });
    }
}

fn benchmarkMatMul(allocator: std.mem.Allocator, runner: *framework.BenchmarkRunner, config: core.config.AIBenchConfig) !void {
    std.debug.print("\n[Matrix Multiplication]\n", .{});
    
    for (config.matrix_sizes[0..@min(2, config.matrix_sizes.len)]) |n| {
        const a = try allocator.alloc(f32, n * n);
        defer allocator.free(a);
        const b = try allocator.alloc(f32, n * n);
        defer allocator.free(b);
        const c = try allocator.alloc(f32, n * n);
        defer allocator.free(c);
        @memset(a, 0.1);
        @memset(b, 0.2);

        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "matmul_f32_{d}x{d}", .{n, n}) catch "matmul";
        
        _ = try runner.run(
            .{ .name = name, .category = "ai/ops", .max_iterations = 50 }, 
            ops.matrixMultiply, 
            .{ a, b, c, n, n, n }
        );
    }
}

fn benchmarkAttention(allocator: std.mem.Allocator, runner: *framework.BenchmarkRunner, config: core.config.AIBenchConfig) !void {
    std.debug.print("\n[Self-Attention Head]\n", .{});
    
    for (config.seq_lengths[0..@min(2, config.seq_lengths.len)]) |seq_len| {
        const head_dim = 64;
        const q = try allocator.alloc(f32, seq_len * head_dim);
        defer allocator.free(q);
        const k = try allocator.alloc(f32, seq_len * head_dim);
        defer allocator.free(k);
        const v = try allocator.alloc(f32, seq_len * head_dim);
        defer allocator.free(v);
        const out = try allocator.alloc(f32, seq_len * head_dim);
        defer allocator.free(out);

        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "attention_s{d}_d{d}", .{seq_len, head_dim}) catch "attn";
        
        _ = try runner.run(
            .{ .name = name, .category = "ai/ops", .max_iterations = 20 },
            struct {
                fn bench(alloc: std.mem.Allocator, bq: []f32, bk: []f32, bv: []f32, bo: []f32, sl: usize, hd: usize) !void {
                    try ops.scaledDotProductAttention(alloc, bq, bk, bv, bo, sl, hd);
                }
            }.bench,
            .{ allocator, q, k, v, out, seq_len, head_dim }
        );
    }
}
