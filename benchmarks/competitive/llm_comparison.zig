//! LLM Inference Comparison Benchmarks
//!
//! Compares ABI's LLM inference against popular frameworks:
//! - llama.cpp (reference implementation)
//! - vLLM (high-throughput serving)
//! - TensorRT-LLM (NVIDIA optimized)
//! - GGML (quantized inference)
//!
//! ## Benchmark Categories
//!
//! 1. **Token Generation** - Tokens/second for autoregressive generation
//! 2. **Prompt Processing** - Time to first token (TTFT)
//! 3. **Batch Throughput** - Concurrent request handling
//! 4. **Memory Efficiency** - Model memory footprint

const std = @import("std");
const abi = @import("abi");
const mod = @import("mod.zig");
const framework = @import("../system/framework.zig");

/// LLM inference reference baselines
pub const LlmBaseline = struct {
    system: []const u8,
    model: []const u8,
    quantization: []const u8,
    hardware: []const u8,
    tokens_per_sec: f64,
    ttft_ms: f64,
    memory_gb: f64,
    batch_size: usize,
};

/// Published baselines from LLM benchmarks
pub const llm_baselines = [_]LlmBaseline{
    // Llama 2 7B benchmarks
    .{ .system = "llama.cpp", .model = "llama2-7b", .quantization = "Q4_K_M", .hardware = "RTX 4090", .tokens_per_sec = 120, .ttft_ms = 50, .memory_gb = 4.5, .batch_size = 1 },
    .{ .system = "llama.cpp", .model = "llama2-7b", .quantization = "Q8_0", .hardware = "RTX 4090", .tokens_per_sec = 90, .ttft_ms = 60, .memory_gb = 7.0, .batch_size = 1 },
    .{ .system = "llama.cpp", .model = "llama2-7b", .quantization = "F16", .hardware = "RTX 4090", .tokens_per_sec = 70, .ttft_ms = 80, .memory_gb = 14.0, .batch_size = 1 },

    .{ .system = "vLLM", .model = "llama2-7b", .quantization = "F16", .hardware = "RTX 4090", .tokens_per_sec = 150, .ttft_ms = 30, .memory_gb = 15.0, .batch_size = 8 },
    .{ .system = "vLLM", .model = "llama2-7b", .quantization = "AWQ", .hardware = "RTX 4090", .tokens_per_sec = 200, .ttft_ms = 25, .memory_gb = 5.0, .batch_size = 8 },

    .{ .system = "TensorRT-LLM", .model = "llama2-7b", .quantization = "INT8", .hardware = "RTX 4090", .tokens_per_sec = 250, .ttft_ms = 20, .memory_gb = 6.0, .batch_size = 8 },

    // Llama 2 13B benchmarks
    .{ .system = "llama.cpp", .model = "llama2-13b", .quantization = "Q4_K_M", .hardware = "RTX 4090", .tokens_per_sec = 70, .ttft_ms = 80, .memory_gb = 8.0, .batch_size = 1 },
    .{ .system = "vLLM", .model = "llama2-13b", .quantization = "AWQ", .hardware = "A100 80GB", .tokens_per_sec = 180, .ttft_ms = 40, .memory_gb = 10.0, .batch_size = 16 },

    // Mistral 7B benchmarks
    .{ .system = "llama.cpp", .model = "mistral-7b", .quantization = "Q4_K_M", .hardware = "RTX 4090", .tokens_per_sec = 130, .ttft_ms = 45, .memory_gb = 4.5, .batch_size = 1 },
    .{ .system = "vLLM", .model = "mistral-7b", .quantization = "AWQ", .hardware = "RTX 4090", .tokens_per_sec = 220, .ttft_ms = 22, .memory_gb = 5.0, .batch_size = 8 },

    // CPU baselines
    .{ .system = "llama.cpp", .model = "llama2-7b", .quantization = "Q4_K_M", .hardware = "M2 Max", .tokens_per_sec = 45, .ttft_ms = 100, .memory_gb = 4.5, .batch_size = 1 },
    .{ .system = "llama.cpp", .model = "llama2-7b", .quantization = "Q4_K_M", .hardware = "i9-13900K", .tokens_per_sec = 35, .ttft_ms = 120, .memory_gb = 4.5, .batch_size = 1 },
};

/// Simulated LLM token generation benchmark
fn benchmarkTokenGeneration(
    allocator: std.mem.Allocator,
    num_tokens: usize,
    context_size: usize,
) !struct { tokens_per_sec: f64, ttft_ms: f64 } {
    // Simulate token generation (replace with actual LLM inference)
    var timer = abi.services.shared.time.Timer.start() catch return error.TimerFailed;

    // Simulate context processing (TTFT)
    var context = try allocator.alloc(f32, context_size * 4096); // Simulated hidden state
    defer allocator.free(context);
    for (context, 0..) |*c, i| {
        c.* = @as(f32, @floatFromInt(i % 256)) / 256.0;
    }
    const ttft_ns = timer.read();

    // Simulate autoregressive generation
    const generated = try allocator.alloc(u32, num_tokens);
    defer allocator.free(generated);

    const gen_start = timer.read();
    for (generated, 0..) |*token, i| {
        // Simulate forward pass
        var sum: f32 = 0;
        for (context[0..1024]) |c| {
            sum += c;
        }
        token.* = @intFromFloat(@mod(sum, 32000.0));
        _ = i;
    }
    const gen_end = timer.read();

    const gen_time_sec = @as(f64, @floatFromInt(gen_end - gen_start)) / 1_000_000_000.0;
    const tokens_per_sec = if (gen_time_sec > 0.0)
        @as(f64, @floatFromInt(num_tokens)) / gen_time_sec
    else
        0.0;
    const ttft_ms = @as(f64, @floatFromInt(ttft_ns)) / 1_000_000.0;

    return .{
        .tokens_per_sec = tokens_per_sec,
        .ttft_ms = ttft_ms,
    };
}

/// Benchmark batch throughput
fn benchmarkBatchThroughput(
    allocator: std.mem.Allocator,
    batch_size: usize,
    tokens_per_request: usize,
) !struct { total_tokens_per_sec: f64, latency_per_request_ms: f64 } {
    if (batch_size == 0) return error.InvalidBatchSize;

    var timer = abi.services.shared.time.Timer.start() catch return error.TimerFailed;

    // Simulate batch processing
    const batch_results = try allocator.alloc([]u32, batch_size);
    var batch_allocated: usize = 0;
    defer {
        for (batch_results[0..batch_allocated]) |r| {
            allocator.free(r);
        }
        allocator.free(batch_results);
    }

    for (batch_results) |*result| {
        result.* = try allocator.alloc(u32, tokens_per_request);
        batch_allocated += 1;
        for (result.*, 0..) |*token, i| {
            token.* = @intCast(i % 32000);
        }
    }

    const elapsed_ns = timer.read();
    const elapsed_sec = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;
    const total_tokens = batch_size * tokens_per_request;
    const total_tokens_per_sec = if (elapsed_sec > 0.0)
        @as(f64, @floatFromInt(total_tokens)) / elapsed_sec
    else
        0.0;
    const latency_per_request_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0 / @as(f64, @floatFromInt(batch_size));

    return .{
        .total_tokens_per_sec = total_tokens_per_sec,
        .latency_per_request_ms = latency_per_request_ms,
    };
}

/// Run all LLM comparison benchmarks
pub fn runBenchmarks(allocator: std.mem.Allocator, config: mod.CompetitiveConfig, runner: *framework.BenchmarkRunner) !void {
    _ = config;
    std.debug.print("Comparing ABI LLM inference against frameworks...\n\n", .{});

    // Single-request generation benchmark
    std.debug.print("Single Request Generation:\n", .{});
    const single_result = try benchmarkTokenGeneration(allocator, 128, 512);
    std.debug.print("  ABI: {d:.0} tokens/sec, TTFT={d:.2}ms\n", .{
        single_result.tokens_per_sec,
        single_result.ttft_ms,
    });

    // Record Single Request Result
    const single_mean_ns = 1_000_000_000.0 / single_result.tokens_per_sec;
    try runner.appendResult(.{
        .config = .{
            .name = "ABI LLM Single Request",
            .category = "llm",
        },
        .stats = .{
            .min_ns = 0,
            .max_ns = 0,
            .mean_ns = single_mean_ns,
            .median_ns = single_mean_ns,
            .std_dev_ns = 0,
            .p50_ns = @intFromFloat(single_result.ttft_ms * 1_000_000.0), // Use TTFT for latency metric
            .p90_ns = @intFromFloat(single_result.ttft_ms * 1_000_000.0),
            .p95_ns = 0,
            .p99_ns = 0,
            .iterations = 128,
            .outliers_removed = 0,
            .total_time_ns = @intFromFloat(single_mean_ns * 128.0),
        },
        .memory_allocated = 0,
        .memory_freed = 0,
        .timestamp = 0,
    });

    // Compare with baselines
    std.debug.print("  Baselines (Q4_K_M, RTX 4090):\n", .{});
    for (llm_baselines) |baseline| {
        if (std.mem.eql(u8, baseline.quantization, "Q4_K_M") and
            std.mem.eql(u8, baseline.hardware, "RTX 4090") and
            baseline.batch_size == 1)
        {
            std.debug.print("    {s}: {d:.0} tokens/sec, TTFT={d:.0}ms\n", .{
                baseline.system,
                baseline.tokens_per_sec,
                baseline.ttft_ms,
            });
        }
    }

    // Batch throughput benchmark
    std.debug.print("\nBatch Throughput (batch_size=8):\n", .{});
    const batch_result = try benchmarkBatchThroughput(allocator, 8, 64);
    std.debug.print("  ABI: {d:.0} total tokens/sec, {d:.2}ms per request\n", .{
        batch_result.total_tokens_per_sec,
        batch_result.latency_per_request_ms,
    });

    // Record Batch Result
    const batch_mean_ns = 1_000_000_000.0 / batch_result.total_tokens_per_sec;
    try runner.appendResult(.{
        .config = .{
            .name = "ABI LLM Batch Throughput",
            .category = "llm",
        },
        .stats = .{
            .min_ns = 0,
            .max_ns = 0,
            .mean_ns = batch_mean_ns,
            .median_ns = batch_mean_ns,
            .std_dev_ns = 0,
            .p50_ns = @intFromFloat(batch_result.latency_per_request_ms * 1_000_000.0),
            .p90_ns = 0,
            .p95_ns = 0,
            .p99_ns = 0,
            .iterations = 8 * 64,
            .outliers_removed = 0,
            .total_time_ns = @intFromFloat(batch_mean_ns * (8.0 * 64.0)),
        },
        .memory_allocated = 0,
        .memory_freed = 0,
        .timestamp = 0,
    });

    // Compare with batch baselines
    std.debug.print("  Baselines (batch_size>=8):\n", .{});
    for (llm_baselines) |baseline| {
        if (baseline.batch_size >= 8) {
            std.debug.print("    {s} ({s}): {d:.0} tokens/sec\n", .{
                baseline.system,
                baseline.model,
                baseline.tokens_per_sec,
            });
        }
    }

    std.debug.print("\n", .{});
}

/// Generate LLM benchmark comparison report
pub fn generateReport(allocator: std.mem.Allocator) !void {
    _ = allocator;

    std.debug.print("\n## LLM Inference Comparison\n\n", .{});

    std.debug.print("### Single-Request Performance (tokens/sec)\n\n", .{});
    std.debug.print("| System | Model | Quantization | GPU | Tokens/sec | TTFT (ms) |\n", .{});
    std.debug.print("|--------|-------|--------------|-----|------------|----------|\n", .{});

    for (llm_baselines) |baseline| {
        if (baseline.batch_size == 1 and std.mem.eql(u8, baseline.model, "llama2-7b")) {
            std.debug.print("| {s} | {s} | {s} | {s} | {d:.0} | {d:.0} |\n", .{
                baseline.system,
                baseline.model,
                baseline.quantization,
                baseline.hardware,
                baseline.tokens_per_sec,
                baseline.ttft_ms,
            });
        }
    }
    std.debug.print("| **ABI** | llama2-7b | Q4_K_M | - | TBD | TBD |\n\n", .{});

    std.debug.print("### Batch Throughput\n\n", .{});
    std.debug.print("| System | Model | Batch Size | Tokens/sec |\n", .{});
    std.debug.print("|--------|-------|------------|------------|\n", .{});

    for (llm_baselines) |baseline| {
        if (baseline.batch_size >= 8) {
            std.debug.print("| {s} | {s} | {d} | {d:.0} |\n", .{
                baseline.system,
                baseline.model,
                baseline.batch_size,
                baseline.tokens_per_sec,
            });
        }
    }
    std.debug.print("| **ABI** | - | 8 | TBD |\n", .{});
}

test "llm token generation benchmark" {
    const allocator = std.testing.allocator;

    const result = try benchmarkTokenGeneration(allocator, 32, 64);

    try std.testing.expect(result.tokens_per_sec > 0);
    try std.testing.expect(result.ttft_ms >= 0);
}

test "llm batch throughput benchmark" {
    const allocator = std.testing.allocator;

    const result = try benchmarkBatchThroughput(allocator, 4, 16);

    try std.testing.expect(result.total_tokens_per_sec > 0);
    try std.testing.expect(result.latency_per_request_ms >= 0);
}
