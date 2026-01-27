//! AI Benchmark Module
//!
//! Consolidated benchmarks for AI/ML operations:
//!
//! - **kernels**: GEMM, activations, normalization, attention
//! - **llm_metrics**: HELM metrics, memory profiling, throughput analysis
//!
//! ## Usage
//!
//! ```zig
//! const ai_bench = @import("ai/mod.zig");
//!
//! // Run all AI benchmarks
//! try ai_bench.runAllBenchmarks(allocator, .standard);
//!
//! // Run specific benchmark suite
//! try ai_bench.kernels.runKernelBenchmarks(allocator, config);
//! ```

const std = @import("std");
const core = @import("../../core/mod.zig");
const framework = @import("../../system/framework.zig");

pub const kernels = @import("kernels.zig");
pub const llm_metrics = @import("llm_metrics.zig");
pub const streaming = @import("streaming.zig");

// Re-export common types
pub const HelmDimension = llm_metrics.HelmDimension;
pub const HelmMetric = llm_metrics.HelmMetric;
pub const HelmEvaluation = llm_metrics.HelmEvaluation;
pub const LlmMemoryProfile = llm_metrics.LlmMemoryProfile;
pub const ThroughputLatencyResult = llm_metrics.ThroughputLatencyResult;
pub const QuantizationAnalysis = llm_metrics.QuantizationAnalysis;

// Streaming benchmark types
pub const StreamingBenchConfig = streaming.StreamingBenchConfig;
pub const StreamingBenchResult = streaming.StreamingBenchResult;
pub const GenerationPattern = streaming.GenerationPattern;
pub const MockTokenGenerator = streaming.MockTokenGenerator;

/// Configuration preset
pub const ConfigPreset = enum {
    quick,
    standard,
    comprehensive,
};

/// Run all AI benchmarks with the given preset
pub fn runAllBenchmarks(allocator: std.mem.Allocator, preset: ConfigPreset) !void {
    const ai_config = switch (preset) {
        .quick => core.config.AIBenchConfig.quick,
        .standard => core.config.AIBenchConfig.standard,
        .comprehensive => core.config.AIBenchConfig.comprehensive,
    };

    const llm_config = switch (preset) {
        .quick => core.config.LLMBenchConfig.quick,
        .standard => core.config.LLMBenchConfig.standard,
        .comprehensive => core.config.LLMBenchConfig.comprehensive,
    };

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("                    AI/ML INFERENCE BENCHMARKS\n", .{});
    std.debug.print("================================================================================\n", .{});

    // Run kernel benchmarks
    try kernels.runKernelBenchmarks(allocator, ai_config);

    // Run LLM metrics benchmarks if not quick mode
    if (preset != .quick) {
        try llm_metrics.runLlmMetricsBenchmarks(allocator, llm_config);
    }

    // Run streaming benchmarks
    const streaming_preset: streaming.ConfigPreset = switch (preset) {
        .quick => .quick,
        .standard => .standard,
        .comprehensive => .comprehensive,
    };
    try streaming.runStreamingBenchmarks(allocator, streaming_preset);

    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("                    AI BENCHMARKS COMPLETE\n", .{});
    std.debug.print("================================================================================\n", .{});
}

/// Run AI benchmarks with custom configuration
pub fn runAIBenchmarks(allocator: std.mem.Allocator, config: core.config.AIBenchConfig) !void {
    try kernels.runKernelBenchmarks(allocator, config);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    try runAllBenchmarks(allocator, .standard);
}

test {
    _ = kernels;
    _ = llm_metrics;
    _ = streaming;
}
