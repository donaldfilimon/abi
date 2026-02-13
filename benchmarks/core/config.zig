//! Parameterized Benchmark Configurations
//!
//! Provides standardized configuration presets for different benchmark scenarios:
//! - Quick: Fast CI runs
//! - Standard: Normal development testing
//! - Comprehensive: Full benchmark suite
//! - ANN-Benchmarks: Industry-standard compatible

const std = @import("std");

// ============================================================================
// Database Benchmark Configuration
// ============================================================================

/// Configuration for database/vector search benchmarks
pub const DatabaseBenchConfig = struct {
    /// Vector dimensions to test
    dimensions: []const usize = &.{ 64, 128, 256, 384, 512, 768, 1024, 1536 },
    /// Dataset sizes (number of vectors)
    dataset_sizes: []const usize = &.{ 1000, 10000, 50000, 100000 },
    /// Batch sizes for batch insertion
    batch_sizes: []const usize = &.{ 1, 10, 100, 1000 },
    /// k values for k-NN search
    k_values: []const usize = &.{ 1, 5, 10, 20, 50, 100 },
    /// Number of query iterations
    query_iterations: usize = 1000,
    /// HNSW M parameter (max connections per layer)
    hnsw_m: usize = 16,
    /// HNSW efConstruction parameter
    hnsw_ef_construction: usize = 200,
    /// HNSW efSearch values to test
    hnsw_ef_search: []const usize = &.{ 16, 32, 64, 128, 256 },
    /// Random seed for reproducibility
    seed: u64 = 42,
    /// Number of clusters for clustered vectors
    num_clusters: usize = 10,
    /// Whether to run concurrent tests
    test_concurrent: bool = true,
    /// Number of concurrent threads
    num_threads: usize = 4,
    /// Minimum benchmark time in nanoseconds
    min_time_ns: u64 = 1_000_000_000,

    /// Quick configuration for CI
    pub const quick = DatabaseBenchConfig{
        .dimensions = &.{ 128, 256 },
        .dataset_sizes = &.{ 500, 2000 },
        .batch_sizes = &.{ 1, 100 },
        .k_values = &.{ 1, 10 },
        .query_iterations = 20,
        .hnsw_ef_search = &.{ 32, 64 },
        .test_concurrent = false,
        .min_time_ns = 50_000_000,
    };

    /// Standard configuration for development
    pub const standard = DatabaseBenchConfig{
        .dimensions = &.{ 128, 256 },
        .dataset_sizes = &.{ 1000, 2000 },
        .batch_sizes = &.{ 1, 100 },
        .k_values = &.{ 1, 10 },
        .query_iterations = 20,
        .hnsw_ef_search = &.{ 32, 64 },
        .min_time_ns = 100_000_000,
    };

    /// Comprehensive configuration for full benchmarking
    pub const comprehensive = DatabaseBenchConfig{
        .dimensions = &.{ 64, 128, 256, 384, 512, 768, 1024, 1536 },
        .dataset_sizes = &.{ 1000, 10000, 50000, 100000 },
        .batch_sizes = &.{ 1, 10, 100, 1000 },
        .k_values = &.{ 1, 5, 10, 20, 50, 100 },
        .query_iterations = 1000,
        .hnsw_ef_search = &.{ 16, 32, 64, 128, 256, 512 },
        .min_time_ns = 2_000_000_000,
    };

    /// ANN-Benchmarks compatible configuration
    pub const ann_benchmarks = DatabaseBenchConfig{
        .dimensions = &.{ 128, 256, 960 }, // SIFT, custom, GIST dimensions
        .dataset_sizes = &.{ 10000, 100000, 1000000 },
        .batch_sizes = &.{1000},
        .k_values = &.{ 1, 10, 100 },
        .query_iterations = 10000,
        .hnsw_m = 16,
        .hnsw_ef_construction = 200,
        .hnsw_ef_search = &.{ 10, 50, 100, 200, 400 },
        .min_time_ns = 5_000_000_000,
    };
};

// ============================================================================
// AI/ML Benchmark Configuration
// ============================================================================

/// Configuration for AI/ML benchmarks
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
    /// Matrix sizes for GEMM benchmarks
    matrix_sizes: []const usize = &.{ 128, 256, 512, 1024 },
    /// Activation sizes for activation benchmarks
    activation_sizes: []const usize = &.{ 256, 1024, 4096, 16384 },
    /// Random seed
    seed: u64 = 42,
    /// Minimum benchmark time in nanoseconds
    min_time_ns: u64 = 500_000_000,
    /// Warmup iterations
    warmup_iterations: usize = 1000,

    /// Quick configuration for CI
    pub const quick = AIBenchConfig{
        .hidden_sizes = &.{ 256, 768 },
        .seq_lengths = &.{ 64, 128 },
        .batch_sizes = &.{ 1, 4 },
        .num_heads = &.{8},
        .vocab_sizes = &.{32000},
        .matrix_sizes = &.{ 128, 256 },
        .activation_sizes = &.{ 256, 1024 },
        .min_time_ns = 100_000_000,
        .warmup_iterations = 100,
    };

    /// Standard configuration for development
    pub const standard = AIBenchConfig{
        .hidden_sizes = &.{ 256, 512, 768 },
        .seq_lengths = &.{ 64, 128 },
        .batch_sizes = &.{ 1, 4 },
        .num_heads = &.{8},
        .vocab_sizes = &.{32000},
        .matrix_sizes = &.{ 128, 256 },
        .activation_sizes = &.{ 256, 1024 },
        .min_time_ns = 100_000_000,
        .warmup_iterations = 100,
    };

    /// Comprehensive configuration for full benchmarking
    pub const comprehensive = AIBenchConfig{
        .hidden_sizes = &.{ 256, 512, 768, 1024, 2048, 4096 },
        .seq_lengths = &.{ 32, 64, 128, 256, 512, 1024 },
        .batch_sizes = &.{ 1, 2, 4, 8, 16, 32 },
        .num_heads = &.{ 8, 12, 16, 32 },
        .vocab_sizes = &.{ 32000, 50257, 128256 },
        .matrix_sizes = &.{ 128, 256, 512, 1024, 2048 },
        .activation_sizes = &.{ 256, 1024, 4096, 16384, 65536 },
        .min_time_ns = 1_000_000_000,
        .warmup_iterations = 1000,
    };
};

// ============================================================================
// LLM Benchmark Configuration
// ============================================================================

/// Configuration for LLM-specific benchmarks
pub const LLMBenchConfig = struct {
    /// Number of evaluation samples
    num_samples: usize = 1000,
    /// Context length for evaluation
    context_length: usize = 512,
    /// Output length for generation
    output_length: usize = 128,
    /// Batch size
    batch_size: usize = 8,
    /// Random seed
    seed: u64 = 42,
    /// Whether to evaluate adversarial robustness
    eval_robustness: bool = true,
    /// Whether to evaluate fairness
    eval_fairness: bool = true,
    /// Model parameter count (for memory profiling)
    model_params: u64 = 7_000_000_000,
    /// Quantization levels to test
    quantization_levels: []const QuantizationLevel = &.{ .fp32, .fp16, .int8, .int4 },

    pub const QuantizationLevel = enum {
        fp32,
        fp16,
        bf16,
        int8,
        int4,
        int2,

        pub fn bitsPerWeight(self: QuantizationLevel) usize {
            return switch (self) {
                .fp32 => 32,
                .fp16 => 16,
                .bf16 => 16,
                .int8 => 8,
                .int4 => 4,
                .int2 => 2,
            };
        }

        pub fn name(self: QuantizationLevel) []const u8 {
            return switch (self) {
                .fp32 => "FP32",
                .fp16 => "FP16",
                .bf16 => "BF16",
                .int8 => "INT8",
                .int4 => "INT4",
                .int2 => "INT2",
            };
        }
    };

    /// Quick configuration for CI
    pub const quick = LLMBenchConfig{
        .num_samples = 100,
        .context_length = 256,
        .output_length = 64,
        .batch_size = 4,
        .eval_robustness = false,
        .eval_fairness = false,
        .model_params = 1_000_000_000,
        .quantization_levels = &.{ .fp16, .int8 },
    };

    /// Standard configuration
    pub const standard = LLMBenchConfig{
        .num_samples = 500,
        .context_length = 512,
        .output_length = 128,
        .batch_size = 8,
        .model_params = 7_000_000_000,
        .quantization_levels = &.{ .fp32, .fp16, .int8, .int4 },
    };

    /// Comprehensive configuration
    pub const comprehensive = LLMBenchConfig{
        .num_samples = 1000,
        .context_length = 2048,
        .output_length = 256,
        .batch_size = 16,
        .model_params = 70_000_000_000,
        .quantization_levels = &.{ .fp32, .fp16, .bf16, .int8, .int4, .int2 },
    };
};

// ============================================================================
// Streaming Benchmark Configuration
// ============================================================================

/// Configuration for streaming inference benchmarks
pub const StreamingBenchConfig = struct {
    /// Number of tokens to generate per run
    tokens_per_run: []const usize = &.{ 32, 64, 128, 256, 512 },
    /// Number of warmup iterations
    warmup_iterations: usize = 10,
    /// Number of benchmark iterations per configuration
    iterations: usize = 100,
    /// Simulated token delay range (nanoseconds) for mock generator
    min_token_delay_ns: u64 = 1_000_000, // 1ms
    max_token_delay_ns: u64 = 50_000_000, // 50ms
    /// Whether to benchmark SSE encoding overhead
    bench_sse_encoding: bool = true,
    /// Whether to benchmark WebSocket framing overhead
    bench_ws_framing: bool = true,
    /// Random seed for reproducibility
    seed: u64 = 42,

    /// Quick configuration for CI
    pub const quick = StreamingBenchConfig{
        .tokens_per_run = &.{ 32, 64 },
        .warmup_iterations = 5,
        .iterations = 50,
        .min_token_delay_ns = 1_000_000,
        .max_token_delay_ns = 10_000_000,
    };

    /// Standard configuration for development
    pub const standard = StreamingBenchConfig{
        .tokens_per_run = &.{ 32, 64, 128, 256 },
        .warmup_iterations = 10,
        .iterations = 100,
        .min_token_delay_ns = 1_000_000,
        .max_token_delay_ns = 30_000_000,
    };

    /// Comprehensive configuration for full benchmarking
    pub const comprehensive = StreamingBenchConfig{
        .tokens_per_run = &.{ 32, 64, 128, 256, 512, 1024 },
        .warmup_iterations = 20,
        .iterations = 200,
        .min_token_delay_ns = 500_000,
        .max_token_delay_ns = 100_000_000,
        .bench_sse_encoding = true,
        .bench_ws_framing = true,
    };
};

// ============================================================================
// Memory Profiling Configuration
// ============================================================================

/// Configuration for memory profiling benchmarks
pub const MemoryProfileConfig = struct {
    /// Model hidden size
    hidden_size: usize = 4096,
    /// Number of layers
    num_layers: usize = 32,
    /// Number of attention heads
    num_heads: usize = 32,
    /// Vocabulary size
    vocab_size: usize = 32000,
    /// Quantization level
    quantization: LLMBenchConfig.QuantizationLevel = .fp16,
    /// Target memory limit (GB)
    memory_limit_gb: f64 = 24.0,
    /// Context length
    context_length: usize = 4096,
    /// Batch size
    batch_size: usize = 8,

    /// Small model profile (7B)
    pub const small = MemoryProfileConfig{
        .hidden_size = 4096,
        .num_layers = 32,
        .num_heads = 32,
        .vocab_size = 32000,
        .memory_limit_gb = 16.0,
    };

    /// Medium model profile (13B)
    pub const medium = MemoryProfileConfig{
        .hidden_size = 5120,
        .num_layers = 40,
        .num_heads = 40,
        .vocab_size = 32000,
        .memory_limit_gb = 24.0,
    };

    /// Large model profile (70B)
    pub const large = MemoryProfileConfig{
        .hidden_size = 8192,
        .num_layers = 80,
        .num_heads = 64,
        .vocab_size = 32000,
        .memory_limit_gb = 80.0,
    };
};

// ============================================================================
// ANN-Benchmarks Dataset Configuration
// ============================================================================

/// Standard ANN-Benchmarks datasets
pub const AnnDataset = enum {
    /// SIFT1M - 1M 128d vectors
    sift_1m,
    /// GIST1M - 1M 960d vectors
    gist_1m,
    /// GloVe - 1.2M 100d word embeddings
    glove_100,
    /// Fashion-MNIST - 60K 784d vectors
    fashion_mnist,
    /// NYTimes - 290K 256d vectors
    nytimes,
    /// Custom dataset
    custom,

    pub fn dimension(self: AnnDataset) usize {
        return switch (self) {
            .sift_1m => 128,
            .gist_1m => 960,
            .glove_100 => 100,
            .fashion_mnist => 784,
            .nytimes => 256,
            .custom => 0,
        };
    }

    pub fn size(self: AnnDataset) usize {
        return switch (self) {
            .sift_1m => 1_000_000,
            .gist_1m => 1_000_000,
            .glove_100 => 1_200_000,
            .fashion_mnist => 60_000,
            .nytimes => 290_000,
            .custom => 0,
        };
    }

    pub fn name(self: AnnDataset) []const u8 {
        return switch (self) {
            .sift_1m => "sift-1m",
            .gist_1m => "gist-1m",
            .glove_100 => "glove-100",
            .fashion_mnist => "fashion-mnist",
            .nytimes => "nytimes",
            .custom => "custom",
        };
    }
};

// ============================================================================
// Tests
// ============================================================================

test "database config presets" {
    const quick = DatabaseBenchConfig.quick;
    const standard = DatabaseBenchConfig.standard;

    try std.testing.expect(quick.dimensions.len < standard.dimensions.len);
    try std.testing.expect(quick.min_time_ns < standard.min_time_ns);
}

test "ai config presets" {
    const quick = AIBenchConfig.quick;
    const comprehensive = AIBenchConfig.comprehensive;

    try std.testing.expect(quick.hidden_sizes.len < comprehensive.hidden_sizes.len);
    try std.testing.expect(quick.warmup_iterations < comprehensive.warmup_iterations);
}

test "ann dataset info" {
    try std.testing.expectEqual(@as(usize, 128), AnnDataset.sift_1m.dimension());
    try std.testing.expectEqual(@as(usize, 1_000_000), AnnDataset.sift_1m.size());
}

test "quantization levels" {
    try std.testing.expectEqual(@as(usize, 32), LLMBenchConfig.QuantizationLevel.fp32.bitsPerWeight());
    try std.testing.expectEqual(@as(usize, 4), LLMBenchConfig.QuantizationLevel.int4.bitsPerWeight());
}

test "streaming config presets" {
    const quick = StreamingBenchConfig.quick;
    const comprehensive = StreamingBenchConfig.comprehensive;

    try std.testing.expect(quick.iterations < comprehensive.iterations);
    try std.testing.expect(quick.warmup_iterations < comprehensive.warmup_iterations);
    try std.testing.expect(quick.tokens_per_run.len < comprehensive.tokens_per_run.len);
}
