//! Core Benchmark Infrastructure
//!
//! This module provides the shared infrastructure for all ABI benchmarks:
//!
//! - **vectors**: Unified vector generation with multiple distributions
//! - **distance**: Distance metrics (Euclidean, cosine, dot product, Manhattan)
//! - **config**: Parameterized configurations with presets
//!
//! ## Usage
//!
//! ```zig
//! const core = @import("core/mod.zig");
//!
//! // Generate vectors
//! const vectors = try core.vectors.generate(allocator, .{
//!     .count = 10000,
//!     .dimension = 128,
//!     .distribution = .normalized,
//! });
//! defer core.vectors.free(allocator, vectors);
//!
//! // Compute distances
//! const dist = core.distance.compute(.euclidean, vectors[0], vectors[1]);
//!
//! // Use preset configurations
//! const db_config = core.config.DatabaseBenchConfig.standard;
//! ```

pub const vectors = @import("vectors.zig");
pub const distance = @import("distance.zig");
pub const config = @import("config.zig");

// Re-export common types for convenience
pub const VectorDistribution = vectors.VectorDistribution;
pub const VectorConfig = vectors.VectorConfig;
pub const Metric = distance.Metric;
pub const DatabaseBenchConfig = config.DatabaseBenchConfig;
pub const AIBenchConfig = config.AIBenchConfig;
pub const LLMBenchConfig = config.LLMBenchConfig;
pub const AnnDataset = config.AnnDataset;

// Re-export common functions for convenience
pub const generateVectors = vectors.generate;
pub const freeVectors = vectors.free;
pub const computeDistance = distance.compute;
pub const computeDistanceSimd = distance.computeSimd;

test {
    _ = vectors;
    _ = distance;
    _ = config;
}
