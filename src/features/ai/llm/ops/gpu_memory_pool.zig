//! LLM-Optimized GPU Memory Pool
//!
//! Provides efficient memory allocation for LLM inference by pooling and
//! reusing GPU device memory buffers. This eliminates the overhead of
//! repeated cudaMalloc/cudaFree calls during token generation.

const std = @import("std");
pub const pool = @import("gpu_memory_pool/pool.zig");
pub const types = @import("gpu_memory_pool/types.zig");
pub const analysis = @import("gpu_memory_pool/analysis.zig");

pub const LlmMemoryPool = pool.LlmMemoryPool;
pub const copyToBuffer = pool.copyToBuffer;
pub const copyFromBuffer = pool.copyFromBuffer;

pub const SIZE_CLASS_COUNT = types.SIZE_CLASS_COUNT;
pub const MIN_ALLOC_SIZE = types.MIN_ALLOC_SIZE;
pub const MAX_ALLOC_SIZE = types.MAX_ALLOC_SIZE;
pub const PoolConfig = types.PoolConfig;
pub const PooledBuffer = types.PooledBuffer;
pub const PoolStats = types.PoolStats;
pub const DefragmentationRecommendation = types.DefragmentationRecommendation;

pub const FragmentationAnalysis = analysis.FragmentationAnalysis;

test {
    _ = @import("gpu_memory_pool_test.zig");
    _ = @import("gpu_memory_pool/pool.zig");
    _ = @import("gpu_memory_pool/types.zig");
    _ = @import("gpu_memory_pool/analysis.zig");
    std.testing.refAllDecls(@This());
}
