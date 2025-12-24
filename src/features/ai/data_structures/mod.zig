//! Data Structures Module - High-performance concurrent data structures
//!
//! This module provides thread-safe and high-performance data structures
//! optimized for AI and machine learning workloads.
//!
//! ## Organization
//!
//! ### Legacy (legacy/) - 17 well-implemented, production-ready structures
//! These are maintained for backward compatibility and stability. They represent
//! mature, well-tested implementations and should not be consolidated further.
//!
//! **Queues & Buffers:**
//! - `BatchQueue` - Batching for high-throughput operations
//! - `CircularBuffer`, `RingBuffer` - Fixed-size circular buffers
//!
//! **Memory Management:**
//! - `MemoryPool` - Object pooling for memory reuse
//! - `ObjectPool` - Type-safe object pools
//! - `ThreadSafeCache`, `LRUCache` - Thread-safe caching
//!
//! **Lock-free:**
//! - `lockFreeQueue` - Lock-free queue (MPMC)
//! - `LockFreeStack` - Lock-free stack
//! - `lockFreeHashMap` - Lock-free hash map
//! - `workStealingDeque` - Work stealing deque
//! - `mpmcQueue` - Multi-producer multi-consumer queue
//!
//! **Matrices & Vectors:**
//! - `DenseMatrix` - Dense matrix storage
//! - `SparseMatrix` - Sparse matrix storage
//! - `CompressedVector` - Compressed vector storage
//! - `VectorStore` - Vector storage with similarity search
//!
//! **Specialized:**
//! - `BloomFilter` - Probabilistic set membership
//! - `CountMinSketch`, `HyperLogLog` - Probabilistic data structures
//! - `KDTree`, `QuadTree`, `BallTree`, `LSHForest` - Spatial indexing
//! - `Graph`, `DirectedGraph`, `BipartiteGraph` - Graph structures
//! - `TimeSeries`, `TimeSeriesBuffer` - Time series data
//! - `SlidingWindow` - Sliding window operations
//! - `ExponentialMovingAverage` - Moving average calculation
//!
//! ### Consolidated (concurrent.zig, memory.zig) - Modern unified types
//! Common configuration and types for new implementations:
//! - `concurrent.LockFreeError`, `concurrent.LockFreeStats`
//! - `memory.MemoryPoolConfig`
//!
//! **Note**: The legacy directory contains mature, well-tested implementations
//! that are actively used in production. They are intentionally kept separate
//! to maintain stability and avoid unnecessary refactoring. Consolidation efforts
//! should focus on creating new unified modules rather than modifying legacy code.

const std = @import("std");
const shared_utils = @import("../../../shared/utils/mod.zig");

// Re-export from legacy directory (maintained for compatibility)
pub const lockFreeQueue = @import("legacy/lockfree.zig").lockFreeQueue;
pub const LockFreeStack = @import("legacy/lockfree.zig").lockFreeStack;
pub const lockFreeHashMap = @import("legacy/lockfree.zig").lockFreeHashMap;
pub const workStealingDeque = @import("legacy/lockfree.zig").workStealingDeque;
pub const mpmcQueue = @import("legacy/lockfree.zig").mpmcQueue;

pub const CircularBuffer = @import("legacy/circular_buffer.zig").CircularBuffer;
pub const RingBuffer = @import("legacy/circular_buffer.zig").RingBuffer;
pub const BatchQueue = @import("legacy/batch_queue.zig").BatchQueue;
pub const MemoryPool = @import("legacy/memory_pool.zig").MemoryPool;
pub const ObjectPool = @import("legacy/object_pool.zig").ObjectPool;
pub const ThreadSafeCache = @import("legacy/cache.zig").ThreadSafeCache;
pub const LRUCache = @import("legacy/cache.zig").LRUCache;
pub const BloomFilter = @import("legacy/bloom_filter.zig").BloomFilter;
pub const CountMinSketch = @import("legacy/probabilistic.zig").CountMinSketch;
pub const HyperLogLog = @import("legacy/probabilistic.zig").HyperLogLog;

pub const VectorStore = @import("legacy/vector_store.zig").VectorStore;
pub const SparseMatrix = @import("legacy/sparse_matrix.zig").SparseMatrix;
pub const DenseMatrix = @import("legacy/dense_matrix.zig").DenseMatrix;
pub const CompressedVector = @import("legacy/compressed_vector.zig").CompressedVector;

pub const KDTree = @import("legacy/spatial.zig").KDTree;
pub const QuadTree = @import("legacy/spatial.zig").QuadTree;
pub const BallTree = @import("legacy/spatial.zig").BallTree;
pub const LSHForest = @import("legacy/spatial.zig").LSHForest;
pub const Graph = @import("legacy/graph.zig").Graph;
pub const DirectedGraph = @import("legacy/graph.zig").DirectedGraph;
pub const BipartiteGraph = @import("legacy/graph.zig").BipartiteGraph;

pub const TimeSeries = @import("legacy/time_series.zig").TimeSeries;
pub const TimeSeriesBuffer = @import("legacy/time_series.zig").TimeSeriesBuffer;
pub const SlidingWindow = @import("legacy/sliding_window.zig").SlidingWindow;
pub const ExponentialMovingAverage = @import("legacy/statistics.zig").ExponentialMovingAverage;

// Re-export consolidated modules (work in progress)
pub const concurrent = @import("concurrent.zig");
pub const memory = @import("memory.zig");

// Re-export commonly used types
pub const Allocator = std.mem.Allocator;

/// Configuration for data structure initialization
pub const DataStructureConfig = struct {
    /// Default initial capacity for dynamic structures
    default_capacity: usize = 1024,
    /// Whether to enable SIMD optimizations
    enable_simd: bool = true,
    /// Memory alignment for cache efficiency
    memory_alignment: u29 = 64,
    /// Whether to enable statistics collection
    enable_stats: bool = false,
    /// Maximum memory usage before cleanup (bytes)
    max_memory_usage: usize = 1024 * 1024 * 1024, // 1GB
    /// Cleanup threshold percentage (0.0-1.0)
    cleanup_threshold: f32 = 0.8,
};

/// Performance statistics for data structures
pub const DataStructureStats = struct {
    /// Total number of operations performed
    total_operations: u64 = 0,
    /// Number of successful operations
    successful_operations: u64 = 0,
    /// Total memory allocated (bytes)
    memory_allocated: usize = 0,
    /// Peak memory usage (bytes)
    peak_memory_usage: usize = 0,
    /// Average operation latency (nanoseconds)
    avg_latency_ns: u64 = 0,
    /// Cache hit rate (0.0-1.0)
    cache_hit_rate: f32 = 0.0,
    /// Lock contention events
    contention_events: u64 = 0,
    /// Last cleanup timestamp
    last_cleanup_time: i64 = 0,

    /// Reset all statistics
    pub fn reset(self: *DataStructureStats) void {
        self.* = DataStructureStats{};
    }

    /// Update operation statistics
    pub fn recordOperation(self: *DataStructureStats, success: bool, latency_ns: u64) void {
        self.total_operations += 1;
        if (success) self.successful_operations += 1;

        // Update average latency using exponential moving average
        const alpha: f64 = 0.1;
        const new_latency = @as(f64, @floatFromInt(latency_ns));
        const old_avg = @as(f64, @floatFromInt(self.avg_latency_ns));
        self.avg_latency_ns = @as(u64, @intFromFloat(alpha * new_latency + (1.0 - alpha) * old_avg));
    }
};

/// Initialize a lock-free queue with specified capacity
pub fn createLockFreeQueue(comptime T: type, allocator: std.mem.Allocator, capacity: usize) !*lockFreeQueue(T) {
    return lockFreeQueue(T).init(allocator, capacity);
}

/// Initialize a lock-free stack with specified capacity
pub fn createLockFreeStack(comptime T: type, allocator: std.mem.Allocator, capacity: usize) !*LockFreeStack(T) {
    return LockFreeStack(T).init(allocator, capacity);
}

/// Initialize a concurrent hash map with specified capacity
pub fn createLockFreeHashMap(comptime K: type, comptime V: type, allocator: std.mem.Allocator, capacity: usize) !*lockFreeHashMap(K, V) {
    return lockFreeHashMap(K, V).init(allocator, capacity);
}

/// Initialize a circular buffer for time series data
pub fn createCircularBuffer(comptime T: type, allocator: std.mem.Allocator, capacity: usize) !*CircularBuffer(T) {
    return CircularBuffer(T).init(allocator, capacity);
}

/// Initialize a memory pool for object reuse
pub fn createMemoryPool(comptime T: type, allocator: std.mem.Allocator, pool_size: usize) !*MemoryPool(T) {
    return try MemoryPool(T).init(allocator, pool_size);
}

/// Initialize a thread-safe LRU cache
pub fn createLRUCache(comptime K: type, comptime V: type, allocator: std.mem.Allocator, capacity: usize) !*LRUCache(K, V) {
    return LRUCache(K, V).init(allocator, capacity);
}

/// Initialize a vector store for embedding storage and similarity search
pub fn createVectorStore(comptime T: type, allocator: std.mem.Allocator, dimensions: usize, capacity: usize) !*VectorStore(T) {
    return VectorStore(T).init(allocator, dimensions, capacity);
}

/// Initialize a KD-tree for spatial indexing
pub fn createKDTree(comptime T: type, allocator: std.mem.Allocator, dimensions: usize) !*KDTree(T) {
    return KDTree(T).init(allocator, dimensions);
}

/// Initialize a sparse matrix for efficient storage of sparse data
pub fn createSparseMatrix(comptime T: type, allocator: std.mem.Allocator, rows: usize, cols: usize) !*SparseMatrix(T) {
    return SparseMatrix(T).init(allocator, rows, cols);
}

/// Initialize a time series buffer with automatic windowing
pub fn createTimeSeriesBuffer(comptime T: type, allocator: std.mem.Allocator, window_size: usize) !*TimeSeriesBuffer(T) {
    return TimeSeriesBuffer(T).init(allocator, window_size);
}

/// Data structure factory for creating optimized instances based on use case
pub const DataStructureFactory = struct {
    allocator: Allocator,
    config: DataStructureConfig,
    stats: DataStructureStats,

    pub fn init(allocator: Allocator, config: DataStructureConfig) DataStructureFactory {
        return DataStructureFactory{
            .allocator = allocator,
            .config = config,
            .stats = DataStructureStats{},
        };
    }

    /// Create an optimized queue for the specified use case
    pub fn createOptimizedQueue(self: *DataStructureFactory, comptime T: type, use_case: enum { high_throughput, low_latency, memory_efficient }) !*lockFreeQueue(T) {
        const capacity = switch (use_case) {
            .high_throughput => self.config.default_capacity * 4,
            .low_latency => self.config.default_capacity / 2,
            .memory_efficient => self.config.default_capacity / 4,
        };
        return createLockFreeQueue(T, self.allocator, capacity);
    }

    /// Create an optimized cache for the specified access pattern
    pub fn createOptimizedCache(self: *DataStructureFactory, comptime K: type, comptime V: type, access_pattern: enum { temporal, random, sequential }) !*LRUCache(K, V) {
        const capacity = switch (access_pattern) {
            .temporal => self.config.default_capacity,
            .random => self.config.default_capacity * 2,
            .sequential => self.config.default_capacity / 2,
        };
        return createLRUCache(K, V, self.allocator, capacity);
    }

    /// Get current statistics
    pub fn getStats(self: DataStructureFactory) DataStructureStats {
        return self.stats;
    }
};

test "Data structures module imports" {
    _ = lockFreeQueue;
    _ = LockFreeStack;
    _ = lockFreeHashMap;
    _ = workStealingDeque;
    _ = mpmcQueue;
    _ = CircularBuffer;
    _ = BatchQueue;
    _ = MemoryPool;
    _ = ThreadSafeCache;
    _ = VectorStore;
    _ = KDTree;
    _ = SparseMatrix;
    _ = TimeSeries;
    _ = DataStructureFactory;
}

test "Data structure configuration" {
    const config = DataStructureConfig{
        .default_capacity = 2048,
        .enable_simd = true,
        .memory_alignment = 32,
    };

    const factory = DataStructureFactory.init(std.testing.allocator, config);
    try std.testing.expect(factory.config.default_capacity == 2048);
    try std.testing.expect(factory.config.enable_simd == true);
}

test "Data structure statistics" {
    var stats = DataStructureStats{};

    stats.recordOperation(true, 1000);
    try std.testing.expect(stats.total_operations == 1);
    try std.testing.expect(stats.successful_operations == 1);
    try std.testing.expect(stats.avg_latency_ns > 0);

    stats.reset();
    try std.testing.expect(stats.total_operations == 0);
    try std.testing.expect(stats.successful_operations == 0);
}
