//! Data Structures Module - High-performance concurrent data structures
//!
//! This module provides thread-safe and high-performance data structures
//! optimized for AI and machine learning workloads:
//! - Lock-free data structures for concurrent access
//! - Memory-efficient implementations
//! - SIMD-accelerated operations where applicable
//! - Specialized data structures for AI workloads
//! - Thread-safe collections with minimal contention
//! - Cache-friendly layouts for better performance

const std = @import("std");

// Re-export lock-free data structures
pub const LockFreeQueue = @import("lockfree.zig").LockFreeQueue;
pub const LockFreeStack = @import("lockfree.zig").LockFreeStack;
pub const AtomicList = @import("lockfree.zig").AtomicList;
pub const ConcurrentHashMap = @import("lockfree.zig").ConcurrentHashMap;

// Re-export specialized AI data structures
pub const CircularBuffer = @import("circular_buffer.zig").CircularBuffer;
pub const RingBuffer = @import("circular_buffer.zig").RingBuffer;
pub const BatchQueue = @import("batch_queue.zig").BatchQueue;
pub const MemoryPool = @import("memory_pool.zig").MemoryPool;
pub const ObjectPool = @import("object_pool.zig").ObjectPool;
pub const ThreadSafeCache = @import("cache.zig").ThreadSafeCache;
pub const LRUCache = @import("cache.zig").LRUCache;
pub const BloomFilter = @import("bloom_filter.zig").BloomFilter;
pub const CountMinSketch = @import("probabilistic.zig").CountMinSketch;
pub const HyperLogLog = @import("probabilistic.zig").HyperLogLog;

// Re-export vector and matrix data structures
pub const VectorStore = @import("vector_store.zig").VectorStore;
pub const SparseMatrix = @import("sparse_matrix.zig").SparseMatrix;
pub const DenseMatrix = @import("dense_matrix.zig").DenseMatrix;
pub const CompressedVector = @import("compressed_vector.zig").CompressedVector;

// Re-export tree and graph structures
pub const KDTree = @import("spatial.zig").KDTree;
pub const QuadTree = @import("spatial.zig").QuadTree;
pub const BallTree = @import("spatial.zig").BallTree;
pub const LSHForest = @import("spatial.zig").LSHForest;
pub const Graph = @import("graph.zig").Graph;
pub const DirectedGraph = @import("graph.zig").DirectedGraph;
pub const BipartiteGraph = @import("graph.zig").BipartiteGraph;

// Re-export time series structures
pub const TimeSeries = @import("time_series.zig").TimeSeries;
pub const TimeSeriesBuffer = @import("time_series.zig").TimeSeriesBuffer;
pub const SlidingWindow = @import("sliding_window.zig").SlidingWindow;
pub const ExponentialMovingAverage = @import("statistics.zig").ExponentialMovingAverage;

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

/// Initialize a lock-free queue with the specified capacity
pub fn createLockFreeQueue(comptime T: type, allocator: std.mem.Allocator, capacity: usize) !*LockFreeQueue(T) {
    return LockFreeQueue(T).init(allocator, capacity);
}

/// Initialize a lock-free stack with the specified capacity
pub fn createLockFreeStack(comptime T: type, allocator: std.mem.Allocator, capacity: usize) !*LockFreeStack(T) {
    return LockFreeStack(T).init(allocator, capacity);
}

/// Initialize a concurrent hash map with the specified capacity
pub fn createConcurrentHashMap(comptime K: type, comptime V: type, allocator: std.mem.Allocator, capacity: usize) !*ConcurrentHashMap(K, V) {
    return ConcurrentHashMap(K, V).init(allocator, capacity);
}

/// Initialize a circular buffer for time series data
pub fn createCircularBuffer(comptime T: type, allocator: std.mem.Allocator, capacity: usize) !*CircularBuffer(T) {
    return CircularBuffer(T).init(allocator, capacity);
}

/// Initialize a memory pool for object reuse
pub fn createMemoryPool(comptime T: type, allocator: std.mem.Allocator, pool_size: usize) !*MemoryPool(T) {
    return MemoryPool(T).init(allocator, pool_size);
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
    pub fn createOptimizedQueue(self: *DataStructureFactory, comptime T: type, use_case: enum { high_throughput, low_latency, memory_efficient }) !*LockFreeQueue(T) {
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
    // Test that all main types are accessible
    _ = LockFreeQueue;
    _ = LockFreeStack;
    _ = AtomicList;
    _ = ConcurrentHashMap;
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

    // Test operation recording
    stats.recordOperation(true, 1000);
    try std.testing.expect(stats.total_operations == 1);
    try std.testing.expect(stats.successful_operations == 1);
    try std.testing.expect(stats.avg_latency_ns > 0);

    // Test reset
    stats.reset();
    try std.testing.expect(stats.total_operations == 0);
    try std.testing.expect(stats.successful_operations == 0);
}
