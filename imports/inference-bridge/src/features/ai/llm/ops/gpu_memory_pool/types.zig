const std = @import("std");

/// Number of size classes (powers of 2 from 256 bytes to 1GB)
pub const SIZE_CLASS_COUNT = 23; // 2^8 to 2^30

/// Minimum allocation size (256 bytes)
pub const MIN_ALLOC_SIZE = 256;

/// Maximum single allocation (1GB)
pub const MAX_ALLOC_SIZE = 1 << 30;

/// Configuration for the memory pool.
pub const PoolConfig = struct {
    /// Maximum GPU memory the pool can use (bytes)
    max_memory_bytes: usize = 4 * 1024 * 1024 * 1024, // 4GB default
    /// Initial buffers to pre-allocate per size class (0 = lazy allocation)
    initial_buffers_per_class: usize = 0,
    /// Maximum buffers to cache per size class
    max_buffers_per_class: usize = 16,
    /// Enable statistics collection
    enable_stats: bool = true,
    /// Minimum size to use GPU memory (smaller uses CPU)
    gpu_threshold_bytes: usize = 4096,
    /// Enable best-fit allocation (search entire free list for best match)
    enable_best_fit: bool = true,
    /// Enable buffer splitting (split larger buffers for smaller requests)
    enable_splitting: bool = true,
    /// Minimum size for a split remainder to be kept (avoid tiny fragments)
    min_split_remainder: usize = 256,
    /// Fragmentation threshold to trigger automatic defragmentation (0 = disabled)
    auto_defrag_threshold: f64 = 0.25, // 25% fragmentation
};

/// A pooled GPU memory buffer.
pub const PooledBuffer = struct {
    /// Pointer to device memory (or null if CPU fallback)
    device_ptr: ?*anyopaque,
    /// Actual allocated size (may be larger than requested)
    allocated_size: usize,
    /// Originally requested size (for fragmentation tracking)
    requested_size: usize,
    /// Size class index
    size_class: u8,
    /// Whether this is GPU memory (false = CPU)
    is_gpu: bool,
    /// Backing CPU allocation (if not GPU)
    cpu_data: ?[]u8,
    /// Last use timestamp for LRU
    last_use_ns: i128,
    /// Unique buffer ID (for coalescing/splitting tracking)
    buffer_id: u64,

    /// Get internal fragmentation for this buffer
    pub fn getFragmentation(self: PooledBuffer) usize {
        return self.allocated_size - self.requested_size;
    }
};

/// Statistics for pool usage.
pub const PoolStats = struct {
    /// Total allocations requested
    total_allocations: u64 = 0,
    /// Allocations served from cache
    cache_hits: u64 = 0,
    /// Allocations requiring new GPU memory
    cache_misses: u64 = 0,
    /// Total bytes currently allocated
    current_bytes: usize = 0,
    /// Peak bytes allocated
    peak_bytes: usize = 0,
    /// Total bytes released
    released_bytes: u64 = 0,
    /// Fallbacks to CPU memory
    cpu_fallbacks: u64 = 0,
    /// Evictions due to memory pressure
    evictions: u64 = 0,
    /// Total bytes wasted due to internal fragmentation
    internal_fragmentation_bytes: u64 = 0,
    /// Number of buffer splits performed
    buffer_splits: u64 = 0,
    /// Number of buffer coalesces performed
    buffer_coalesces: u64 = 0,
    /// Best-fit selections (picked non-first buffer)
    best_fit_selections: u64 = 0,
    /// Defragmentation runs
    defrag_runs: u64 = 0,
    /// Total bytes in free list (external fragmentation potential)
    free_list_bytes: u64 = 0,
    /// Count of free blocks that are too small to be useful
    unusable_free_blocks: u64 = 0,
    /// Bytes in unusable free blocks (external fragmentation)
    external_fragmentation_bytes: u64 = 0,
    /// Number of allocation requests that couldn't use free blocks due to size
    fragmentation_induced_misses: u64 = 0,

    /// Get cache hit rate as percentage.
    pub fn hitRate(self: PoolStats) f64 {
        if (self.total_allocations == 0) return 0;
        return @as(f64, @floatFromInt(self.cache_hits)) / @as(f64, @floatFromInt(self.total_allocations)) * 100.0;
    }

    /// Get GPU utilization (GPU vs CPU allocations).
    pub fn gpuUtilization(self: PoolStats) f64 {
        if (self.total_allocations == 0) return 0;
        const gpu_allocs = self.total_allocations - self.cpu_fallbacks;
        return @as(f64, @floatFromInt(gpu_allocs)) / @as(f64, @floatFromInt(self.total_allocations)) * 100.0;
    }

    /// Get internal fragmentation percentage.
    pub fn fragmentationRate(self: PoolStats) f64 {
        if (self.current_bytes == 0) return 0;
        return @as(f64, @floatFromInt(self.internal_fragmentation_bytes)) / @as(f64, @floatFromInt(self.current_bytes)) * 100.0;
    }

    /// Get external fragmentation ratio (0.0 to 1.0).
    /// External fragmentation = unusable free blocks / total free list bytes.
    /// High values indicate many small free blocks that can't satisfy allocations.
    pub fn externalFragmentationRatio(self: PoolStats) f64 {
        if (self.free_list_bytes == 0) return 0;
        return @as(f64, @floatFromInt(self.external_fragmentation_bytes)) /
            @as(f64, @floatFromInt(self.free_list_bytes));
    }

    /// Get combined fragmentation ratio (internal + external).
    /// This represents total memory inefficiency.
    pub fn totalFragmentationRatio(self: PoolStats) f64 {
        const total_waste = self.internal_fragmentation_bytes + self.external_fragmentation_bytes;
        const total_managed = self.current_bytes + self.free_list_bytes;
        if (total_managed == 0) return 0;
        return @as(f64, @floatFromInt(total_waste)) / @as(f64, @floatFromInt(total_managed));
    }

    /// Check if defragmentation is recommended (fragmentation > 30%).
    /// Returns a recommendation struct with details.
    pub fn checkDefragmentationNeeded(self: PoolStats) DefragmentationRecommendation {
        const ext_frag = self.externalFragmentationRatio();
        const total_frag = self.totalFragmentationRatio();
        const threshold: f64 = 0.30; // 30% threshold

        return DefragmentationRecommendation{
            .recommended = ext_frag > threshold or total_frag > threshold,
            .external_fragmentation_ratio = ext_frag,
            .total_fragmentation_ratio = total_frag,
            .unusable_blocks = self.unusable_free_blocks,
            .wasted_bytes = self.external_fragmentation_bytes + self.internal_fragmentation_bytes,
            .severity = if (total_frag > 0.5) .critical else if (total_frag > 0.3) .high else if (total_frag > 0.15) .moderate else .low,
        };
    }
};

/// Defragmentation recommendation from fragmentation analysis.
pub const DefragmentationRecommendation = struct {
    /// Whether defragmentation is recommended
    recommended: bool,
    /// External fragmentation ratio (0.0 to 1.0)
    external_fragmentation_ratio: f64,
    /// Combined internal + external fragmentation ratio
    total_fragmentation_ratio: f64,
    /// Number of unusable free blocks
    unusable_blocks: u64,
    /// Total wasted bytes (internal + external)
    wasted_bytes: u64,
    /// Severity level of fragmentation
    severity: FragmentationSeverity,

    /// Severity levels for fragmentation
    pub const FragmentationSeverity = enum {
        low, // < 15%
        moderate, // 15-30%
        high, // 30-50%
        critical, // > 50%

        pub fn toString(self: FragmentationSeverity) []const u8 {
            return switch (self) {
                .low => "low",
                .moderate => "moderate",
                .high => "high - defragmentation recommended",
                .critical => "critical - immediate defragmentation required",
            };
        }
    };

    /// Format the recommendation as a human-readable message.
    pub fn getMessage(self: DefragmentationRecommendation) []const u8 {
        if (!self.recommended) {
            return "Fragmentation levels are acceptable. No action needed.";
        }
        return switch (self.severity) {
            .low => "Fragmentation is low. Continue monitoring.",
            .moderate => "Fragmentation is moderate. Consider defragmentation during low-usage periods.",
            .high => "Fragmentation exceeds 30%. Defragmentation is recommended to improve memory efficiency.",
            .critical => "Critical fragmentation detected (>50%). Immediate defragmentation required to prevent allocation failures.",
        };
    }
};

/// Free list node for a size class.
pub const FreeNode = struct {
    buffer: PooledBuffer,
    next: ?*FreeNode,
};
