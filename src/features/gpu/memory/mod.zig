//! GPU Memory Management
//!
//! Unified module for GPU buffer allocation, memory pools, and lock-free
//! resource management.
//!
//! - `base`: Core buffer types, synchronous pool, async transfers
//! - `pool`: Advanced size-class allocation with fragmentation mitigation
//! - `lockfree`: Lock-free resource pool with generational handles

pub const base = @import("base.zig");
pub const pool = @import("pool.zig");
pub const lockfree = @import("lockfree.zig");

// Re-export core types from base
pub const MemoryError = base.MemoryError;
pub const BufferFlags = base.BufferFlags;
pub const GpuBuffer = base.GpuBuffer;
pub const GpuMemoryPool = base.GpuMemoryPool;
pub const MemoryStats = base.MemoryStats;
pub const AsyncTransfer = base.AsyncTransfer;

// Re-export advanced pool types
pub const AdvancedMemoryPool = pool.AdvancedMemoryPool;
pub const PoolConfig = pool.PoolConfig;
pub const PoolStats = pool.PoolStats;
pub const SizeClassStats = pool.SizeClassStats;
pub const DetailedPoolStats = pool.DetailedPoolStats;

// Re-export lock-free pool types
pub const LockFreeResourcePool = lockfree.LockFreeResourcePool;
pub const ResourceHandle = lockfree.ResourceHandle;
pub const LockFreePoolConfig = lockfree.PoolConfig;
pub const StatsSnapshot = lockfree.StatsSnapshot;
pub const ConcurrentCommandPool = lockfree.ConcurrentCommandPool;
pub const INVALID_HANDLE = lockfree.INVALID_HANDLE;
pub const CACHE_LINE_SIZE = lockfree.CACHE_LINE_SIZE;

test {
    _ = base;
    _ = pool;
    _ = lockfree;
}
