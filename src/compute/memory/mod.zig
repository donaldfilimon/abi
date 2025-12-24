//! Memory management primitives
//!
//! Provides aligned buffers, pool allocators, and arena utilities for
//! high-performance memory management.

pub const aligned_buffer = @import("aligned_buffer.zig");
pub const pool_allocator = @import("pool_allocator.zig");
pub const arena_utils = @import("arena_utils.zig");

pub const CacheAlignedBuffer = aligned_buffer.CacheAlignedBuffer;
pub const PoolAllocator = pool_allocator.PoolAllocator;
