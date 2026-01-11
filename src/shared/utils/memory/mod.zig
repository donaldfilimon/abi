//! Memory management utilities.
//!
//! This module provides comprehensive memory management primitives:
//!
//! ## Allocators
//! - `pool` - Size-segregated memory pool with O(1) allocation
//! - `stack` - Stack-based bump allocator for temporary allocations
//! - `tracking` - Debug allocator with leak detection
//! - `aligned` - SIMD/cache-aligned allocation utilities
//! - `thread_cache` - Per-thread allocation caching
//!
//! ## Buffers
//! - `zerocopy` - Reference-counted zero-copy buffers
//! - `ring` - Lock-free ring buffers for streaming
//!
//! ## Quick Start
//! ```zig
//! const memory = @import("shared/utils/memory/mod.zig");
//!
//! // Use tracking allocator for debugging
//! var tracker = memory.tracking.TrackingAllocator.init(allocator, .{});
//! defer {
//!     if (tracker.detectLeaks()) tracker.dumpLeaks(std.io.getStdErr().writer());
//!     tracker.deinit();
//! }
//!
//! // Use ring buffer for streaming
//! var ring = try memory.ring.ByteRing.init(allocator, 4096);
//! defer ring.deinit();
//!
//! // Use stack allocator for scratch space
//! var buffer: [1024]u8 = undefined;
//! var stack = memory.stack.StackAllocator.init(&buffer);
//! ```

pub const pool = @import("pool.zig");
pub const zerocopy = @import("zerocopy.zig");
pub const tracking = @import("tracking.zig");
pub const ring = @import("ring.zig");
pub const aligned = @import("aligned.zig");
pub const stack = @import("stack.zig");
pub const thread_cache = @import("thread_cache.zig");

// Re-export commonly used types
pub const MemoryPool = pool.MemoryPool;
pub const PoolConfig = pool.PoolConfig;
pub const SlabPool = pool.SlabPool;
pub const ZeroCopyBuffer = zerocopy.ZeroCopyBuffer;
pub const SharedBuffer = zerocopy.SharedBuffer;
pub const TrackingAllocator = tracking.TrackingAllocator;
pub const RingBuffer = ring.RingBuffer;
pub const ByteRing = ring.ByteRing;
pub const StackAllocator = stack.StackAllocator;
pub const AlignedAllocator = aligned.AlignedAllocator;
pub const ThreadCache = thread_cache.ThreadCache;
pub const ThreadArena = thread_cache.ThreadArena;

test {
    _ = pool;
    _ = zerocopy;
    _ = tracking;
    _ = ring;
    _ = aligned;
    _ = stack;
    _ = thread_cache;
}
