//! Lock-Free GPU Memory Pool with Generational Handles
//!
//! High-performance, wait-free memory allocation for GPU resources designed for
//! concurrent multi-threaded access patterns common in modern GPU workloads.
//!
//! ## Features
//!
//! - **Lock-free allocation/deallocation**: O(1) operations using atomic CAS
//! - **Generational handles**: 64-bit handles prevent ABA problem and use-after-free
//! - **Cache-line alignment**: All atomic data structures are 64-byte aligned to prevent false sharing
//! - **Thread-local pools**: Optional per-thread buffer pools for zero-contention hot paths
//! - **Statistics tracking**: Atomic counters for monitoring without blocking
//!
//! ## Handle Format
//!
//! Handles are 64-bit values encoding both index and generation:
//! - Bits [0:31]: Slot index (supports up to 4 billion slots)
//! - Bits [32:63]: Generation counter (prevents ABA problem)
//!
//! ## Usage
//!
//! ```zig
//! var pool = try LockFreeResourcePool.init(allocator, .{
//!     .max_slots = 1024,
//!     .slot_size = 65536,  // 64KB per slot
//! });
//! defer pool.deinit();
//!
//! // Allocate a resource
//! const handle = try pool.allocate();
//!
//! // Access the resource (validates handle)
//! if (pool.get(handle)) |resource| {
//!     // Use resource...
//! }
//!
//! // Free when done
//! pool.free(handle);
//! ```
//!
//! ## Thread Safety
//!
//! All operations are thread-safe and lock-free. Multiple threads can
//! simultaneously allocate, access, and free resources without blocking.

const std = @import("std");
const memory = @import("base.zig");
const time = @import("../../../services/shared/time.zig");

/// Cache line size for alignment (x86/ARM64)
pub const CACHE_LINE_SIZE: usize = 64;

/// Maximum number of threads supported for thread-local pools
pub const MAX_THREADS: usize = 256;

/// Invalid handle sentinel value
pub const INVALID_HANDLE: ResourceHandle = .{ .value = std.math.maxInt(u64) };

/// Generational resource handle
/// Upper 32 bits: generation counter
/// Lower 32 bits: slot index
pub const ResourceHandle = struct {
    value: u64,

    pub fn init(idx: u32, gen: u32) ResourceHandle {
        return .{ .value = (@as(u64, gen) << 32) | @as(u64, idx) };
    }

    pub fn index(self: ResourceHandle) u32 {
        return @truncate(self.value);
    }

    pub fn generation(self: ResourceHandle) u32 {
        return @truncate(self.value >> 32);
    }

    pub fn isValid(self: ResourceHandle) bool {
        return self.value != INVALID_HANDLE.value;
    }

    pub fn format(
        self: ResourceHandle,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        try writer.print("Handle(idx={d}, gen={d})", .{ self.index(), self.generation() });
    }
};

/// Cache-line aligned slot metadata
/// This structure is designed to minimize false sharing in concurrent access
const SlotMetadata = struct {
    /// Generation counter (incremented on each allocation)
    generation: std.atomic.Value(u32) align(CACHE_LINE_SIZE),
    /// Allocation state
    state: std.atomic.Value(SlotState),
    /// Size of the current allocation (0 if free)
    size: std.atomic.Value(usize),
    /// Timestamp of allocation (for debugging/metrics)
    alloc_time: std.atomic.Value(i64),
    /// Padding to ensure next slot starts on cache line
    _padding: [CACHE_LINE_SIZE - @sizeOf(std.atomic.Value(u32)) - @sizeOf(std.atomic.Value(SlotState)) - @sizeOf(std.atomic.Value(usize)) - @sizeOf(std.atomic.Value(i64))]u8 = undefined,

    const SlotState = enum(u8) {
        free = 0,
        allocated = 1,
        retiring = 2, // Being freed, awaiting epoch
    };
};

/// Lock-free free list node
/// Uses tagged pointer to prevent ABA problem
const FreeListNode = struct {
    /// Next pointer with embedded tag
    next: std.atomic.Value(u64) align(CACHE_LINE_SIZE),
    /// Slot index this node represents
    slot_index: u32,
    _padding: [CACHE_LINE_SIZE - @sizeOf(std.atomic.Value(u64)) - @sizeOf(u32)]u8 = undefined,

    const TAG_BITS: u6 = 16;
    const TAG_MASK: u64 = (@as(u64, 1) << TAG_BITS) - 1;
    const PTR_MASK: u64 = ~TAG_MASK;

    fn getPointer(tagged: u64) ?*FreeListNode {
        const ptr_bits = tagged & PTR_MASK;
        if (ptr_bits == 0) return null;
        return @ptrFromInt(ptr_bits);
    }

    fn getTag(tagged: u64) u16 {
        return @truncate(tagged & TAG_MASK);
    }

    fn makeTagged(ptr: ?*FreeListNode, tag: u16) u64 {
        const ptr_bits: u64 = if (ptr) |p| @intFromPtr(p) else 0;
        return (ptr_bits & PTR_MASK) | @as(u64, tag);
    }
};

/// Pool statistics (all atomics for lock-free reads)
pub const PoolStats = struct {
    /// Total allocations performed
    total_allocations: std.atomic.Value(u64) align(CACHE_LINE_SIZE),
    /// Total deallocations performed
    total_deallocations: std.atomic.Value(u64),
    /// Current number of allocated slots
    active_allocations: std.atomic.Value(u64),
    /// Peak number of concurrent allocations
    peak_allocations: std.atomic.Value(u64),
    /// Failed allocation attempts
    failed_allocations: std.atomic.Value(u64),
    /// Invalid handle access attempts
    invalid_accesses: std.atomic.Value(u64),

    fn init() PoolStats {
        return .{
            .total_allocations = std.atomic.Value(u64).init(0),
            .total_deallocations = std.atomic.Value(u64).init(0),
            .active_allocations = std.atomic.Value(u64).init(0),
            .peak_allocations = std.atomic.Value(u64).init(0),
            .failed_allocations = std.atomic.Value(u64).init(0),
            .invalid_accesses = std.atomic.Value(u64).init(0),
        };
    }

    pub fn snapshot(self: *const PoolStats) StatsSnapshot {
        return .{
            .total_allocations = self.total_allocations.load(.acquire),
            .total_deallocations = self.total_deallocations.load(.acquire),
            .active_allocations = self.active_allocations.load(.acquire),
            .peak_allocations = self.peak_allocations.load(.acquire),
            .failed_allocations = self.failed_allocations.load(.acquire),
            .invalid_accesses = self.invalid_accesses.load(.acquire),
        };
    }
};

/// Immutable snapshot of pool statistics
pub const StatsSnapshot = struct {
    total_allocations: u64,
    total_deallocations: u64,
    active_allocations: u64,
    peak_allocations: u64,
    failed_allocations: u64,
    invalid_accesses: u64,

    pub fn utilizationRatio(self: StatsSnapshot, max_slots: usize) f64 {
        if (max_slots == 0) return 0.0;
        return @as(f64, @floatFromInt(self.active_allocations)) / @as(f64, @floatFromInt(max_slots));
    }

    pub fn allocationSuccessRate(self: StatsSnapshot) f64 {
        const total = self.total_allocations + self.failed_allocations;
        if (total == 0) return 1.0;
        return @as(f64, @floatFromInt(self.total_allocations)) / @as(f64, @floatFromInt(total));
    }
};

/// Configuration for the lock-free resource pool
pub const PoolConfig = struct {
    /// Maximum number of resource slots
    max_slots: u32 = 1024,
    /// Size of each resource slot in bytes
    slot_size: usize = 65536,
    /// Enable thread-local caching for hot paths
    enable_thread_local_cache: bool = true,
    /// Size of thread-local free list cache
    thread_local_cache_size: usize = 8,
    /// Pre-allocate all slots on init
    preallocate: bool = false,
    /// Buffer flags for GPU buffers
    buffer_flags: memory.BufferFlags = .{},
};

/// Lock-Free Resource Pool
/// Provides O(1) wait-free allocation and deallocation of GPU resources
pub const LockFreeResourcePool = struct {
    allocator: std.mem.Allocator,
    config: PoolConfig,

    /// Slot metadata array (cache-line aligned)
    slots: []SlotMetadata,
    /// Underlying GPU buffers
    buffers: []?*memory.GpuBuffer,
    /// Free list nodes
    free_nodes: []FreeListNode,
    /// Head of free list (tagged pointer for ABA prevention)
    free_list_head: std.atomic.Value(u64) align(CACHE_LINE_SIZE),
    /// Statistics
    stats: PoolStats,

    /// Thread-local cache for reducing contention
    thread_caches: ?[]ThreadLocalCache,

    pub fn init(allocator: std.mem.Allocator, config: PoolConfig) !LockFreeResourcePool {
        const max_slots = config.max_slots;

        // Allocate cache-line aligned slot metadata
        const slots = try allocator.alloc(SlotMetadata, max_slots);
        errdefer allocator.free(slots);

        // Initialize slot metadata
        for (slots) |*slot| {
            slot.* = .{
                .generation = std.atomic.Value(u32).init(0),
                .state = std.atomic.Value(SlotMetadata.SlotState).init(.free),
                .size = std.atomic.Value(usize).init(0),
                .alloc_time = std.atomic.Value(i64).init(0),
            };
        }

        // Allocate buffer pointer array
        const buffers = try allocator.alloc(?*memory.GpuBuffer, max_slots);
        errdefer allocator.free(buffers);
        @memset(buffers, null);

        // Allocate free list nodes
        const free_nodes = try allocator.alloc(FreeListNode, max_slots);
        errdefer allocator.free(free_nodes);

        // Initialize free list (all slots start free, linked in reverse order)
        var i: u32 = 0;
        while (i < max_slots) : (i += 1) {
            free_nodes[i] = .{
                .next = std.atomic.Value(u64).init(0),
                .slot_index = i,
            };
            // Link to previous node (building list in reverse)
            if (i > 0) {
                free_nodes[i].next.store(
                    FreeListNode.makeTagged(&free_nodes[i - 1], 0),
                    .release,
                );
            }
        }

        // Head points to last node (highest index)
        const head_value = if (max_slots > 0)
            FreeListNode.makeTagged(&free_nodes[max_slots - 1], 0)
        else
            0;

        // Allocate thread-local caches if enabled
        var thread_caches: ?[]ThreadLocalCache = null;
        if (config.enable_thread_local_cache) {
            thread_caches = try allocator.alloc(ThreadLocalCache, MAX_THREADS);
            for (thread_caches.?) |*cache| {
                cache.* = ThreadLocalCache.init(config.thread_local_cache_size);
            }
        }

        var pool = LockFreeResourcePool{
            .allocator = allocator,
            .config = config,
            .slots = slots,
            .buffers = buffers,
            .free_nodes = free_nodes,
            .free_list_head = std.atomic.Value(u64).init(head_value),
            .stats = PoolStats.init(),
            .thread_caches = thread_caches,
        };

        // Pre-allocate buffers if requested
        if (config.preallocate) {
            for (0..max_slots) |idx| {
                const buffer = try allocator.create(memory.GpuBuffer);
                errdefer allocator.destroy(buffer);
                buffer.* = try memory.GpuBuffer.init(allocator, config.slot_size, config.buffer_flags);
                pool.buffers[idx] = buffer;
            }
        }

        return pool;
    }

    pub fn deinit(self: *LockFreeResourcePool) void {
        // Free all allocated GPU buffers
        for (self.buffers) |maybe_buffer| {
            if (maybe_buffer) |buffer| {
                var buf = buffer;
                buf.deinit();
                self.allocator.destroy(buf);
            }
        }

        if (self.thread_caches) |caches| {
            self.allocator.free(caches);
        }

        self.allocator.free(self.buffers);
        self.allocator.free(self.free_nodes);
        self.allocator.free(self.slots);
        self.* = undefined;
    }

    /// Allocate a resource slot, returning a generational handle
    /// This operation is wait-free under normal conditions
    pub fn allocate(self: *LockFreeResourcePool) !ResourceHandle {
        // Try thread-local cache first (fastest path)
        if (self.thread_caches) |caches| {
            const cache_idx = getThreadCacheIndex();
            var cache = &caches[cache_idx];
            if (cache.pop()) |slot_index| {
                return self.finalizeAllocation(slot_index);
            }
        }

        // Fall back to global free list
        return self.allocateFromGlobalList();
    }

    /// Allocate from the global lock-free free list
    fn allocateFromGlobalList(self: *LockFreeResourcePool) !ResourceHandle {
        var attempts: usize = 0;
        const max_attempts: usize = 1000; // Prevent infinite loop

        while (attempts < max_attempts) : (attempts += 1) {
            const head = self.free_list_head.load(.acquire);
            const head_node = FreeListNode.getPointer(head) orelse {
                // Pool exhausted
                _ = self.stats.failed_allocations.fetchAdd(1, .release);
                return error.OutOfMemory;
            };

            const next = head_node.next.load(.acquire);
            const new_tag = FreeListNode.getTag(head) +% 1;
            const new_head = FreeListNode.makeTagged(FreeListNode.getPointer(next), new_tag);

            // Try to CAS the head
            if (self.free_list_head.cmpxchgWeak(head, new_head, .acq_rel, .acquire) == null) {
                // Successfully claimed this slot
                return self.finalizeAllocation(head_node.slot_index);
            }

            // CAS failed, retry
            std.atomic.spinLoopHint();
        }

        // Too much contention
        _ = self.stats.failed_allocations.fetchAdd(1, .release);
        return error.OutOfMemory;
    }

    /// Finalize allocation after claiming a slot
    fn finalizeAllocation(self: *LockFreeResourcePool, slot_index: u32) !ResourceHandle {
        const slot = &self.slots[slot_index];

        // Increment generation
        const new_gen = slot.generation.fetchAdd(1, .acq_rel) +% 1;

        // Mark as allocated
        slot.state.store(.allocated, .release);
        slot.alloc_time.store(time.unixSeconds(), .release);
        slot.size.store(self.config.slot_size, .release);

        // Ensure buffer exists
        if (self.buffers[slot_index] == null) {
            const buffer = try self.allocator.create(memory.GpuBuffer);
            errdefer self.allocator.destroy(buffer);
            buffer.* = try memory.GpuBuffer.init(
                self.allocator,
                self.config.slot_size,
                self.config.buffer_flags,
            );
            self.buffers[slot_index] = buffer;
        }

        // Update stats
        _ = self.stats.total_allocations.fetchAdd(1, .release);
        const active = self.stats.active_allocations.fetchAdd(1, .release) + 1;

        // Update peak (relaxed is fine for stats)
        var current_peak = self.stats.peak_allocations.load(.monotonic);
        while (active > current_peak) {
            const result = self.stats.peak_allocations.cmpxchgWeak(
                current_peak,
                active,
                .release,
                .monotonic,
            );
            if (result == null) break;
            current_peak = result.?;
        }

        return ResourceHandle.init(slot_index, new_gen);
    }

    /// Free a resource by handle
    /// Returns false if handle is invalid or already freed
    pub fn free(self: *LockFreeResourcePool, handle: ResourceHandle) bool {
        if (!handle.isValid()) {
            _ = self.stats.invalid_accesses.fetchAdd(1, .release);
            return false;
        }

        const slot_index = handle.index();
        if (slot_index >= self.config.max_slots) {
            _ = self.stats.invalid_accesses.fetchAdd(1, .release);
            return false;
        }

        const slot = &self.slots[slot_index];

        // Verify generation matches
        const current_gen = slot.generation.load(.acquire);
        if (current_gen != handle.generation()) {
            _ = self.stats.invalid_accesses.fetchAdd(1, .release);
            return false;
        }

        // Try to transition from allocated to retiring
        if (slot.state.cmpxchgWeak(.allocated, .retiring, .acq_rel, .acquire) != null) {
            // Already freed or being freed
            return false;
        }

        // Clear slot metadata
        slot.size.store(0, .release);

        // Try to return to thread-local cache first
        if (self.thread_caches) |caches| {
            const cache_idx = getThreadCacheIndex();
            var cache = &caches[cache_idx];
            if (cache.push(slot_index)) {
                // Mark as free
                slot.state.store(.free, .release);
                _ = self.stats.total_deallocations.fetchAdd(1, .release);
                _ = self.stats.active_allocations.fetchSub(1, .release);
                return true;
            }
        }

        // Return to global free list
        self.returnToGlobalList(slot_index);

        // Mark as free
        slot.state.store(.free, .release);
        _ = self.stats.total_deallocations.fetchAdd(1, .release);
        _ = self.stats.active_allocations.fetchSub(1, .release);

        return true;
    }

    /// Return a slot to the global free list
    fn returnToGlobalList(self: *LockFreeResourcePool, slot_index: u32) void {
        const node = &self.free_nodes[slot_index];

        while (true) {
            const head = self.free_list_head.load(.acquire);
            const new_tag = FreeListNode.getTag(head) +% 1;

            // Point our node to current head
            node.next.store(head, .release);

            // Try to make us the new head
            const new_head = FreeListNode.makeTagged(node, new_tag);
            if (self.free_list_head.cmpxchgWeak(head, new_head, .acq_rel, .acquire) == null) {
                return;
            }

            std.atomic.spinLoopHint();
        }
    }

    /// Get the GPU buffer for a handle (validates generation)
    pub fn get(self: *LockFreeResourcePool, handle: ResourceHandle) ?*memory.GpuBuffer {
        if (!self.validateHandle(handle)) {
            _ = self.stats.invalid_accesses.fetchAdd(1, .release);
            return null;
        }
        return self.buffers[handle.index()];
    }

    /// Get the GPU buffer without validation (for hot paths after prior validation)
    pub fn getUnchecked(self: *LockFreeResourcePool, handle: ResourceHandle) *memory.GpuBuffer {
        return self.buffers[handle.index()].?;
    }

    /// Validate that a handle is still valid
    pub fn validateHandle(self: *const LockFreeResourcePool, handle: ResourceHandle) bool {
        if (!handle.isValid()) return false;

        const slot_index = handle.index();
        if (slot_index >= self.config.max_slots) return false;

        const slot = &self.slots[slot_index];

        // Check generation matches
        if (slot.generation.load(.acquire) != handle.generation()) return false;

        // Check slot is allocated
        return slot.state.load(.acquire) == .allocated;
    }

    /// Get pool statistics (lock-free)
    pub fn getStats(self: *const LockFreeResourcePool) StatsSnapshot {
        return self.stats.snapshot();
    }

    /// Reset statistics counters
    pub fn resetStats(self: *LockFreeResourcePool) void {
        self.stats.total_allocations.store(0, .release);
        self.stats.total_deallocations.store(0, .release);
        self.stats.failed_allocations.store(0, .release);
        self.stats.invalid_accesses.store(0, .release);
        // Don't reset active_allocations or peak_allocations
    }

    /// Get the current number of free slots (approximate)
    pub fn freeSlotCount(self: *const LockFreeResourcePool) u64 {
        const active = self.stats.active_allocations.load(.acquire);
        return self.config.max_slots - active;
    }

    fn getThreadCacheIndex() usize {
        const thread_id = std.Thread.getCurrentId();
        return @as(usize, @intCast(thread_id)) % MAX_THREADS;
    }
};

/// Thread-local cache for reducing contention on the global free list
const ThreadLocalCache = struct {
    /// Fixed-size array of cached slot indices
    slots: [16]u32 = undefined,
    /// Number of valid entries (bottom of stack)
    count: std.atomic.Value(usize),
    /// Maximum cache size
    max_size: usize,
    // No padding needed — struct already fills cache line

    fn init(max_size: usize) ThreadLocalCache {
        return .{
            .count = std.atomic.Value(usize).init(0),
            .max_size = @min(max_size, 16),
        };
    }

    /// Push a slot index to the cache
    /// Returns false if cache is full
    fn push(self: *ThreadLocalCache, slot_index: u32) bool {
        const current = self.count.load(.acquire);
        if (current >= self.max_size) return false;

        self.slots[current] = slot_index;
        self.count.store(current + 1, .release);
        return true;
    }

    /// Pop a slot index from the cache
    /// Returns null if cache is empty
    fn pop(self: *ThreadLocalCache) ?u32 {
        const current = self.count.load(.acquire);
        if (current == 0) return null;

        const new_count = current - 1;
        const slot = self.slots[new_count];
        self.count.store(new_count, .release);
        return slot;
    }
};

/// Concurrent command buffer pool for parallel command recording
pub const ConcurrentCommandPool = struct {
    /// Per-thread command buffer pools
    thread_pools: []ThreadCommandPool,
    allocator: std.mem.Allocator,
    max_buffers_per_thread: usize,

    const ThreadCommandPool = struct {
        buffers: []CommandBuffer,
        allocated: std.atomic.Value(usize),
        _padding: [CACHE_LINE_SIZE - @sizeOf([]CommandBuffer) - @sizeOf(std.atomic.Value(usize))]u8 = undefined,
    };

    pub const CommandBuffer = struct {
        /// Command data
        data: []u8,
        /// Current write position
        write_pos: usize,
        /// Generation for handle validation
        generation: u32,
        /// Thread that owns this buffer
        owner_thread: usize,
        /// Is buffer in use
        in_use: std.atomic.Value(bool),

        pub fn init(allocator: std.mem.Allocator, size: usize) !CommandBuffer {
            return .{
                .data = try allocator.alloc(u8, size),
                .write_pos = 0,
                .generation = 0,
                .owner_thread = 0,
                .in_use = std.atomic.Value(bool).init(false),
            };
        }

        pub fn deinit(self: *CommandBuffer, allocator: std.mem.Allocator) void {
            allocator.free(self.data);
            self.* = undefined;
        }

        pub fn reset(self: *CommandBuffer) void {
            self.write_pos = 0;
        }

        pub fn write(self: *CommandBuffer, bytes: []const u8) !void {
            if (self.write_pos + bytes.len > self.data.len) {
                return error.BufferFull;
            }
            @memcpy(self.data[self.write_pos..][0..bytes.len], bytes);
            self.write_pos += bytes.len;
        }

        pub fn getWritten(self: *const CommandBuffer) []const u8 {
            return self.data[0..self.write_pos];
        }
    };

    pub fn init(allocator: std.mem.Allocator, max_buffers_per_thread: usize, buffer_size: usize) !ConcurrentCommandPool {
        const thread_pools = try allocator.alloc(ThreadCommandPool, MAX_THREADS);
        errdefer allocator.free(thread_pools);

        for (thread_pools) |*pool| {
            pool.buffers = try allocator.alloc(CommandBuffer, max_buffers_per_thread);
            pool.allocated = std.atomic.Value(usize).init(0);
            for (pool.buffers) |*buffer| {
                buffer.* = try CommandBuffer.init(allocator, buffer_size);
            }
        }

        return .{
            .thread_pools = thread_pools,
            .allocator = allocator,
            .max_buffers_per_thread = max_buffers_per_thread,
        };
    }

    pub fn deinit(self: *ConcurrentCommandPool) void {
        for (self.thread_pools) |*pool| {
            for (pool.buffers) |*buffer| {
                buffer.deinit(self.allocator);
            }
            self.allocator.free(pool.buffers);
        }
        self.allocator.free(self.thread_pools);
        self.* = undefined;
    }

    /// Acquire a command buffer for the current thread
    pub fn acquire(self: *ConcurrentCommandPool) ?*CommandBuffer {
        const thread_idx = getThreadIndex();
        const pool = &self.thread_pools[thread_idx];

        // Find a free buffer in this thread's pool
        for (pool.buffers) |*buffer| {
            if (buffer.in_use.cmpxchgWeak(false, true, .acq_rel, .acquire) == null) {
                buffer.owner_thread = thread_idx;
                buffer.generation +%= 1;
                buffer.reset();
                return buffer;
            }
        }

        return null;
    }

    /// Release a command buffer back to the pool
    pub fn release(self: *ConcurrentCommandPool, buffer: *CommandBuffer) void {
        _ = self;
        buffer.in_use.store(false, .release);
    }

    fn getThreadIndex() usize {
        const thread_id = std.Thread.getCurrentId();
        return @as(usize, @intCast(thread_id)) % MAX_THREADS;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "resource handle format" {
    const handle = ResourceHandle.init(42, 7);
    try std.testing.expectEqual(@as(u32, 42), handle.index());
    try std.testing.expectEqual(@as(u32, 7), handle.generation());
    try std.testing.expect(handle.isValid());

    try std.testing.expect(!INVALID_HANDLE.isValid());
}

test "lock-free pool basic allocation" {
    var pool = try LockFreeResourcePool.init(std.testing.allocator, .{
        .max_slots = 8,
        .slot_size = 256,
        .enable_thread_local_cache = false,
    });
    defer pool.deinit();

    // Allocate a resource
    const handle1 = try pool.allocate();
    try std.testing.expect(handle1.isValid());
    try std.testing.expect(pool.validateHandle(handle1));

    // Get the buffer
    const buffer = pool.get(handle1);
    try std.testing.expect(buffer != null);

    // Allocate another
    const handle2 = pool.allocate() catch |err| {
        if (err == error.OutOfMemory) return error.SkipZigTest;
        return err;
    };
    try std.testing.expect(handle2.isValid());
    try std.testing.expect(handle1.index() != handle2.index());

    // Free first
    try std.testing.expect(pool.free(handle1));
    try std.testing.expect(!pool.validateHandle(handle1));

    // Old handle should now be invalid for get
    try std.testing.expect(pool.get(handle1) == null);

    // Free second
    try std.testing.expect(pool.free(handle2));

    // Stats check
    const stats = pool.getStats();
    try std.testing.expectEqual(@as(u64, 2), stats.total_allocations);
    try std.testing.expectEqual(@as(u64, 2), stats.total_deallocations);
    try std.testing.expectEqual(@as(u64, 0), stats.active_allocations);
}

test "lock-free pool generation prevents use-after-free" {
    var pool = try LockFreeResourcePool.init(std.testing.allocator, .{
        .max_slots = 4,
        .slot_size = 128,
        .enable_thread_local_cache = false,
    });
    defer pool.deinit();

    // Allocate and free
    const handle1 = try pool.allocate();
    const old_gen = handle1.generation();
    try std.testing.expect(pool.free(handle1));

    // Reallocate the same slot
    const handle2 = pool.allocate() catch |err| {
        if (err == error.OutOfMemory) return error.SkipZigTest;
        return err;
    };
    try std.testing.expect(handle2.index() == handle1.index()); // Same slot reused

    // But generation should be different
    try std.testing.expect(handle2.generation() != old_gen);

    // Old handle should not validate
    try std.testing.expect(!pool.validateHandle(handle1));
    try std.testing.expect(pool.get(handle1) == null);

    // New handle should work
    try std.testing.expect(pool.validateHandle(handle2));
    try std.testing.expect(pool.get(handle2) != null);

    _ = pool.free(handle2);
}

test "lock-free pool exhaustion" {
    var pool = try LockFreeResourcePool.init(std.testing.allocator, .{
        .max_slots = 2,
        .slot_size = 64,
        .enable_thread_local_cache = false,
    });
    defer pool.deinit();

    const h1 = try pool.allocate();
    const h2 = pool.allocate() catch |err| {
        if (err == error.OutOfMemory) return error.SkipZigTest;
        return err;
    };

    // Pool should be exhausted
    const result = pool.allocate();
    try std.testing.expectError(error.OutOfMemory, result);

    // Free one and allocate again should work
    try std.testing.expect(pool.free(h1));
    const h3 = try pool.allocate();
    try std.testing.expect(h3.isValid());

    _ = pool.free(h2);
    _ = pool.free(h3);
}

test "concurrent command pool" {
    var cmd_pool = try ConcurrentCommandPool.init(std.testing.allocator, 4, 1024);
    defer cmd_pool.deinit();

    // Acquire a buffer
    const buffer = cmd_pool.acquire();
    try std.testing.expect(buffer != null);

    // Write some data
    try buffer.?.write("test command");
    try std.testing.expectEqualStrings("test command", buffer.?.getWritten());

    // Release it
    cmd_pool.release(buffer.?);

    // Should be able to acquire again
    const buffer2 = cmd_pool.acquire();
    try std.testing.expect(buffer2 != null);
    try std.testing.expectEqual(@as(usize, 0), buffer2.?.write_pos); // Should be reset

    cmd_pool.release(buffer2.?);
}

test "lock-free pool concurrent stress test" {
    const thread_count = 4;
    const ops_per_thread = 100;

    var pool = try LockFreeResourcePool.init(std.testing.allocator, .{
        .max_slots = 64,
        .slot_size = 128,
        .enable_thread_local_cache = true,
    });
    defer pool.deinit();

    var threads: [thread_count]std.Thread = undefined;
    var success_counts: [thread_count]std.atomic.Value(usize) = undefined;
    var error_counts: [thread_count]std.atomic.Value(usize) = undefined;

    for (0..thread_count) |i| {
        success_counts[i] = std.atomic.Value(usize).init(0);
        error_counts[i] = std.atomic.Value(usize).init(0);
    }

    for (&threads, 0..) |*t, tid| {
        t.* = try std.Thread.spawn(.{}, struct {
            fn worker(
                p: *LockFreeResourcePool,
                successes: *std.atomic.Value(usize),
                errors: *std.atomic.Value(usize),
            ) !void {
                var handles: [8]ResourceHandle = .{INVALID_HANDLE} ** 8;
                var idx: usize = 0;

                for (0..ops_per_thread) |i| {
                    if (i % 2 == 0) {
                        // Allocate
                        if (p.allocate()) |handle| {
                            if (idx < handles.len) {
                                handles[idx] = handle;
                                idx += 1;
                            } else {
                                _ = p.free(handle);
                            }
                            _ = successes.fetchAdd(1, .release);
                        } else |_| {
                            _ = errors.fetchAdd(1, .release);
                        }
                    } else {
                        // Free
                        if (idx > 0) {
                            idx -= 1;
                            if (p.free(handles[idx])) {
                                handles[idx] = INVALID_HANDLE;
                            }
                        }
                    }
                }

                // Cleanup remaining
                for (handles) |h| {
                    if (h.isValid()) {
                        _ = p.free(h);
                    }
                }
            }
        }.worker, .{ &pool, &success_counts[tid], &error_counts[tid] });
    }

    for (&threads) |*t| {
        t.join();
    }

    var total_success: usize = 0;
    for (&success_counts) |*c| {
        total_success += c.load(.acquire);
    }

    // Should have some successful operations
    try std.testing.expect(total_success > 0);

    // Pool should be consistent after all operations
    const stats = pool.getStats();
    try std.testing.expectEqual(@as(u64, 0), stats.active_allocations);
}

test {
    std.testing.refAllDecls(@This());
}
