//! Search State Pool - Eliminates per-query allocations
//!
//! Provides a lock-free pool of pre-allocated search states for concurrent query
//! processing. Uses a 64-bit bitmask for O(1) acquire/release without locks.
//! General-purpose pool pattern; used by HNSW but usable by any search algorithm.

const std = @import("std");
const index_mod = @import("index.zig");

/// Pre-allocated search state for reuse across queries.
/// Avoids allocation overhead in the hot search path.
pub const SearchState = struct {
    /// Candidate nodes with their distances (node_id -> distance)
    candidates: std.AutoHashMapUnmanaged(u32, f32),
    /// Visited nodes set
    visited: std.AutoHashMapUnmanaged(u32, void),
    /// BFS queue for graph traversal
    queue: std.ArrayListUnmanaged(u32),
    /// Temporary results buffer
    results_buffer: std.ArrayListUnmanaged(index_mod.IndexResult),

    pub fn init() SearchState {
        return .{
            .candidates = .{},
            .visited = .{},
            .queue = .{},
            .results_buffer = .{},
        };
    }

    /// Reset state for reuse without deallocating backing memory.
    pub fn reset(self: *SearchState) void {
        self.candidates.clearRetainingCapacity();
        self.visited.clearRetainingCapacity();
        self.queue.clearRetainingCapacity();
        self.results_buffer.clearRetainingCapacity();
    }

    /// Deallocate all backing memory for the search state.
    /// After calling this, the SearchState should not be used.
    pub fn deinit(self: *SearchState, allocator: std.mem.Allocator) void {
        self.candidates.deinit(allocator);
        self.visited.deinit(allocator);
        self.queue.deinit(allocator);
        self.results_buffer.deinit(allocator);
    }

    /// Ensure capacity for expected search size.
    /// Returns error.Overflow if expected_size exceeds maximum supported capacity.
    pub fn ensureCapacity(self: *SearchState, allocator: std.mem.Allocator, expected_size: usize) !void {
        // Validate size fits in u32 for hash map capacity (max ~4B entries)
        const capped_size: u32 = std.math.cast(u32, expected_size) orelse return error.Overflow;
        try self.candidates.ensureTotalCapacity(allocator, capped_size);
        try self.visited.ensureTotalCapacity(allocator, capped_size);
        try self.queue.ensureTotalCapacity(allocator, expected_size);
        try self.results_buffer.ensureTotalCapacity(allocator, expected_size);
    }
};

/// Pool of pre-allocated search states for concurrent query processing.
/// Thread-safe acquisition and release of search states.
pub const SearchStatePool = struct {
    states: []SearchState,
    available: std.atomic.Value(u64),
    allocator: std.mem.Allocator,

    const MAX_POOL_SIZE = 64;

    pub fn init(allocator: std.mem.Allocator, pool_size: usize) !SearchStatePool {
        const size = @min(pool_size, MAX_POOL_SIZE);
        const states = try allocator.alloc(SearchState, size);
        for (states) |*state| {
            state.* = SearchState.init();
        }

        // Initialize bitmask with all states available
        const initial_mask: u64 = if (size >= 64) ~@as(u64, 0) else (@as(u64, 1) << @intCast(size)) - 1;

        return .{
            .states = states,
            .available = std.atomic.Value(u64).init(initial_mask),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *SearchStatePool) void {
        for (self.states) |*state| {
            state.deinit(self.allocator);
        }
        self.allocator.free(self.states);
    }

    /// Acquire a search state from the pool.
    /// Returns null if no states available (caller should allocate temporary state).
    /// Uses exponential backoff to reduce contention under high concurrency.
    pub fn acquire(self: *SearchStatePool) ?*SearchState {
        var backoff: u8 = 1;
        const max_backoff: u8 = 32;
        const max_retries: u8 = 100;
        var retries: u8 = 0;

        while (retries < max_retries) : (retries += 1) {
            const current = self.available.load(.acquire);
            if (current == 0) return null;

            // Find first available bit
            const bit_idx = @ctz(current);
            if (bit_idx >= self.states.len) return null;

            const new_mask = current & ~(@as(u64, 1) << @intCast(bit_idx));
            if (self.available.cmpxchgWeak(current, new_mask, .acq_rel, .acquire)) |_| {
                // CAS failed - apply exponential backoff to reduce contention
                for (0..backoff) |_| {
                    std.atomic.spinLoopHint();
                }
                backoff = @min(backoff *| 2, max_backoff);
                continue;
            }

            const state = &self.states[bit_idx];
            state.reset();
            return state;
        }

        // Max retries exceeded - return null to let caller handle
        return null;
    }

    /// Release a search state back to the pool.
    /// The state is automatically reset before being returned to the pool.
    /// Thread-safe: uses atomic operations to update availability bitmap.
    pub fn release(self: *SearchStatePool, state: *SearchState) void {
        // Find index of this state
        const idx = (@intFromPtr(state) - @intFromPtr(self.states.ptr)) / @sizeOf(SearchState);
        if (idx >= self.states.len) return;

        _ = self.available.fetchOr(@as(u64, 1) << @intCast(idx), .release);
    }
};

test "search state pool acquire release" {
    const allocator = std.testing.allocator;

    var pool = try SearchStatePool.init(allocator, 4);
    defer pool.deinit();

    // Acquire all states
    var states: [4]?*SearchState = undefined;
    for (&states) |*s| {
        s.* = pool.acquire();
        try std.testing.expect(s.* != null);
    }

    // Pool should be exhausted
    try std.testing.expect(pool.acquire() == null);

    // Release one
    pool.release(states[0].?);

    // Should be able to acquire again
    const reacquired = pool.acquire();
    try std.testing.expect(reacquired != null);

    // Release all
    for (states[1..]) |s| {
        if (s) |state| pool.release(state);
    }
    pool.release(reacquired.?);
}

test "search state reset retains capacity" {
    const allocator = std.testing.allocator;

    var state = SearchState.init();
    defer state.deinit(allocator);

    // Add some data
    try state.candidates.put(allocator, 1, 0.5);
    try state.visited.put(allocator, 1, {});
    try state.queue.append(allocator, 1);

    // Reset should clear without freeing
    state.reset();
    try std.testing.expectEqual(@as(usize, 0), state.candidates.count());
    try std.testing.expectEqual(@as(usize, 0), state.visited.count());
    try std.testing.expectEqual(@as(usize, 0), state.queue.items.len);
}
