//! Kernel Descriptor Ring Buffer
//!
//! Lock-free ring buffer for caching recent kernel launch configurations.
//! Enables fast-path kernel launches by reusing cached descriptors.

const std = @import("std");

/// Lock-free ring buffer for kernel launch descriptors.
/// Enables fast-path kernel launches by caching recent configurations.
pub const KernelRing = struct {
    pub const CAPACITY: u32 = 256;

    /// Kernel launch descriptor.
    pub const Descriptor = struct {
        kernel_handle: u64,
        grid_dim: [3]u32,
        block_dim: [3]u32,
        shared_mem: u32,

        /// Compute hash for fast lookup.
        pub fn hash(self: Descriptor) u64 {
            var h: u64 = self.kernel_handle;
            h ^= @as(u64, self.grid_dim[0]) << 32;
            h ^= @as(u64, self.grid_dim[1]) << 16;
            h ^= @as(u64, self.grid_dim[2]);
            h ^= @as(u64, self.block_dim[0]) << 48;
            h ^= @as(u64, self.block_dim[1]) << 40;
            h ^= @as(u64, self.block_dim[2]) << 32;
            h ^= @as(u64, self.shared_mem);
            return h;
        }

        /// Check equality of two descriptors.
        pub fn eql(a: Descriptor, b: Descriptor) bool {
            return a.kernel_handle == b.kernel_handle and
                std.mem.eql(u32, &a.grid_dim, &b.grid_dim) and
                std.mem.eql(u32, &a.block_dim, &b.block_dim) and
                a.shared_mem == b.shared_mem;
        }
    };

    /// Ring buffer storage.
    buffer: [CAPACITY]Descriptor,
    /// Hash lookup table for fast reuse detection.
    lookup: [CAPACITY]u64,
    /// Head pointer (oldest entry).
    head: std.atomic.Value(u32),
    /// Tail pointer (next write position).
    tail: std.atomic.Value(u32),

    /// Initialize an empty ring buffer.
    pub fn init() KernelRing {
        return .{
            .buffer = [_]Descriptor{.{
                .kernel_handle = 0,
                .grid_dim = .{ 0, 0, 0 },
                .block_dim = .{ 0, 0, 0 },
                .shared_mem = 0,
            }} ** CAPACITY,
            .lookup = [_]u64{0} ** CAPACITY,
            .head = std.atomic.Value(u32).init(0),
            .tail = std.atomic.Value(u32).init(0),
        };
    }

    /// Push a new descriptor to the ring buffer.
    /// Returns the slot index where descriptor was stored.
    pub fn push(self: *KernelRing, desc: Descriptor) ?u32 {
        const tail = self.tail.load(.acquire);
        const next_tail = (tail + 1) % CAPACITY;

        // Check if full - advance head to make room (drop oldest)
        const head = self.head.load(.acquire);
        if (next_tail == head) {
            _ = self.head.fetchAdd(1, .release);
        }

        self.buffer[tail] = desc;
        self.lookup[tail] = desc.hash();
        self.tail.store(next_tail, .release);

        return tail;
    }

    /// Push a descriptor, or reuse an existing slot if a matching descriptor exists.
    /// Returns the slot index (either existing or new).
    pub fn pushOrReuse(self: *KernelRing, desc: Descriptor) ?u32 {
        const target_hash = desc.hash();

        // Check recent entries for match (most recent first)
        const head = self.head.load(.acquire);
        const tail = self.tail.load(.acquire);

        if (tail == head) {
            // Empty buffer, just push
            return self.push(desc);
        }

        var i = tail;
        var checked: u32 = 0;
        const max_check: u32 = 16; // Only check last 16 entries for performance

        while (checked < max_check) {
            // Move backwards from tail
            i = if (i == 0) CAPACITY - 1 else i - 1;

            if (self.lookup[i] == target_hash and Descriptor.eql(self.buffer[i], desc)) {
                return i; // Found matching descriptor
            }

            checked += 1;

            // Stop if we've reached head
            if (i == head) break;
        }

        // No match found, push new
        return self.push(desc);
    }

    /// Get descriptor at given slot.
    pub fn get(self: *const KernelRing, slot: u32) Descriptor {
        return self.buffer[slot % CAPACITY];
    }

    /// Get current count of entries in the ring.
    pub fn count(self: *const KernelRing) u32 {
        const head = self.head.load(.acquire);
        const tail = self.tail.load(.acquire);
        if (tail >= head) {
            return tail - head;
        } else {
            return CAPACITY - head + tail;
        }
    }

    /// Check if the ring is empty.
    pub fn isEmpty(self: *const KernelRing) bool {
        return self.head.load(.acquire) == self.tail.load(.acquire);
    }

    /// Clear all entries from the ring.
    pub fn clear(self: *KernelRing) void {
        self.head.store(0, .release);
        self.tail.store(0, .release);
    }
};

// Tests
test "KernelRing stores and retrieves kernel descriptors" {
    var ring = KernelRing.init();

    const desc = KernelRing.Descriptor{
        .kernel_handle = 42,
        .grid_dim = .{ 64, 1, 1 },
        .block_dim = .{ 256, 1, 1 },
        .shared_mem = 0,
    };

    const slot = ring.push(desc);
    try std.testing.expect(slot != null);

    const retrieved = ring.get(slot.?);
    try std.testing.expectEqual(@as(u64, 42), retrieved.kernel_handle);
    try std.testing.expectEqual(@as(u32, 64), retrieved.grid_dim[0]);
}

test "KernelRing fast-path reuses recent descriptors" {
    var ring = KernelRing.init();

    const desc = KernelRing.Descriptor{
        .kernel_handle = 100,
        .grid_dim = .{ 32, 32, 1 },
        .block_dim = .{ 16, 16, 1 },
        .shared_mem = 4096,
    };

    // Push descriptor
    const slot1 = ring.push(desc);

    // Try to push same descriptor again - should reuse
    const slot2 = ring.pushOrReuse(desc);

    // Should reuse the same slot
    try std.testing.expectEqual(slot1, slot2);
}

test "KernelRing wraps around when full" {
    var ring = KernelRing.init();

    // Fill the ring and then some
    var i: u64 = 0;
    while (i < KernelRing.CAPACITY + 10) : (i += 1) {
        const desc = KernelRing.Descriptor{
            .kernel_handle = i,
            .grid_dim = .{ 1, 1, 1 },
            .block_dim = .{ 1, 1, 1 },
            .shared_mem = 0,
        };
        _ = ring.push(desc);
    }

    // Ring should still be functional and not exceed capacity
    try std.testing.expect(ring.count() <= KernelRing.CAPACITY);
    try std.testing.expect(ring.count() > 0);
}

test "KernelRing descriptor hash and equality" {
    const desc1 = KernelRing.Descriptor{
        .kernel_handle = 1,
        .grid_dim = .{ 64, 64, 1 },
        .block_dim = .{ 16, 16, 1 },
        .shared_mem = 1024,
    };

    const desc2 = KernelRing.Descriptor{
        .kernel_handle = 1,
        .grid_dim = .{ 64, 64, 1 },
        .block_dim = .{ 16, 16, 1 },
        .shared_mem = 1024,
    };

    const desc3 = KernelRing.Descriptor{
        .kernel_handle = 2,
        .grid_dim = .{ 64, 64, 1 },
        .block_dim = .{ 16, 16, 1 },
        .shared_mem = 1024,
    };

    try std.testing.expectEqual(desc1.hash(), desc2.hash());
    try std.testing.expect(KernelRing.Descriptor.eql(desc1, desc2));
    try std.testing.expect(!KernelRing.Descriptor.eql(desc1, desc3));
}

test "KernelRing isEmpty and clear" {
    var ring = KernelRing.init();

    try std.testing.expect(ring.isEmpty());

    const desc = KernelRing.Descriptor{
        .kernel_handle = 1,
        .grid_dim = .{ 1, 1, 1 },
        .block_dim = .{ 1, 1, 1 },
        .shared_mem = 0,
    };

    _ = ring.push(desc);
    try std.testing.expect(!ring.isEmpty());

    ring.clear();
    try std.testing.expect(ring.isEmpty());
}

test {
    std.testing.refAllDecls(@This());
}
