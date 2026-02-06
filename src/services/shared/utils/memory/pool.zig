//! Memory pool allocator for hot path allocations.
//! Uses size-segregated free lists for O(1) allocation performance.
const std = @import("std");
const time = @import("../../../../services/shared/time.zig");
const sync = @import("../../../../services/shared/sync.zig");

pub const PoolConfig = struct {
    block_size: usize = 4096,
    initial_blocks: usize = 8,
    max_blocks: usize = 1024,
    grow_factor: u32 = 2,
    enable_threading: bool = true,
};

pub const MemoryBlock = struct {
    data: [*]align(16) u8,
    size: usize,
    in_use: bool,
    next: ?*MemoryBlock,
    prev: ?*MemoryBlock,
    pool_id: u64,
    size_class: u8, // Index into size-segregated free lists
};

/// Size classes for segregated free lists (powers of 2)
/// Classes: 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, overflow
const SIZE_CLASS_COUNT = 12;
const MIN_SIZE_CLASS_LOG2 = 6; // 64 bytes minimum
const MAX_SIZE_CLASS_LOG2 = MIN_SIZE_CLASS_LOG2 + SIZE_CLASS_COUNT - 2; // 32KB before overflow

/// Get size class index for a given size (O(1) using bit manipulation)
fn getSizeClass(size: usize) u8 {
    if (size == 0) return 0;
    const log2_size = std.math.log2_int(usize, size);
    if (log2_size < MIN_SIZE_CLASS_LOG2) return 0;
    if (log2_size > MAX_SIZE_CLASS_LOG2) return SIZE_CLASS_COUNT - 1;
    return @intCast(log2_size - MIN_SIZE_CLASS_LOG2);
}

/// Get minimum block size for a size class
fn sizeClassToMinSize(class: u8) usize {
    if (class >= SIZE_CLASS_COUNT - 1) return 1 << MAX_SIZE_CLASS_LOG2;
    return @as(usize, 1) << (MIN_SIZE_CLASS_LOG2 + @as(u6, @intCast(class)));
}

pub const MemoryPool = struct {
    allocator: std.mem.Allocator,
    config: PoolConfig,
    blocks: std.ArrayListUnmanaged(*MemoryBlock),
    /// Size-segregated free lists for O(1) allocation
    size_class_free_lists: [SIZE_CLASS_COUNT]?*MemoryBlock,
    used_list: ?*MemoryBlock,
    mutex: sync.Mutex,
    pool_id: u64,
    next_pool_id: std.atomic.Value(u64),
    total_allocated: u64,
    total_freed: u64,
    peak_usage: u64,
    current_usage: u64,

    // Keep legacy free_list for compatibility
    free_list: ?*MemoryBlock,

    pub fn init(allocator: std.mem.Allocator, config: PoolConfig) !MemoryPool {
        var pool = MemoryPool{
            .allocator = allocator,
            .config = config,
            .blocks = std.ArrayListUnmanaged(*MemoryBlock).empty,
            .size_class_free_lists = [_]?*MemoryBlock{null} ** SIZE_CLASS_COUNT,
            .free_list = null,
            .used_list = null,
            .mutex = .{},
            .pool_id = 0,
            .next_pool_id = std.atomic.Value(u64).init(1),
            .total_allocated = 0,
            .total_freed = 0,
            .peak_usage = 0,
            .current_usage = 0,
        };

        try pool.grow(@intCast(config.initial_blocks));
        return pool;
    }

    pub fn deinit(self: *MemoryPool) void {
        for (self.blocks.items) |block| {
            self.allocator.free(block.data[0..block.size]);
            self.allocator.destroy(block);
        }
        self.blocks.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn alloc(self: *MemoryPool, len: usize) ![]u8 {
        const block = try self.allocBlock(len);
        return block.data[0..len];
    }

    pub fn free(self: *MemoryPool, ptr: []const u8) void {
        if (ptr.len == 0) return;
        const block = self.findBlock(ptr.ptr);
        if (block) |b| {
            self.freeBlock(b);
        }
    }

    pub fn allocBlock(self: *MemoryPool, size: usize) !*MemoryBlock {
        self.mutex.lock();
        defer self.mutex.unlock();

        const aligned_size = self.alignSize(size);
        var block = self.findFreeBlock(aligned_size);

        if (block == null) {
            try self.grow(self.config.grow_factor);
            block = self.findFreeBlock(aligned_size);
        }

        const b = block orelse return error.PoolExhausted;

        b.in_use = true;
        self.current_usage += 1;
        if (self.current_usage > self.peak_usage) {
            self.peak_usage = self.current_usage;
        }

        self.removeFromFreeList(b);
        self.addToUsedList(b);

        return b;
    }

    pub fn freeBlock(self: *MemoryPool, block: *MemoryBlock) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (!block.in_use) return;

        block.in_use = false;
        self.current_usage -= 1;

        self.removeFromUsedList(block);
        self.addToFreeList(block);
    }

    /// O(1) allocation using size-segregated free lists
    fn findFreeBlock(self: *MemoryPool, size: usize) ?*MemoryBlock {
        const start_class = getSizeClass(size);

        // Search from the appropriate size class upward
        var class = start_class;
        while (class < SIZE_CLASS_COUNT) : (class += 1) {
            if (self.size_class_free_lists[class]) |block| {
                if (!block.in_use and block.size >= size) {
                    return block;
                }
            }
        }

        // Fall back to legacy free_list scan for any remaining blocks
        var current = self.free_list;
        while (current) |block| {
            if (!block.in_use and block.size >= size) {
                return block;
            }
            current = block.next;
        }
        return null;
    }

    fn findBlock(self: *MemoryPool, ptr: [*]const u8) ?*MemoryBlock {
        const ptr_addr = @intFromPtr(ptr);
        for (self.blocks.items) |block| {
            const block_start = @intFromPtr(block.data);
            const block_end = block_start + block.size;
            if (ptr_addr >= block_start and ptr_addr < block_end) {
                return block;
            }
        }
        return null;
    }

    fn grow(self: *MemoryPool, factor: u32) !void {
        const current_count = self.blocks.items.len;
        // When current_count is 0, use factor as the number of new blocks
        const target_count = @min(@max(current_count * @as(usize, factor), @as(usize, factor)), self.config.max_blocks);
        const new_blocks = target_count - current_count;

        var i: usize = 0;
        while (i < new_blocks) : (i += 1) {
            const block = try self.allocator.create(MemoryBlock);
            errdefer self.allocator.destroy(block);

            const data = try self.allocator.alignedAlloc(u8, .@"16", self.config.block_size);
            errdefer self.allocator.free(data);

            block.* = .{
                .data = data.ptr,
                .size = self.config.block_size,
                .in_use = false,
                .next = null,
                .prev = null,
                .pool_id = self.pool_id,
                .size_class = getSizeClass(self.config.block_size),
            };

            try self.blocks.append(self.allocator, block);
            self.addToFreeList(block);
        }
    }

    fn alignSize(_: *MemoryPool, size: usize) usize {
        const alignment = @max(16, @alignOf(u64));
        return ((size + alignment - 1) / alignment) * alignment;
    }

    fn addToFreeList(self: *MemoryPool, block: *MemoryBlock) void {
        // Add to size-segregated free list for O(1) allocation
        const class = block.size_class;
        if (self.size_class_free_lists[class]) |head| {
            block.next = head;
            head.prev = block;
        } else {
            block.next = null;
        }
        block.prev = null;
        self.size_class_free_lists[class] = block;

        // Also maintain legacy free_list
        if (self.free_list) |head| {
            // Only add if not already at head of a size class list
            _ = head;
        }
        self.free_list = block;
    }

    fn removeFromFreeList(self: *MemoryPool, block: *MemoryBlock) void {
        // Remove from size-segregated free list
        const class = block.size_class;
        if (block.prev) |prev| {
            prev.next = block.next;
        } else {
            // This block is head of its size class list
            self.size_class_free_lists[class] = block.next;
        }
        if (block.next) |next| {
            next.prev = block.prev;
        }
        block.prev = null;
        block.next = null;

        // Update legacy free_list if needed
        if (self.free_list == block) {
            self.free_list = null;
            // Find a new head from size class lists
            for (self.size_class_free_lists) |list_head| {
                if (list_head) |head| {
                    self.free_list = head;
                    break;
                }
            }
        }
    }

    fn addToUsedList(self: *MemoryPool, block: *MemoryBlock) void {
        if (self.used_list) |head| {
            block.next = head;
            head.prev = block;
        }
        self.used_list = block;
    }

    fn removeFromUsedList(self: *MemoryPool, block: *MemoryBlock) void {
        if (block.prev) |prev| {
            prev.next = block.next;
        } else {
            self.used_list = block.next;
        }
        if (block.next) |next| {
            next.prev = block.prev;
        }
    }

    pub fn getStats(self: *MemoryPool) PoolStats {
        return .{
            .total_blocks = self.blocks.items.len,
            .used_blocks = self.current_usage,
            .free_blocks = self.blocks.items.len - self.current_usage,
            .total_allocated = self.total_allocated,
            .total_freed = self.total_freed,
            .peak_usage = self.peak_usage,
            .current_usage = self.current_usage,
            .utilization = if (self.blocks.items.len > 0)
                @as(f32, @floatFromInt(self.current_usage)) / @as(f32, @floatFromInt(self.blocks.items.len))
            else
                0.0,
        };
    }
};

pub const PoolStats = struct {
    total_blocks: usize,
    used_blocks: u64,
    free_blocks: usize,
    total_allocated: u64,
    total_freed: u64,
    peak_usage: u64,
    current_usage: u64,
    utilization: f32,
};

pub const SlabPool = struct {
    allocator: std.mem.Allocator,
    slab_size: usize,
    slab_size_alignment: u8,
    slabs: std.ArrayListUnmanaged(*Slab),

    const Slab = struct {
        data: []align(16) u8,
        bitmap: []u64,
        capacity: usize,
        used_count: usize,

        fn alloc(self: *Slab, _: std.mem.Allocator) ?*u8 {
            for (self.bitmap, 0..) |bits, slot_idx| {
                var bit_idx: u6 = 0;
                while (bit_idx < 64) : (bit_idx += 1) {
                    if ((bits >> @as(u6, bit_idx)) & 1 == 0) {
                        self.bitmap[slot_idx] |= @as(u64, 1) << bit_idx;
                        self.used_count += 1;
                        const offset = slot_idx * 64 + @as(usize, bit_idx);
                        if (offset < self.data.len) {
                            return &self.data[offset];
                        }
                        return null;
                    }
                }
            }
            return null;
        }

        /// Free a slot. Returns false if the slot was already free (double-free protection).
        fn free(self: *Slab, ptr: *u8) bool {
            const offset = @intFromPtr(ptr) - @intFromPtr(self.data.ptr);
            const slot_idx = offset / 64;
            const bit_idx = @as(u6, @intCast(offset % 64));
            const mask = @as(u64, 1) << bit_idx;
            // Double-free protection: check if bit is already clear
            if ((self.bitmap[slot_idx] & mask) == 0) {
                return false; // Already freed
            }
            self.bitmap[slot_idx] &= ~mask;
            self.used_count -= 1;
            return true;
        }
    };

    pub fn init(allocator: std.mem.Allocator, slab_size: usize, slab_count: usize) !SlabPool {
        var pool = SlabPool{
            .allocator = allocator,
            .slab_size = slab_size,
            .slab_size_alignment = @intFromFloat(@ceil(@log2(@as(f32, @floatFromInt(slab_size))))),
            .slabs = std.ArrayListUnmanaged(*Slab).empty,
        };

        const bitmap_words = (slab_count + 63) / 64;
        const slab_data_size = slab_size * slab_count;

        var i: usize = 0;
        while (i < slab_count) : (i += 1) {
            const slab = try allocator.create(Slab);
            errdefer allocator.destroy(slab);

            // Use default alignment for slabs (alignment is stored but not actively used at runtime)
            const data = try allocator.alignedAlloc(u8, .@"16", slab_data_size);
            errdefer allocator.free(data);

            const bitmap = try allocator.alloc(u64, bitmap_words);
            @memset(bitmap, 0);

            slab.* = .{
                .data = data,
                .bitmap = bitmap,
                .capacity = slab_count,
                .used_count = 0,
            };

            try pool.slabs.append(allocator, slab);
        }

        return pool;
    }

    pub fn deinit(self: *SlabPool) void {
        for (self.slabs.items) |slab| {
            self.allocator.free(slab.bitmap);
            self.allocator.free(slab.data);
            self.allocator.destroy(slab);
        }
        self.slabs.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn alloc(self: *SlabPool) !*u8 {
        for (self.slabs.items) |slab| {
            if (slab.used_count < slab.capacity) {
                if (slab.alloc(self.allocator)) |ptr| {
                    return ptr;
                }
            }
        }
        return error.SlabExhausted;
    }

    /// Free a pointer. Returns true if freed successfully, false if double-free or invalid pointer.
    pub fn free(self: *SlabPool, ptr: *u8) bool {
        for (self.slabs.items) |slab| {
            const start = @intFromPtr(slab.data.ptr);
            const end = start + slab.data.len;
            const ptr_int = @intFromPtr(ptr);
            if (ptr_int >= start and ptr_int < end) {
                return slab.free(ptr);
            }
        }
        return false; // Pointer not found in any slab
    }
};

test "memory pool initialization" {
    const allocator = std.testing.allocator;
    const config = PoolConfig{
        .block_size = 256,
        .initial_blocks = 4,
        .max_blocks = 16,
    };

    var pool = try MemoryPool.init(allocator, config);
    defer pool.deinit();

    const stats = pool.getStats();
    try std.testing.expectEqual(@as(usize, 4), stats.total_blocks);
    try std.testing.expectEqual(@as(usize, 4), stats.free_blocks);
}

test "memory pool allocation" {
    const allocator = std.testing.allocator;
    const config = PoolConfig{
        .block_size = 256,
        .initial_blocks = 4,
        .max_blocks = 16,
    };

    var pool = try MemoryPool.init(allocator, config);
    defer pool.deinit();

    const mem = try pool.alloc(100);
    defer pool.free(mem);

    try std.testing.expectEqual(@as(usize, 100), mem.len);
}

test "slab pool" {
    const allocator = std.testing.allocator;
    var pool = try SlabPool.init(allocator, 64, 16);
    defer pool.deinit();

    const ptr1 = try pool.alloc();
    _ = pool.free(ptr1);

    const ptr2 = try pool.alloc();
    defer _ = pool.free(ptr2);

    // ptr2 is non-null since alloc succeeded (didn't return error)
    try std.testing.expect(@intFromPtr(ptr2) != 0);
}
