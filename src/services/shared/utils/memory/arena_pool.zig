// ============================================================================
// ABI Framework — Arena Pool (Bump Allocator)
// Adapted from abi-system-v2.0/memory.zig
// ============================================================================
//
// Bump allocator backed by a contiguous memory region.
// Allocations are O(1) — just advance a pointer. Reset is O(1) — just
// rewind the pointer. No individual frees.
//
// Also includes VectorPool (SIMD-aligned), SlabPool (fixed-size object pool),
// and ScratchAllocator (double-buffered arenas).
// ============================================================================

const std = @import("std");
const builtin = @import("builtin");

// ─── Platform Constants ────────────────────────────────────────────────────

const cache_line_size: usize = if (builtin.cpu.arch == .aarch64) 128 else 64;
const page_size: usize = 4096;
const simd_alignment: usize = std.simd.suggestVectorLength(f32) orelse 4;
const simd_byte_width: usize = simd_alignment * @sizeOf(f32);

// ─── Math Helpers (delegated to v2 primitives) ────────────────────────────
const Math = @import("../v2_primitives.zig").Math;

const isPowerOfTwo = Math.isPowerOfTwo;

fn alignUpVal(value: usize, alignment: usize) usize {
    return Math.alignUp(usize, value, alignment);
}

inline fn alignUpComptime(value: usize, comptime alignment: usize) usize {
    return Math.alignUpComptime(usize, value, alignment);
}

// ─── Arena Pool ────────────────────────────────────────────────────────────

/// Bump allocator backed by a contiguous memory region.
/// Allocations are O(1) — just advance a pointer. Reset is O(1) — just
/// rewind the pointer. No individual frees.
pub const ArenaPool = struct {
    const Self = @This();

    buffer: []align(cache_line_size) u8,
    offset: usize,
    peak: usize,
    backing: std.mem.Allocator,
    alloc_count: usize = 0,

    pub const Config = struct {
        size: usize = 1024 * 1024, // 1 MiB default
    };

    pub fn init(backing: std.mem.Allocator, config: Config) !Self {
        const size = alignUpVal(config.size, page_size);
        const buf = try backing.alignedAlloc(u8, cache_line_size, size);
        return Self{
            .buffer = buf,
            .offset = 0,
            .peak = 0,
            .backing = backing,
        };
    }

    pub fn deinit(self: *Self) void {
        self.backing.free(self.buffer);
        self.* = undefined;
    }

    /// Allocate `size` bytes with the given alignment.
    /// Returns null if the arena is exhausted.
    pub fn alloc(self: *Self, size: usize, comptime alignment: usize) ?[]align(alignment) u8 {
        comptime std.debug.assert(isPowerOfTwo(alignment));

        const aligned_offset = alignUpComptime(self.offset, alignment);
        const end = aligned_offset + size;

        if (end > self.buffer.len) return null;

        self.offset = end;
        self.peak = @max(self.peak, end);
        self.alloc_count += 1;

        const ptr: [*]align(alignment) u8 = @alignCast(self.buffer.ptr + aligned_offset);
        return ptr[0..size];
    }

    /// Type-safe allocation
    pub fn create(self: *Self, comptime T: type) ?*T {
        const bytes = self.alloc(@sizeOf(T), @alignOf(T)) orelse return null;
        return @ptrCast(bytes.ptr);
    }

    /// Allocate a typed slice
    pub fn allocSlice(self: *Self, comptime T: type, count: usize) ?[]T {
        const bytes = self.alloc(@sizeOf(T) * count, @alignOf(T)) orelse return null;
        const ptr: [*]T = @ptrCast(@alignCast(bytes.ptr));
        return ptr[0..count];
    }

    /// Reset to beginning — O(1), no deallocation
    pub fn reset(self: *Self) void {
        self.offset = 0;
        self.alloc_count = 0;
    }

    pub fn usedBytes(self: *const Self) usize {
        return self.offset;
    }

    pub fn remainingBytes(self: *const Self) usize {
        return self.buffer.len - self.offset;
    }

    pub fn utilizationPercent(self: *const Self) f64 {
        return @as(f64, @floatFromInt(self.offset)) / @as(f64, @floatFromInt(self.buffer.len)) * 100.0;
    }

    /// Wrap as std.mem.Allocator for interop with standard library
    pub fn allocator(self: *Self) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = arenaAllocFn,
                .resize = arenaResizeFn,
                .remap = arenaRemapFn,
                .free = arenaFreeFn,
            },
        };
    }

    fn arenaAllocFn(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, _: usize) ?[*]u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));
        const align_bytes = alignment.toByteUnits();
        const aligned_offset = std.mem.alignForward(usize, self.offset, align_bytes);
        const end = aligned_offset + len;
        if (end > self.buffer.len) return null;
        self.offset = end;
        self.peak = @max(self.peak, end);
        self.alloc_count += 1;
        return self.buffer.ptr + aligned_offset;
    }

    fn arenaResizeFn(_: *anyopaque, _: [*]u8, _: usize, _: usize, _: std.mem.Alignment, _: usize) bool {
        return false; // arena doesn't support resize
    }

    fn arenaRemapFn(_: *anyopaque, _: [*]u8, _: usize, _: usize, _: std.mem.Alignment, _: usize) ?[*]u8 {
        return null; // arena doesn't support remap
    }

    fn arenaFreeFn(_: *anyopaque, _: [*]u8, _: usize, _: std.mem.Alignment, _: usize) void {
        // Arena doesn't free individual allocations — use reset()
    }
};

// ─── Vector Pool ───────────────────────────────────────────────────────────

/// SIMD-aligned memory pool for compute buffers.
/// Allocations are aligned to the platform's SIMD width ensuring
/// vectorized loads/stores never cross cache line boundaries.
pub const VectorPool = struct {
    const Self = @This();

    pub const vector_alignment: usize = @max(simd_byte_width, cache_line_size);

    arena: ArenaPool,
    vector_count: usize = 0,

    pub const Config = struct {
        size: usize = 4 * 1024 * 1024, // 4 MiB default
    };

    pub fn init(backing: std.mem.Allocator, config: Config) !Self {
        return Self{
            .arena = try ArenaPool.init(backing, .{ .size = config.size }),
        };
    }

    pub fn deinit(self: *Self) void {
        self.arena.deinit();
    }

    /// Allocate a SIMD-aligned f32 vector
    pub fn allocF32(self: *Self, count: usize) ?[]align(vector_alignment) f32 {
        const lanes = simd_alignment;
        const padded = alignUpVal(count, lanes);
        const byte_size = padded * @sizeOf(f32);

        const bytes = self.arena.alloc(byte_size, vector_alignment) orelse return null;
        self.vector_count += 1;

        const ptr: [*]align(vector_alignment) f32 = @ptrCast(@alignCast(bytes.ptr));
        const result = ptr[0..padded];

        // Zero-initialize padding lanes
        @memset(result[count..padded], 0);
        return result[0..count];
    }

    /// Allocate a SIMD-aligned f64 vector
    pub fn allocF64(self: *Self, count: usize) ?[]align(vector_alignment) f64 {
        const f64_lanes = simd_byte_width / @sizeOf(f64);
        const padded = alignUpVal(count, f64_lanes);
        const byte_size = padded * @sizeOf(f64);

        const bytes = self.arena.alloc(byte_size, vector_alignment) orelse return null;
        self.vector_count += 1;

        const ptr: [*]align(vector_alignment) f64 = @ptrCast(@alignCast(bytes.ptr));
        const result = ptr[0..padded];
        @memset(result[count..padded], 0);
        return result[0..count];
    }

    /// Allocate a raw SIMD-aligned byte buffer
    pub fn allocBytes(self: *Self, size: usize) ?[]align(vector_alignment) u8 {
        return self.arena.alloc(size, vector_alignment);
    }

    pub fn reset(self: *Self) void {
        self.arena.reset();
        self.vector_count = 0;
    }

    pub const Stats = struct {
        used: usize,
        remaining: usize,
        vectors: usize,
    };

    pub fn stats(self: *const Self) Stats {
        return .{
            .used = self.arena.usedBytes(),
            .remaining = self.arena.remainingBytes(),
            .vectors = self.vector_count,
        };
    }
};

// ─── Slab Pool ─────────────────────────────────────────────────────────────

/// Fixed-size object pool using an intrusive free list.
/// O(1) alloc and free. Zero fragmentation.
pub fn SlabPool(comptime T: type) type {
    const slot_size = @max(@sizeOf(T), @sizeOf(usize));
    const slot_align = @max(@alignOf(T), @alignOf(usize));

    return struct {
        const Self = @This();

        const FreeNode = struct {
            next: ?*FreeNode = null,
        };

        backing: std.mem.Allocator,
        buffer: []align(slot_align) u8,
        free_head: ?*FreeNode,
        capacity: usize,
        allocated: usize = 0,

        pub fn init(backing: std.mem.Allocator, count: usize) !Self {
            const total_size = slot_size * count;
            const buf = try backing.alignedAlloc(u8, slot_align, total_size);

            // Build free list (back to front so first alloc returns first slot)
            var head: ?*FreeNode = null;
            var i: usize = count;
            while (i > 0) {
                i -= 1;
                const node: *FreeNode = @ptrCast(@alignCast(buf.ptr + i * slot_size));
                node.next = head;
                head = node;
            }

            return Self{
                .backing = backing,
                .buffer = buf,
                .free_head = head,
                .capacity = count,
            };
        }

        pub fn deinit(self: *Self) void {
            self.backing.free(self.buffer);
            self.* = undefined;
        }

        pub fn create(self: *Self) ?*T {
            const node = self.free_head orelse return null;
            self.free_head = node.next;
            self.allocated += 1;
            const obj: *T = @ptrCast(@alignCast(node));
            obj.* = std.mem.zeroes(T);
            return obj;
        }

        pub fn destroy(self: *Self, obj: *T) void {
            const node: *FreeNode = @ptrCast(@alignCast(obj));
            node.next = self.free_head;
            self.free_head = node;
            self.allocated -= 1;
        }

        pub fn availableSlots(self: *const Self) usize {
            return self.capacity - self.allocated;
        }

        pub fn utilizationPercent(self: *const Self) f64 {
            return @as(f64, @floatFromInt(self.allocated)) /
                @as(f64, @floatFromInt(self.capacity)) * 100.0;
        }
    };
}

// ─── Scratch Allocator ─────────────────────────────────────────────────────

/// Double-buffered arena for ping-pong allocation patterns.
/// Swap front/back each frame or request cycle for zero-stall reuse.
pub const ScratchAllocator = struct {
    arenas: [2]ArenaPool,
    active: u1 = 0,

    pub fn init(backing: std.mem.Allocator, size_per_arena: usize) !ScratchAllocator {
        return ScratchAllocator{
            .arenas = .{
                try ArenaPool.init(backing, .{ .size = size_per_arena }),
                try ArenaPool.init(backing, .{ .size = size_per_arena }),
            },
        };
    }

    pub fn deinit(self: *ScratchAllocator) void {
        self.arenas[0].deinit();
        self.arenas[1].deinit();
    }

    pub fn current(self: *ScratchAllocator) *ArenaPool {
        return &self.arenas[self.active];
    }

    /// Swap active arena and reset the newly active one
    pub fn swap(self: *ScratchAllocator) void {
        self.active ^= 1;
        self.arenas[self.active].reset();
    }
};
