//! Zig 0.16 Performance Optimizations
//!
//! This module contains performance optimizations specifically designed for
//! Zig 0.16, leveraging the latest compiler features and standard library improvements.

const std = @import("std");
const builtin = @import("builtin");
const imports = @import("../imports.zig");

// =============================================================================
// SIMD OPTIMIZATIONS
// =============================================================================

/// SIMD operations optimized for Zig 0.16
pub const SIMD = struct {
    /// Vector length for optimal SIMD operations on current target
    pub const optimal_vector_len = switch (builtin.cpu.arch) {
        .x86_64 => if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f)) 16 else if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx2)) 8 else 4,
        .aarch64 => 4, // NEON 128-bit vectors
        .wasm32, .wasm64 => 4, // WASM SIMD 128-bit vectors
        else => 4, // Conservative default
    };
    
    /// Optimized vector addition using Zig 0.16 SIMD
    pub fn vectorAdd(comptime T: type, a: []const T, b: []const T, result: []T) void {
        std.debug.assert(a.len == b.len and b.len == result.len);
        
        if (comptime shouldUseSIMD(T)) {
            vectorAddSIMD(T, a, b, result);
        } else {
            vectorAddScalar(T, a, b, result);
        }
    }
    
    /// Optimized dot product using Zig 0.16 SIMD
    pub fn dotProduct(comptime T: type, a: []const T, b: []const T) T {
        std.debug.assert(a.len == b.len);
        
        if (comptime shouldUseSIMD(T)) {
            return dotProductSIMD(T, a, b);
        } else {
            return dotProductScalar(T, a, b);
        }
    }
    
    /// Check if SIMD should be used for given type
    fn shouldUseSIMD(comptime T: type) bool {
        return switch (T) {
            f32, f64 => true,
            i8, i16, i32, i64 => true,
            u8, u16, u32, u64 => true,
            else => false,
        };
    }
    
    /// SIMD vector addition implementation
    fn vectorAddSIMD(comptime T: type, a: []const T, b: []const T, result: []T) void {
        const VectorType = @Vector(optimal_vector_len, T);
        const len = a.len;
        const simd_len = len - (len % optimal_vector_len);
        
        // Process SIMD chunks
        var i: usize = 0;
        while (i < simd_len) : (i += optimal_vector_len) {
            const va: VectorType = a[i..i + optimal_vector_len][0..optimal_vector_len].*;
            const vb: VectorType = b[i..i + optimal_vector_len][0..optimal_vector_len].*;
            const vr = va + vb;
            result[i..i + optimal_vector_len][0..optimal_vector_len].* = vr;
        }
        
        // Process remaining elements
        while (i < len) : (i += 1) {
            result[i] = a[i] + b[i];
        }
    }
    
    /// Scalar vector addition fallback
    fn vectorAddScalar(comptime T: type, a: []const T, b: []const T, result: []T) void {
        for (a, b, result) |av, bv, *rv| {
            rv.* = av + bv;
        }
    }
    
    /// SIMD dot product implementation
    fn dotProductSIMD(comptime T: type, a: []const T, b: []const T) T {
        const VectorType = @Vector(optimal_vector_len, T);
        const len = a.len;
        const simd_len = len - (len % optimal_vector_len);
        
        var sum_vector: VectorType = @splat(0);
        
        // Process SIMD chunks
        var i: usize = 0;
        while (i < simd_len) : (i += optimal_vector_len) {
            const va: VectorType = a[i..i + optimal_vector_len][0..optimal_vector_len].*;
            const vb: VectorType = b[i..i + optimal_vector_len][0..optimal_vector_len].*;
            sum_vector += va * vb;
        }
        
        // Reduce vector to scalar
        var result: T = 0;
        for (0..optimal_vector_len) |j| {
            result += sum_vector[j];
        }
        
        // Process remaining elements
        while (i < len) : (i += 1) {
            result += a[i] * b[i];
        }
        
        return result;
    }
    
    /// Scalar dot product fallback
    fn dotProductScalar(comptime T: type, a: []const T, b: []const T) T {
        var result: T = 0;
        for (a, b) |av, bv| {
            result += av * bv;
        }
        return result;
    }
};

// =============================================================================
// MEMORY OPTIMIZATIONS
// =============================================================================

/// Memory operations optimized for Zig 0.16
pub const Memory = struct {
    /// Optimized memory copy using Zig 0.16 features
    pub fn optimizedCopy(comptime T: type, dest: []T, src: []const T) void {
        std.debug.assert(dest.len >= src.len);
        
        if (comptime shouldUseMemcpy(T)) {
            @memcpy(dest[0..src.len], src);
        } else {
            // Custom copy for complex types
            for (src, dest[0..src.len]) |s, *d| {
                d.* = s;
            }
        }
    }
    
    /// Optimized memory set using Zig 0.16 features
    pub fn optimizedSet(comptime T: type, dest: []T, value: T) void {
        if (comptime shouldUseMemset(T)) {
            @memset(dest, value);
        } else {
            // Custom set for complex types
            for (dest) |*d| {
                d.* = value;
            }
        }
    }
    
    /// Check if memcpy should be used
    fn shouldUseMemcpy(comptime T: type) bool {
        return switch (@typeInfo(T)) {
            .Int, .Float => true,
            .Struct => |info| !info.layout != .Auto, // Use for packed/extern structs
            else => false,
        };
    }
    
    /// Check if memset should be used
    fn shouldUseMemset(comptime T: type) bool {
        return switch (@typeInfo(T)) {
            .Int => @sizeOf(T) == 1, // Only for u8/i8
            else => false,
        };
    }
    
    /// Cache-friendly memory access patterns
    pub const CacheOptimized = struct {
        /// Process data in cache-friendly chunks
        pub fn processInChunks(
            comptime T: type,
            data: []T,
            chunk_size: usize,
            process_fn: *const fn ([]T) void,
        ) void {
            var i: usize = 0;
            while (i < data.len) {
                const end = @min(i + chunk_size, data.len);
                process_fn(data[i..end]);
                i = end;
            }
        }
        
        /// Prefetch memory for better cache performance
        pub fn prefetch(ptr: anytype, comptime locality: u2) void {
            if (comptime builtin.cpu.arch == .x86_64) {
                // Use x86 prefetch instructions
                switch (locality) {
                    0 => asm volatile ("prefetchnta %[ptr]" : : [ptr] "*m" (ptr.*)),
                    1 => asm volatile ("prefetcht2 %[ptr]" : : [ptr] "*m" (ptr.*)),
                    2 => asm volatile ("prefetcht1 %[ptr]" : : [ptr] "*m" (ptr.*)),
                    3 => asm volatile ("prefetcht0 %[ptr]" : : [ptr] "*m" (ptr.*)),
                }
            }
            // Other architectures would have their own prefetch instructions
        }
    };
};

// =============================================================================
// ALLOCATION OPTIMIZATIONS
// =============================================================================

/// Allocation strategies optimized for Zig 0.16
pub const Allocation = struct {
    /// Pool allocator for frequent same-size allocations
    pub fn PoolAllocator(comptime T: type) type {
        return struct {
            const Self = @This();
            const Node = struct {
                next: ?*Node,
                data: T,
            };
            
            backing_allocator: imports.Allocator,
            free_list: ?*Node,
            allocated_nodes: imports.ArrayList(*Node),
            
            pub fn init(backing_allocator: imports.Allocator) Self {
                return Self{
                    .backing_allocator = backing_allocator,
                    .free_list = null,
                    .allocated_nodes = imports.ArrayList(*Node).init(backing_allocator),
                };
            }
            
            pub fn deinit(self: *Self) void {
                for (self.allocated_nodes.items) |node| {
                    self.backing_allocator.destroy(node);
                }
                self.allocated_nodes.deinit();
            }
            
            pub fn create(self: *Self) !*T {
                const node = if (self.free_list) |free_node| blk: {
                    self.free_list = free_node.next;
                    break :blk free_node;
                } else blk: {
                    const new_node = try self.backing_allocator.create(Node);
                    try self.allocated_nodes.append(new_node);
                    break :blk new_node;
                };
                
                return &node.data;
            }
            
            pub fn destroy(self: *Self, ptr: *T) void {
                const node = @fieldParentPtr(Node, "data", ptr);
                node.next = self.free_list;
                self.free_list = node;
            }
        };
    }
    
    /// Stack allocator for temporary allocations
    pub const StackAllocator = struct {
        buffer: []u8,
        offset: usize,
        
        pub fn init(buffer: []u8) StackAllocator {
            return StackAllocator{
                .buffer = buffer,
                .offset = 0,
            };
        }
        
        pub fn allocator(self: *StackAllocator) imports.Allocator {
            return imports.Allocator{
                .ptr = self,
                .vtable = &.{
                    .alloc = alloc,
                    .resize = resize,
                    .free = free,
                },
            };
        }
        
        fn alloc(ctx: *anyopaque, len: usize, log2_ptr_align: u8, ret_addr: usize) ?[*]u8 {
            _ = ret_addr;
            const self: *StackAllocator = @ptrCast(@alignCast(ctx));
            
            const ptr_align = @as(usize, 1) << @intCast(log2_ptr_align);
            const aligned_offset = std.mem.alignForward(usize, self.offset, ptr_align);
            
            if (aligned_offset + len > self.buffer.len) {
                return null; // Out of memory
            }
            
            const result = self.buffer[aligned_offset..aligned_offset + len];
            self.offset = aligned_offset + len;
            
            return result.ptr;
        }
        
        fn resize(ctx: *anyopaque, buf: []u8, log2_buf_align: u8, new_len: usize, ret_addr: usize) bool {
            _ = ctx;
            _ = buf;
            _ = log2_buf_align;
            _ = new_len;
            _ = ret_addr;
            return false; // Stack allocator doesn't support resize
        }
        
        fn free(ctx: *anyopaque, buf: []u8, log2_buf_align: u8, ret_addr: usize) void {
            _ = ctx;
            _ = buf;
            _ = log2_buf_align;
            _ = ret_addr;
            // Stack allocator doesn't support individual free
        }
        
        pub fn reset(self: *StackAllocator) void {
            self.offset = 0;
        }
    };
};

// =============================================================================
// CONCURRENCY OPTIMIZATIONS
// =============================================================================

/// Concurrency utilities optimized for Zig 0.16
pub const Concurrency = struct {
    /// Lock-free queue implementation
    pub fn LockFreeQueue(comptime T: type) type {
        return struct {
            const Self = @This();
            const Node = struct {
                data: T,
                next: std.atomic.Value(?*Node),
            };
            
            head: std.atomic.Value(?*Node),
            tail: std.atomic.Value(?*Node),
            allocator: imports.Allocator,
            
            pub fn init(allocator: imports.Allocator) Self {
                const dummy = allocator.create(Node) catch unreachable;
                dummy.* = Node{
                    .data = undefined,
                    .next = std.atomic.Value(?*Node).init(null),
                };
                
                return Self{
                    .head = std.atomic.Value(?*Node).init(dummy),
                    .tail = std.atomic.Value(?*Node).init(dummy),
                    .allocator = allocator,
                };
            }
            
            pub fn deinit(self: *Self) void {
                while (self.dequeue()) |_| {}
                
                // Clean up dummy node
                if (self.head.load(.acquire)) |head| {
                    self.allocator.destroy(head);
                }
            }
            
            pub fn enqueue(self: *Self, data: T) !void {
                const new_node = try self.allocator.create(Node);
                new_node.* = Node{
                    .data = data,
                    .next = std.atomic.Value(?*Node).init(null),
                };
                
                while (true) {
                    const tail = self.tail.load(.acquire) orelse continue;
                    const next = tail.next.load(.acquire);
                    
                    if (next == null) {
                        if (tail.next.cmpxchgWeak(null, new_node, .release, .acquire) == null) {
                            _ = self.tail.cmpxchgWeak(tail, new_node, .release, .acquire);
                            break;
                        }
                    } else {
                        _ = self.tail.cmpxchgWeak(tail, next, .release, .acquire);
                    }
                }
            }
            
            pub fn dequeue(self: *Self) ?T {
                while (true) {
                    const head = self.head.load(.acquire) orelse continue;
                    const tail = self.tail.load(.acquire) orelse continue;
                    const next = head.next.load(.acquire);
                    
                    if (head == tail) {
                        if (next == null) {
                            return null; // Queue is empty
                        }
                        _ = self.tail.cmpxchgWeak(tail, next, .release, .acquire);
                    } else if (next) |next_node| {
                        if (self.head.cmpxchgWeak(head, next_node, .release, .acquire) == head) {
                            const data = next_node.data;
                            self.allocator.destroy(head);
                            return data;
                        }
                    }
                }
            }
        };
    }
    
    /// Work-stealing deque for parallel processing
    pub fn WorkStealingDeque(comptime T: type) type {
        return struct {
            const Self = @This();
            
            buffer: []std.atomic.Value(T),
            top: std.atomic.Value(u64),
            bottom: std.atomic.Value(u64),
            allocator: imports.Allocator,
            
            pub fn init(allocator: imports.Allocator, capacity: usize) !Self {
                const buffer = try allocator.alloc(std.atomic.Value(T), capacity);
                for (buffer) |*item| {
                    item.* = std.atomic.Value(T).init(undefined);
                }
                
                return Self{
                    .buffer = buffer,
                    .top = std.atomic.Value(u64).init(0),
                    .bottom = std.atomic.Value(u64).init(0),
                    .allocator = allocator,
                };
            }
            
            pub fn deinit(self: *Self) void {
                self.allocator.free(self.buffer);
            }
            
            pub fn pushBottom(self: *Self, item: T) void {
                const b = self.bottom.load(.acquire);
                self.buffer[b % self.buffer.len].store(item, .release);
                self.bottom.store(b + 1, .release);
            }
            
            pub fn popBottom(self: *Self) ?T {
                const b = self.bottom.load(.acquire) -% 1;
                self.bottom.store(b, .release);
                
                const t = self.top.load(.acquire);
                if (t <= b) {
                    const item = self.buffer[b % self.buffer.len].load(.acquire);
                    if (t == b) {
                        if (self.top.cmpxchgWeak(t, t + 1, .acq_rel, .acquire) != t) {
                            self.bottom.store(b + 1, .release);
                            return null;
                        }
                    }
                    return item;
                } else {
                    self.bottom.store(b + 1, .release);
                    return null;
                }
            }
            
            pub fn steal(self: *Self) ?T {
                const t = self.top.load(.acquire);
                const b = self.bottom.load(.acquire);
                
                if (t < b) {
                    const item = self.buffer[t % self.buffer.len].load(.acquire);
                    if (self.top.cmpxchgWeak(t, t + 1, .acq_rel, .acquire) == t) {
                        return item;
                    }
                }
                return null;
            }
        };
    }
};

// =============================================================================
// TESTS
// =============================================================================

test "SIMD vector operations" {
    const testing = imports.testing;
    const allocator = testing.allocator;
    
    const len = 1000;
    const a = try allocator.alloc(f32, len);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, len);
    defer allocator.free(b);
    const result = try allocator.alloc(f32, len);
    defer allocator.free(result);
    
    // Initialize test data
    for (a, b, 0..) |*av, *bv, i| {
        av.* = @floatFromInt(i);
        bv.* = @floatFromInt(i * 2);
    }
    
    // Test vector addition
    SIMD.vectorAdd(f32, a, b, result);
    
    // Verify results
    for (a, b, result, 0..) |av, bv, rv, i| {
        const expected = av + bv;
        try testing.expectApproxEqAbs(expected, rv, 0.001);
        _ = i;
    }
    
    // Test dot product
    const dot = SIMD.dotProduct(f32, a, b);
    var expected_dot: f32 = 0;
    for (a, b) |av, bv| {
        expected_dot += av * bv;
    }
    try testing.expectApproxEqAbs(expected_dot, dot, 0.001);
}

test "memory optimizations" {
    const testing = imports.testing;
    const allocator = testing.allocator;
    
    // Test optimized copy
    const src = [_]u32{ 1, 2, 3, 4, 5 };
    var dest = [_]u32{ 0, 0, 0, 0, 0 };
    
    Memory.optimizedCopy(u32, &dest, &src);
    try testing.expectEqualSlices(u32, &src, &dest);
    
    // Test optimized set
    var buffer = [_]u8{ 0, 0, 0, 0, 0 };
    Memory.optimizedSet(u8, &buffer, 42);
    
    for (buffer) |b| {
        try testing.expectEqual(@as(u8, 42), b);
    }
}

test "pool allocator" {
    const testing = imports.testing;
    
    var pool = Allocation.PoolAllocator(u32).init(testing.allocator);
    defer pool.deinit();
    
    // Allocate some items
    const item1 = try pool.create();
    const item2 = try pool.create();
    const item3 = try pool.create();
    
    item1.* = 1;
    item2.* = 2;
    item3.* = 3;
    
    try testing.expectEqual(@as(u32, 1), item1.*);
    try testing.expectEqual(@as(u32, 2), item2.*);
    try testing.expectEqual(@as(u32, 3), item3.*);
    
    // Free and reuse
    pool.destroy(item2);
    const item4 = try pool.create();
    item4.* = 4;
    try testing.expectEqual(@as(u32, 4), item4.*);
}

test "lock-free queue" {
    const testing = imports.testing;
    
    var queue = Concurrency.LockFreeQueue(u32).init(testing.allocator);
    defer queue.deinit();
    
    // Test basic operations
    try queue.enqueue(1);
    try queue.enqueue(2);
    try queue.enqueue(3);
    
    try testing.expectEqual(@as(?u32, 1), queue.dequeue());
    try testing.expectEqual(@as(?u32, 2), queue.dequeue());
    try testing.expectEqual(@as(?u32, 3), queue.dequeue());
    try testing.expectEqual(@as(?u32, null), queue.dequeue());
}