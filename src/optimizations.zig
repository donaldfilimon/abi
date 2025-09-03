//! Performance optimizations for WDBX-AI

const std = @import("std");
const core = @import("core/mod.zig");
const simd = @import("simd/mod.zig");

/// Optimized memory operations
pub const Memory = struct {
    /// Optimized memory copy using SIMD when available
    pub fn fastCopy(dest: []u8, src: []const u8) void {
        std.debug.assert(dest.len >= src.len);
        
        if (src.len < 64) {
            // Small copies - use standard memcpy
            @memcpy(dest[0..src.len], src);
            return;
        }
        
        if (simd.Vector.isSimdAvailable(32)) {
            // Use AVX2 256-bit copies
            simdCopy(u8, 32, dest, src);
        } else if (simd.Vector.isSimdAvailable(16)) {
            // Use SSE 128-bit copies
            simdCopy(u8, 16, dest, src);
        } else {
            // Fallback to optimized scalar copy
            optimizedScalarCopy(dest, src);
        }
    }
    
    fn simdCopy(comptime T: type, comptime vec_size: usize, dest: []T, src: []const T) void {
        const Vector = @Vector(vec_size, T);
        const vec_bytes = vec_size * @sizeOf(T);
        
        var i: usize = 0;
        while (i + vec_bytes <= src.len) : (i += vec_bytes) {
            const vec = @as(*const Vector, @ptrCast(@alignCast(&src[i]))).*;
            @as(*Vector, @ptrCast(@alignCast(&dest[i]))).* = vec;
        }
        
        // Copy remaining bytes
        if (i < src.len) {
            @memcpy(dest[i..i + (src.len - i)], src[i..]);
        }
    }
    
    fn optimizedScalarCopy(dest: []u8, src: []const u8) void {
        // Copy 8 bytes at a time
        var i: usize = 0;
        while (i + 8 <= src.len) : (i += 8) {
            @as(*u64, @ptrCast(@alignCast(&dest[i]))).* = 
                @as(*const u64, @ptrCast(@alignCast(&src[i]))).*;
        }
        
        // Copy remaining bytes
        while (i < src.len) : (i += 1) {
            dest[i] = src[i];
        }
    }
    
    /// Optimized memory comparison
    pub fn fastCompare(a: []const u8, b: []const u8) bool {
        if (a.len != b.len) return false;
        if (a.ptr == b.ptr) return true;
        
        if (a.len < 64) {
            return std.mem.eql(u8, a, b);
        }
        
        if (simd.Vector.isSimdAvailable(32)) {
            return simdCompare(u8, 32, a, b);
        } else if (simd.Vector.isSimdAvailable(16)) {
            return simdCompare(u8, 16, a, b);
        } else {
            return optimizedScalarCompare(a, b);
        }
    }
    
    fn simdCompare(comptime T: type, comptime vec_size: usize, a: []const T, b: []const T) bool {
        const Vector = @Vector(vec_size, T);
        const vec_bytes = vec_size * @sizeOf(T);
        
        var i: usize = 0;
        while (i + vec_bytes <= a.len) : (i += vec_bytes) {
            const vec_a = @as(*const Vector, @ptrCast(@alignCast(&a[i]))).*;
            const vec_b = @as(*const Vector, @ptrCast(@alignCast(&b[i]))).*;
            
            if (!@reduce(.And, vec_a == vec_b)) {
                return false;
            }
        }
        
        // Compare remaining bytes
        return std.mem.eql(T, a[i..], b[i..]);
    }
    
    fn optimizedScalarCompare(a: []const u8, b: []const u8) bool {
        var i: usize = 0;
        while (i + 8 <= a.len) : (i += 8) {
            const a64 = @as(*const u64, @ptrCast(@alignCast(&a[i]))).*;
            const b64 = @as(*const u64, @ptrCast(@alignCast(&b[i]))).*;
            if (a64 != b64) return false;
        }
        
        while (i < a.len) : (i += 1) {
            if (a[i] != b[i]) return false;
        }
        
        return true;
    }
};

/// Optimized hash functions
pub const Hash = struct {
    /// Fast hash using SIMD
    pub fn fastHash(data: []const u8) u64 {
        if (data.len < 16) {
            return fnv1a(data);
        }
        
        if (simd.Vector.isSimdAvailable(16)) {
            return simdHash(data);
        } else {
            return xxhash64(data);
        }
    }
    
    fn fnv1a(data: []const u8) u64 {
        var hash: u64 = 0xcbf29ce484222325;
        for (data) |byte| {
            hash ^= byte;
            hash *%= 0x100000001b3;
        }
        return hash;
    }
    
    fn xxhash64(data: []const u8) u64 {
        const prime1: u64 = 0x9E3779B185EBCA87;
        const prime2: u64 = 0xC2B2AE3D27D4EB4F;
        const prime3: u64 = 0x165667B19E3779F9;
        const prime4: u64 = 0x85EBCA77C2B2AE63;
        const prime5: u64 = 0x27D4EB2F165667C5;
        
        var hash: u64 = undefined;
        
        if (data.len >= 32) {
            var v1 = prime1 +% prime2;
            var v2 = prime2;
            var v3 = 0;
            var v4 = 0 -% prime1;
            
            var i: usize = 0;
            while (i + 32 <= data.len) : (i += 32) {
                v1 = round64(v1, std.mem.readInt(u64, data[i..][0..8], .little));
                v2 = round64(v2, std.mem.readInt(u64, data[i + 8..][0..8], .little));
                v3 = round64(v3, std.mem.readInt(u64, data[i + 16..][0..8], .little));
                v4 = round64(v4, std.mem.readInt(u64, data[i + 24..][0..8], .little));
            }
            
            hash = std.math.rotl(u64, v1, 1) +% 
                   std.math.rotl(u64, v2, 7) +% 
                   std.math.rotl(u64, v3, 12) +% 
                   std.math.rotl(u64, v4, 18);
            
            hash = mergeRound64(hash, v1);
            hash = mergeRound64(hash, v2);
            hash = mergeRound64(hash, v3);
            hash = mergeRound64(hash, v4);
        } else {
            hash = prime5;
        }
        
        hash +%= data.len;
        
        // Process remaining bytes
        var i = data.len & ~@as(usize, 31);
        while (i + 8 <= data.len) : (i += 8) {
            const k1 = round64(0, std.mem.readInt(u64, data[i..][0..8], .little));
            hash ^= k1;
            hash = std.math.rotl(u64, hash, 27) *% prime1 +% prime4;
        }
        
        if (i + 4 <= data.len) {
            hash ^= @as(u64, std.mem.readInt(u32, data[i..][0..4], .little)) *% prime1;
            hash = std.math.rotl(u64, hash, 23) *% prime2 +% prime3;
            i += 4;
        }
        
        while (i < data.len) : (i += 1) {
            hash ^= @as(u64, data[i]) *% prime5;
            hash = std.math.rotl(u64, hash, 11) *% prime1;
        }
        
        // Final mixing
        hash ^= hash >> 33;
        hash *%= prime2;
        hash ^= hash >> 29;
        hash *%= prime3;
        hash ^= hash >> 32;
        
        return hash;
    }
    
    fn round64(acc: u64, input: u64) u64 {
        const prime1: u64 = 0x9E3779B185EBCA87;
        const prime2: u64 = 0xC2B2AE3D27D4EB4F;
        var acc_mut = acc;
        acc_mut +%= input *% prime2;
        acc_mut = std.math.rotl(u64, acc_mut, 31);
        acc_mut *%= prime1;
        return acc_mut;
    }
    
    fn mergeRound64(acc: u64, val: u64) u64 {
        const prime1: u64 = 0x9E3779B185EBCA87;
        const prime4: u64 = 0x85EBCA77C2B2AE63;
        const roundedVal = round64(0, val);
        var acc_mut = acc;
        acc_mut ^= roundedVal;
        acc_mut *%= prime1;
        acc_mut +%= prime4;
        return acc_mut;
    }
    
    fn simdHash(data: []const u8) u64 {
        const Vector16 = @Vector(16, u8);
        var hash_vec = @splat(16, @as(u8, 0));
        
        var i: usize = 0;
        while (i + 16 <= data.len) : (i += 16) {
            const data_vec = @as(*const Vector16, @ptrCast(@alignCast(&data[i]))).*;
            hash_vec ^= data_vec;
            
            // Mix using multiplication
            const hash_u64 = @as([2]u64, @bitCast(hash_vec));
            hash_vec = @bitCast([2]u64{ 
                hash_u64[0] *% 0x87c37b91114253d5,
                hash_u64[1] *% 0x4cf5ad432745937f 
            });
        }
        
        // Final reduction
        const final_hash = @as([2]u64, @bitCast(hash_vec));
        var result = final_hash[0] ^ final_hash[1];
        
        // Hash remaining bytes
        while (i < data.len) : (i += 1) {
            result ^= @as(u64, data[i]) *% 0x27d4eb2f165667c5;
            result = std.math.rotl(u64, result, 33) *% 0x165667b19e3779f9;
        }
        
        return result;
    }
};

/// Optimized sorting algorithms
pub const Sort = struct {
    /// Radix sort for integers
    pub fn radixSort(comptime T: type, items: []T) void {
        if (items.len <= 1) return;
        
        const bits = @bitSizeOf(T);
        const radix_bits = 8;
        const buckets_count = 1 << radix_bits;
        const passes = (bits + radix_bits - 1) / radix_bits;
        
        var allocator = std.heap.page_allocator;
        const temp = allocator.alloc(T, items.len) catch {
            // Fallback to standard sort
            std.mem.sort(T, items, {}, std.sort.asc(T));
            return;
        };
        defer allocator.free(temp);
        
        var src = items;
        var dst = temp;
        
        var pass: usize = 0;
        while (pass < passes) : (pass += 1) {
            var counts = [_]usize{0} ** buckets_count;
            
            // Count occurrences
            for (src) |value| {
                const bucket = (value >> @as(std.math.Log2Int(T), @intCast(pass * radix_bits))) & (buckets_count - 1);
                counts[@as(usize, @intCast(bucket))] += 1;
            }
            
            // Compute offsets
            var offset: usize = 0;
            for (&counts) |*count| {
                const tmp = count.*;
                count.* = offset;
                offset += tmp;
            }
            
            // Place elements
            for (src) |value| {
                const bucket = (value >> @as(std.math.Log2Int(T), @intCast(pass * radix_bits))) & (buckets_count - 1);
                const idx = counts[@as(usize, @intCast(bucket))];
                dst[idx] = value;
                counts[@as(usize, @intCast(bucket))] += 1;
            }
            
            // Swap buffers
            const tmp = src;
            src = dst;
            dst = tmp;
        }
        
        // Copy result if needed
        if (src.ptr != items.ptr) {
            @memcpy(items, src);
        }
    }
    
    /// Parallel quicksort
    pub fn parallelSort(
        comptime T: type,
        items: []T,
        context: anytype,
        comptime lessThan: fn (context: @TypeOf(context), lhs: T, rhs: T) bool,
        pool: *core.threading.ThreadPool,
    ) void {
        if (items.len <= 1000) {
            // Use standard sort for small arrays
            std.mem.sort(T, items, context, lessThan);
            return;
        }
        
        parallelQuicksort(T, items, context, lessThan, pool, 0);
    }
    
    fn parallelQuicksort(
        comptime T: type,
        items: []T,
        context: anytype,
        comptime lessThan: fn (context: @TypeOf(context), lhs: T, rhs: T) bool,
        pool: *core.threading.ThreadPool,
        depth: usize,
    ) void {
        if (items.len <= 1) return;
        
        if (items.len <= 100 or depth > 10) {
            // Use insertion sort for small arrays or deep recursion
            insertionSort(T, items, context, lessThan);
            return;
        }
        
        // Choose pivot (median of three)
        const pivot = medianOfThree(T, items, context, lessThan);
        
        // Partition
        var i: usize = 0;
        var j = items.len - 1;
        while (i < j) {
            while (i < j and lessThan(context, items[i], pivot)) : (i += 1) {}
            while (i < j and !lessThan(context, items[j], pivot)) : (j -= 1) {}
            if (i < j) {
                std.mem.swap(T, &items[i], &items[j]);
            }
        }
        
        // Parallel recursive calls
        const left = items[0..i];
        const right = items[i..];
        
        if (left.len > 1000 and right.len > 1000) {
            // Both partitions are large, sort in parallel
            var left_done = std.Thread.ResetEvent{};
            
            pool.submit(struct {
                fn sort(args: *struct {
                    slice: []T,
                    ctx: @TypeOf(context),
                    pool_ref: *core.threading.ThreadPool,
                    depth_val: usize,
                    event: *std.Thread.ResetEvent,
                }) void {
                    parallelQuicksort(T, args.slice, args.ctx, lessThan, args.pool_ref, args.depth_val + 1);
                    args.event.set();
                }
            }.sort, &.{
                .slice = left,
                .ctx = context,
                .pool_ref = pool,
                .depth_val = depth,
                .event = &left_done,
            }) catch {
                // Fallback to sequential
                parallelQuicksort(T, left, context, lessThan, pool, depth + 1);
            };
            
            parallelQuicksort(T, right, context, lessThan, pool, depth + 1);
            left_done.wait();
        } else {
            // Sort sequentially
            parallelQuicksort(T, left, context, lessThan, pool, depth + 1);
            parallelQuicksort(T, right, context, lessThan, pool, depth + 1);
        }
    }
    
    fn medianOfThree(
        comptime T: type,
        items: []T,
        context: anytype,
        comptime lessThan: fn (context: @TypeOf(context), lhs: T, rhs: T) bool,
    ) T {
        const a = items[0];
        const b = items[items.len / 2];
        const c = items[items.len - 1];
        
        if (lessThan(context, a, b)) {
            if (lessThan(context, b, c)) {
                return b;
            } else if (lessThan(context, a, c)) {
                return c;
            } else {
                return a;
            }
        } else {
            if (lessThan(context, a, c)) {
                return a;
            } else if (lessThan(context, b, c)) {
                return c;
            } else {
                return b;
            }
        }
    }
    
    fn insertionSort(
        comptime T: type,
        items: []T,
        context: anytype,
        comptime lessThan: fn (context: @TypeOf(context), lhs: T, rhs: T) bool,
    ) void {
        var i: usize = 1;
        while (i < items.len) : (i += 1) {
            const key = items[i];
            var j = i;
            while (j > 0 and lessThan(context, key, items[j - 1])) : (j -= 1) {
                items[j] = items[j - 1];
            }
            items[j] = key;
        }
    }
};

/// Cache-friendly data structures
pub const Cache = struct {
    /// Cache line size (typical)
    pub const line_size = 64;
    
    /// Align data to cache line
    pub fn alignToCacheLine(comptime T: type) type {
        return struct {
            data: T align(line_size),
        };
    }
    
    /// Prefetch data
    pub inline fn prefetch(ptr: anytype, comptime rw: std.builtin.PrefetchRw, comptime locality: u2) void {
        @prefetch(ptr, .{
            .rw = rw,
            .locality = locality,
            .cache = .data,
        });
    }
    
    /// Cache-oblivious matrix transpose
    pub fn transposeMatrix(comptime T: type, dst: []T, src: []const T, rows: usize, cols: usize) void {
        std.debug.assert(dst.len >= rows * cols);
        std.debug.assert(src.len >= rows * cols);
        
        if (rows <= 32 and cols <= 32) {
            // Base case: small matrix
            var i: usize = 0;
            while (i < rows) : (i += 1) {
                var j: usize = 0;
                while (j < cols) : (j += 1) {
                    dst[j * rows + i] = src[i * cols + j];
                }
            }
        } else if (rows >= cols) {
            // Split rows
            const mid = rows / 2;
            transposeMatrix(T, dst, src, mid, cols);
            transposeMatrix(T, dst[mid..], src[mid * cols..], rows - mid, cols);
        } else {
            // Split columns
            const mid = cols / 2;
            transposeMatrix(T, dst, src, rows, mid);
            transposeMatrix(T, dst[mid * rows..], src[mid..], rows, cols - mid);
        }
    }
};

test "optimized memory operations" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    // Test fast copy
    const src = try allocator.alloc(u8, 1024);
    defer allocator.free(src);
    const dest = try allocator.alloc(u8, 1024);
    defer allocator.free(dest);
    
    for (src, 0..) |*byte, i| {
        byte.* = @as(u8, @intCast(i & 0xFF));
    }
    
    Memory.fastCopy(dest, src);
    try testing.expect(Memory.fastCompare(dest, src));
}

test "optimized hash functions" {
    const testing = std.testing;
    
    const data1 = "Hello, World!";
    const data2 = "Hello, World!";
    const data3 = "Different data";
    
    const hash1 = Hash.fastHash(data1);
    const hash2 = Hash.fastHash(data2);
    const hash3 = Hash.fastHash(data3);
    
    try testing.expectEqual(hash1, hash2);
    try testing.expect(hash1 != hash3);
}

test "radix sort" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    var data = try allocator.alloc(u32, 1000);
    defer allocator.free(data);
    
    // Fill with random data
    var prng = std.rand.DefaultPrng.init(42);
    const random = prng.random();
    for (data) |*item| {
        item.* = random.int(u32);
    }
    
    Sort.radixSort(u32, data);
    
    // Verify sorted
    var i: usize = 1;
    while (i < data.len) : (i += 1) {
        try testing.expect(data[i - 1] <= data[i]);
    }
}