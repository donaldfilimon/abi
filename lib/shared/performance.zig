//! Performance optimization utilities and compile-time configuration
//!
//! This module provides utilities for optimizing code paths at compile time
//! and runtime, including inlining hints, SIMD detection, and performance
//! profiling helpers.

const std = @import("std");
const builtin = @import("builtin");

/// Force inline for performance-critical functions
pub inline fn forceInline(comptime func: anytype) @TypeOf(func) {
    return @call(.always_inline, func, .{});
}

/// Optimization hint: likely branch
pub inline fn likely(condition: bool) bool {
    if (builtin.mode == .Debug) return condition;
    return @call(.{ .modifier = .always_inline }, @import("std").debug.assert, .{condition});
}

/// Optimization hint: unlikely branch  
pub inline fn unlikely(condition: bool) bool {
    if (builtin.mode == .Debug) return condition;
    return !likely(!condition);
}

/// Check if SIMD is available for the target
pub fn simdAvailable() bool {
    return switch (builtin.cpu.arch) {
        .x86_64, .x86, .aarch64, .arm => true,
        else => false,
    };
}

/// Get optimal SIMD width for f32 operations
pub fn simdWidth() comptime_int {
    return switch (builtin.cpu.arch) {
        .x86_64, .x86 => 4, // SSE
        .aarch64 => 4, // NEON
        .arm => 2,
        else => 1, // scalar fallback
    };
}

/// Memory prefetch hint for better cache performance
pub inline fn prefetch(ptr: anytype) void {
    if (builtin.mode == .Debug) return;
    @prefetch(ptr, .{ .rw = .read, .locality = 3, .cache = .data });
}

/// Cache line size for optimal alignment
pub const cache_line_size: usize = switch (builtin.cpu.arch) {
    .x86_64, .x86 => 64,
    .aarch64, .arm => 64,
    else => 64, // reasonable default
};

/// Align data to cache line boundaries
pub fn alignToCacheLine(comptime T: type) type {
    return struct {
        data: T align(cache_line_size),
    };
}

/// Fast integer power for compile-time calculations
pub fn comptime_pow(base: comptime_int, exp: comptime_int) comptime_int {
    var result: comptime_int = 1;
    var i: comptime_int = 0;
    while (i < exp) : (i += 1) {
        result *= base;
    }
    return result;
}

/// Allocation size hints for better memory layout
pub const AllocHint = enum {
    small, // < 256 bytes
    medium, // 256 bytes - 4KB
    large, // > 4KB
    
    pub fn fromSize(size: usize) AllocHint {
        if (size < 256) return .small;
        if (size < 4096) return .medium;
        return .large;
    }
};

/// Fast modulo for power of 2
pub inline fn fastMod(value: anytype, comptime divisor: comptime_int) @TypeOf(value) {
    comptime {
        if (!std.math.isPowerOfTwo(divisor)) {
            @compileError("fastMod requires power of 2 divisor");
        }
    }
    return value & (divisor - 1);
}

/// Branchless min/max for optimization
pub inline fn branchlessMin(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    const diff = @as(i64, @intCast(a)) - @as(i64, @intCast(b));
    const mask = diff >> 63; // arithmetic shift creates mask
    return @intCast(a - ((diff & mask)));
}

pub inline fn branchlessMax(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    return branchlessMin(b, a);
}

/// Performance profiling timer
pub const Timer = struct {
    start_time: i64,
    
    pub fn start() Timer {
        return .{ .start_time = std.time.nanoTimestamp() };
    }
    
    pub fn elapsed(self: Timer) i64 {
        return std.time.nanoTimestamp() - self.start_time;
    }
    
    pub fn elapsedMicros(self: Timer) i64 {
        return @divFloor(self.elapsed(), 1000);
    }
    
    pub fn elapsedMillis(self: Timer) i64 {
        return @divFloor(self.elapsed(), 1_000_000);
    }
    
    pub fn reset(self: *Timer) void {
        self.start_time = std.time.nanoTimestamp();
    }
};

/// Memory-efficient buffer pooling
pub fn BufferPool(comptime size: usize, comptime count: usize) type {
    return struct {
        const Self = @This();
        const Buffer = [size]u8;
        
        buffers: [count]Buffer align(cache_line_size),
        available: std.bit_set.IntegerBitSet(count),
        mutex: std.Thread.Mutex,
        
        pub fn init() Self {
            return .{
                .buffers = undefined,
                .available = std.bit_set.IntegerBitSet(count).initFull(),
                .mutex = .{},
            };
        }
        
        pub fn acquire(self: *Self) ?[]u8 {
            self.mutex.lock();
            defer self.mutex.unlock();
            
            const idx = self.available.findFirstSet() orelse return null;
            self.available.unset(idx);
            return &self.buffers[idx];
        }
        
        pub fn release(self: *Self, buffer: []u8) void {
            self.mutex.lock();
            defer self.mutex.unlock();
            
            const buf_ptr = @intFromPtr(buffer.ptr);
            const base_ptr = @intFromPtr(&self.buffers[0]);
            const idx = @divExact(buf_ptr - base_ptr, size);
            
            std.debug.assert(idx < count);
            self.available.set(idx);
        }
    };
}

/// Compile-time feature detection
pub const features = struct {
    pub const has_simd = simdAvailable();
    pub const optimal_simd_width = simdWidth();
    pub const is_debug = (builtin.mode == .Debug);
    pub const is_release = !is_debug;
    pub const target_os = builtin.os.tag;
    pub const target_arch = builtin.cpu.arch;
};

test "performance utilities" {
    // Test timer
    var timer = Timer.start();
    std.time.sleep(1_000_000); // 1ms
    const elapsed = timer.elapsedMicros();
    try std.testing.expect(elapsed >= 900); // Allow some variance
    
    // Test fast mod
    try std.testing.expectEqual(@as(u32, 7), fastMod(@as(u32, 23), 16));
    
    // Test branchless operations
    try std.testing.expectEqual(@as(i32, 5), branchlessMin(@as(i32, 5), @as(i32, 10)));
    try std.testing.expectEqual(@as(i32, 10), branchlessMax(@as(i32, 5), @as(i32, 10)));
}

test "buffer pool" {
    var pool = BufferPool(1024, 4).init();
    
    // Acquire buffers
    const buf1 = pool.acquire();
    try std.testing.expect(buf1 != null);
    
    const buf2 = pool.acquire();
    try std.testing.expect(buf2 != null);
    
    // Release and reacquire
    pool.release(buf1.?);
    const buf3 = pool.acquire();
    try std.testing.expect(buf3 != null);
    
    pool.release(buf2.?);
    pool.release(buf3.?);
}
