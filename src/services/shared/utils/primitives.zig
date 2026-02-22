// ============================================================================
// ABI Framework — Utility Primitives
// ============================================================================
//
// Adapted from abi-system-v2.0 src/utils.zig for the ABI framework.
//
// Zero-allocation fast paths, lock-free concurrency, compile-time platform
// detection. Every function here is either comptime-evaluable or sub-us.
// ============================================================================

const std = @import("std");
const builtin = @import("builtin");
const time = @import("../time.zig");

// --- Mathematical Utilities ------------------------------------------------

pub const Math = struct {
    pub inline fn isPowerOfTwo(x: anytype) bool {
        return x > 0 and (x & (x - 1)) == 0;
    }

    pub fn nextPowerOfTwo(comptime T: type, x: T) T {
        if (x == 0) return 1;
        if (isPowerOfTwo(x)) return x;
        var n = x - 1;
        var shift: usize = 1;
        while (shift < @bitSizeOf(T)) : (shift <<= 1) {
            n |= n >> @as(std.math.Log2Int(T), @intCast(shift));
        }
        return n +| 1; // saturating add prevents overflow on max values
    }

    /// Align `value` up to `alignment`. Alignment must be a power of two.
    /// Uses runtime assert (not comptime) since alignment may be a runtime value.
    pub fn alignUp(comptime T: type, value: T, alignment: T) T {
        std.debug.assert(isPowerOfTwo(alignment));
        const mask = alignment - 1;
        return (value + mask) & ~mask;
    }

    /// Comptime-only alignment when alignment is known at compile time.
    pub inline fn alignUpComptime(comptime T: type, value: T, comptime alignment: T) T {
        comptime std.debug.assert(isPowerOfTwo(alignment));
        const mask = alignment - 1;
        return (value + mask) & ~mask;
    }

    pub fn clamp(comptime T: type, value: T, min_val: T, max_val: T) T {
        return @max(min_val, @min(max_val, value));
    }

    pub fn divCeil(comptime T: type, numerator: T, denominator: T) T {
        return (numerator + denominator - 1) / denominator;
    }

    /// Fast log2 for power-of-two values
    pub fn log2(comptime T: type, x: T) std.math.Log2Int(T) {
        std.debug.assert(isPowerOfTwo(x));
        return @intCast(@ctz(x));
    }
};

// --- String Utilities ------------------------------------------------------

pub const String = struct {
    pub fn eqlIgnoreCase(a: []const u8, b: []const u8) bool {
        if (a.len != b.len) return false;
        for (a, b) |ca, cb| {
            if (std.ascii.toLower(ca) != std.ascii.toLower(cb)) return false;
        }
        return true;
    }

    /// Zero-copy string formatting into a caller-owned buffer
    pub fn formatBuf(buf: []u8, comptime fmt: []const u8, args: anytype) []const u8 {
        return std.fmt.bufPrint(buf, fmt, args) catch return buf[0..0];
    }

    pub const Builder = struct {
        buffer: std.ArrayListUnmanaged(u8) = .{},
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) Builder {
            return .{ .allocator = allocator };
        }

        pub fn deinit(self: *Builder) void {
            self.buffer.deinit(self.allocator);
        }

        pub fn append(self: *Builder, str: []const u8) !void {
            try self.buffer.appendSlice(self.allocator, str);
        }

        pub fn appendFmt(self: *Builder, comptime format: []const u8, args: anytype) !void {
            try self.buffer.writer(self.allocator).print(format, args);
        }

        pub fn toOwned(self: *Builder) ![]u8 {
            return self.buffer.toOwnedSlice(self.allocator);
        }

        pub fn slice(self: *const Builder) []const u8 {
            return self.buffer.items;
        }

        pub fn reset(self: *Builder) void {
            self.buffer.clearRetainingCapacity();
        }
    };
};

// --- High-Resolution Timing ------------------------------------------------

pub const Time = struct {
    pub const Stopwatch = struct {
        timer: time.Timer,

        pub fn begin() Stopwatch {
            return .{
                .timer = time.Timer.start() catch .{
                    .start_instant = .{ .nanos = 0 },
                },
            };
        }

        pub fn elapsed(self: *Stopwatch) u64 {
            return self.timer.read();
        }

        pub fn lap(self: *Stopwatch) u64 {
            return self.timer.lap();
        }

        pub fn elapsedMs(self: *Stopwatch) f64 {
            return @as(f64, @floatFromInt(self.elapsed())) / 1_000_000.0;
        }

        pub fn elapsedUs(self: *Stopwatch) f64 {
            return @as(f64, @floatFromInt(self.elapsed())) / 1_000.0;
        }
    };
};

// --- Lock-Free Concurrent Primitives ---------------------------------------

pub const Atomic = struct {
    /// Spinlock with exponential backoff -- reduces cache coherence traffic
    /// by capping spin iterations, preventing thundering herd on contention.
    pub const SpinLock = struct {
        state: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),

        pub fn acquire(self: *SpinLock) void {
            var backoff: u32 = 1;
            while (self.state.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) {
                var i: u32 = 0;
                while (i < backoff) : (i += 1) std.atomic.spinLoopHint();
                backoff = @min(backoff * 2, 1024);
            }
        }

        pub fn release(self: *SpinLock) void {
            self.state.store(0, .release);
        }

        pub fn tryAcquire(self: *SpinLock) bool {
            return self.state.cmpxchgStrong(0, 1, .acquire, .monotonic) == null;
        }

        /// RAII-style lock guard
        pub fn guard(self: *SpinLock) Guard {
            self.acquire();
            return .{ .lock = self };
        }

        pub const Guard = struct {
            lock: *SpinLock,
            pub fn release(self: Guard) void {
                self.lock.release();
            }
        };
    };

    /// Single-producer single-consumer bounded queue.
    /// Head and tail are on separate cache lines to avoid false sharing.
    pub fn SpscQueue(comptime T: type, comptime capacity: usize) type {
        comptime std.debug.assert(Math.isPowerOfTwo(capacity));
        const mask = capacity - 1;

        return struct {
            const Self = @This();

            buffer: [capacity]T = undefined,
            head: std.atomic.Value(usize) align(Platform.cache_line_size) = std.atomic.Value(usize).init(0),
            tail: std.atomic.Value(usize) align(Platform.cache_line_size) = std.atomic.Value(usize).init(0),

            pub fn push(self: *Self, item: T) bool {
                const tail = self.tail.load(.monotonic);
                const next = (tail + 1) & mask;
                if (next == self.head.load(.acquire)) return false;
                self.buffer[tail] = item;
                self.tail.store(next, .release);
                return true;
            }

            pub fn pop(self: *Self) ?T {
                const head = self.head.load(.monotonic);
                if (head == self.tail.load(.acquire)) return null;
                const item = self.buffer[head];
                self.head.store((head + 1) & mask, .release);
                return item;
            }

            pub fn len(self: *const Self) usize {
                const tail = self.tail.load(.acquire);
                const head = self.head.load(.acquire);
                return (tail -% head) & mask;
            }

            pub fn isEmpty(self: *const Self) bool {
                return self.head.load(.acquire) == self.tail.load(.acquire);
            }
        };
    }

    /// Atomic counter with relaxed increments and acquire reads
    pub const Counter = struct {
        value: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),

        pub fn increment(self: *Counter) u64 {
            return self.value.fetchAdd(1, .monotonic);
        }

        pub fn add(self: *Counter, n: u64) u64 {
            return self.value.fetchAdd(n, .monotonic);
        }

        pub fn load(self: *const Counter) u64 {
            return self.value.load(.acquire);
        }

        pub fn reset(self: *Counter) void {
            self.value.store(0, .release);
        }
    };
};

// --- Platform Detection (Compile-Time Constants) ---------------------------

pub const Platform = struct {
    pub const is_debug = builtin.mode == .Debug;
    pub const is_linux = builtin.os.tag == .linux;
    pub const is_windows = builtin.os.tag == .windows;
    pub const is_macos = builtin.os.tag == .macos;
    pub const is_x86_64 = builtin.cpu.arch == .x86_64;
    pub const is_aarch64 = builtin.cpu.arch == .aarch64;

    pub const has_avx2 = is_x86_64 and
        std.Target.x86.featureSetHas(builtin.cpu.features, .avx2);

    pub const has_avx512 = is_x86_64 and
        std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f);

    pub const has_neon = is_aarch64 and
        std.Target.aarch64.featureSetHas(builtin.cpu.features, .neon);

    pub const cache_line_size: usize = if (is_aarch64) 128 else 64;
    pub const page_size: usize = if (is_linux or is_macos) 4096 else 65536;
    pub const simd_width: usize = if (has_avx512) 64 else if (has_avx2) 32 else if (has_neon) 16 else 16;

    pub fn description() []const u8 {
        return comptime blk: {
            const arch = if (is_x86_64) "x86_64" else if (is_aarch64) "aarch64" else "unknown";
            const os = if (is_linux) "linux" else if (is_macos) "macos" else if (is_windows) "windows" else "unknown";
            const simd = if (has_avx512) "+avx512" else if (has_avx2) "+avx2" else if (has_neon) "+neon" else "scalar";
            break :blk arch ++ "-" ++ os ++ " " ++ simd;
        };
    }
};

// --- Monadic Result Type ---------------------------------------------------

pub fn Result(comptime T: type, comptime E: type) type {
    return union(enum) {
        ok: T,
        err: E,

        const Self = @This();

        pub fn isOk(self: Self) bool {
            return self == .ok;
        }

        pub fn isErr(self: Self) bool {
            return self == .err;
        }

        pub fn unwrap(self: Self) error{UnwrapError}!T {
            return switch (self) {
                .ok => |val| val,
                .err => error.UnwrapError,
            };
        }

        pub fn unwrapOr(self: Self, default: T) T {
            return switch (self) {
                .ok => |val| val,
                .err => default,
            };
        }

        pub fn unwrapErr(self: Self) error{UnwrapError}!E {
            return switch (self) {
                .ok => error.UnwrapError,
                .err => |e| e,
            };
        }

        /// Transform the ok value, propagating errors
        pub fn map(self: Self, comptime R: type, comptime func: *const fn (T) R) Result(R, E) {
            return switch (self) {
                .ok => |val| .{ .ok = func(val) },
                .err => |e| .{ .err = e },
            };
        }

        /// Convert to Zig's native error union
        pub fn toErrorUnion(self: Self) E!T {
            return switch (self) {
                .ok => |val| val,
                .err => |e| e,
            };
        }
    };
}

// --- Ring Buffer -----------------------------------------------------------

pub fn RingBuffer(comptime T: type, comptime capacity: usize) type {
    comptime std.debug.assert(Math.isPowerOfTwo(capacity));
    const mask = capacity - 1;

    return struct {
        const Self = @This();

        buffer: [capacity]T = undefined,
        head: usize = 0,
        tail: usize = 0,
        count: usize = 0,

        pub fn push(self: *Self, item: T) bool {
            if (self.count == capacity) return false;
            self.buffer[self.tail] = item;
            self.tail = (self.tail + 1) & mask;
            self.count += 1;
            return true;
        }

        pub fn pop(self: *Self) ?T {
            if (self.count == 0) return null;
            const item = self.buffer[self.head];
            self.head = (self.head + 1) & mask;
            self.count -= 1;
            return item;
        }

        pub fn peek(self: *const Self) ?T {
            if (self.count == 0) return null;
            return self.buffer[self.head];
        }

        pub fn isFull(self: *const Self) bool {
            return self.count == capacity;
        }

        pub fn isEmpty(self: *const Self) bool {
            return self.count == 0;
        }

        pub fn len(self: *const Self) usize {
            return self.count;
        }

        pub fn clear(self: *Self) void {
            self.head = 0;
            self.tail = 0;
            self.count = 0;
        }
    };
}

// --- Compile-Time Type Utilities -------------------------------------------

pub const TypeUtils = struct {
    /// Returns the element type of a slice or array
    pub fn ElemType(comptime T: type) type {
        return switch (@typeInfo(T)) {
            .pointer => |p| p.child,
            .array => |a| a.child,
            else => @compileError("Expected slice or array, got " ++ @typeName(T)),
        };
    }

    /// Check if a type has a specific declaration
    pub fn hasDecl(comptime T: type, comptime name: []const u8) bool {
        return @hasDecl(T, name);
    }

    /// Comptime string equal
    pub fn comptimeStrEql(comptime a: []const u8, comptime b: []const u8) bool {
        return std.mem.eql(u8, a, b);
    }
};

// ---- Tests ------------------------------------------------------------------

test "Math.isPowerOfTwo" {
    try std.testing.expect(Math.isPowerOfTwo(@as(u32, 1)));
    try std.testing.expect(Math.isPowerOfTwo(@as(u32, 2)));
    try std.testing.expect(Math.isPowerOfTwo(@as(u32, 256)));
    try std.testing.expect(!Math.isPowerOfTwo(@as(u32, 0)));
    try std.testing.expect(!Math.isPowerOfTwo(@as(u32, 3)));
    try std.testing.expect(!Math.isPowerOfTwo(@as(u32, 100)));
}

test "Math.nextPowerOfTwo" {
    try std.testing.expectEqual(@as(u32, 1), Math.nextPowerOfTwo(u32, 0));
    try std.testing.expectEqual(@as(u32, 1), Math.nextPowerOfTwo(u32, 1));
    try std.testing.expectEqual(@as(u32, 4), Math.nextPowerOfTwo(u32, 3));
    try std.testing.expectEqual(@as(u32, 8), Math.nextPowerOfTwo(u32, 5));
    try std.testing.expectEqual(@as(u32, 16), Math.nextPowerOfTwo(u32, 16));
    try std.testing.expectEqual(@as(u32, 1024), Math.nextPowerOfTwo(u32, 513));
}

test "Math.alignUp" {
    try std.testing.expectEqual(@as(u64, 64), Math.alignUp(u64, 50, 64));
    try std.testing.expectEqual(@as(u64, 64), Math.alignUp(u64, 64, 64));
    try std.testing.expectEqual(@as(u64, 128), Math.alignUp(u64, 65, 64));
    try std.testing.expectEqual(@as(u32, 16), Math.alignUp(u32, 1, 16));
}

test "Math.clamp and divCeil" {
    try std.testing.expectEqual(@as(i32, 5), Math.clamp(i32, 10, 0, 5));
    try std.testing.expectEqual(@as(i32, 0), Math.clamp(i32, -1, 0, 5));
    try std.testing.expectEqual(@as(i32, 3), Math.clamp(i32, 3, 0, 5));

    try std.testing.expectEqual(@as(u32, 3), Math.divCeil(u32, 7, 3));
    try std.testing.expectEqual(@as(u32, 2), Math.divCeil(u32, 6, 3));
    try std.testing.expectEqual(@as(u32, 1), Math.divCeil(u32, 1, 4));
}

test "String.eqlIgnoreCase" {
    try std.testing.expect(String.eqlIgnoreCase("Hello", "hello"));
    try std.testing.expect(String.eqlIgnoreCase("ABC", "abc"));
    try std.testing.expect(!String.eqlIgnoreCase("abc", "abcd"));
    try std.testing.expect(!String.eqlIgnoreCase("abc", "abd"));
}

test "String.Builder" {
    const alloc = std.testing.allocator;
    var sb = String.Builder.init(alloc);
    defer sb.deinit();

    try sb.append("hello");
    try sb.append(" world");
    try std.testing.expectEqualStrings("hello world", sb.slice());

    sb.reset();
    try std.testing.expectEqual(@as(usize, 0), sb.slice().len);
}

test "RingBuffer basic operations" {
    var rb = RingBuffer(u32, 4){};
    try std.testing.expect(rb.isEmpty());
    try std.testing.expect(!rb.isFull());

    try std.testing.expect(rb.push(1));
    try std.testing.expect(rb.push(2));
    try std.testing.expect(rb.push(3));
    try std.testing.expect(rb.push(4));
    try std.testing.expect(rb.isFull());
    try std.testing.expect(!rb.push(5)); // full

    try std.testing.expectEqual(@as(?u32, 1), rb.pop());
    try std.testing.expectEqual(@as(?u32, 2), rb.pop());
    try std.testing.expectEqual(@as(usize, 2), rb.len());

    // Can push again after popping
    try std.testing.expect(rb.push(5));
    try std.testing.expect(rb.push(6));
    try std.testing.expectEqual(@as(?u32, 3), rb.pop());
    try std.testing.expectEqual(@as(?u32, 4), rb.pop());
    try std.testing.expectEqual(@as(?u32, 5), rb.pop());
    try std.testing.expectEqual(@as(?u32, 6), rb.pop());
    try std.testing.expect(rb.isEmpty());
    try std.testing.expectEqual(@as(?u32, null), rb.pop());
}

test "Result monadic type" {
    const R = Result(u32, error{NotFound});

    const ok_val: R = .{ .ok = 42 };
    try std.testing.expect(ok_val.isOk());
    try std.testing.expect(!ok_val.isErr());
    try std.testing.expectEqual(@as(u32, 42), try ok_val.unwrap());
    try std.testing.expectEqual(@as(u32, 42), ok_val.unwrapOr(0));

    const err_val: R = .{ .err = error.NotFound };
    try std.testing.expect(err_val.isErr());
    try std.testing.expectEqual(@as(u32, 99), err_val.unwrapOr(99));
    try std.testing.expectError(error.UnwrapError, err_val.unwrap());
}

test "SpscQueue basic push/pop" {
    var q = Atomic.SpscQueue(u32, 4){};
    try std.testing.expect(q.isEmpty());
    try std.testing.expect(q.push(10));
    try std.testing.expect(q.push(20));
    try std.testing.expectEqual(@as(usize, 2), q.len());
    try std.testing.expectEqual(@as(?u32, 10), q.pop());
    try std.testing.expectEqual(@as(?u32, 20), q.pop());
    try std.testing.expectEqual(@as(?u32, null), q.pop());
}

test "Atomic.Counter" {
    var c = Atomic.Counter{};
    try std.testing.expectEqual(@as(u64, 0), c.load());
    _ = c.increment();
    _ = c.increment();
    _ = c.add(3);
    try std.testing.expectEqual(@as(u64, 5), c.load());
    c.reset();
    try std.testing.expectEqual(@as(u64, 0), c.load());
}

test "Platform description is non-empty" {
    const desc = Platform.description();
    try std.testing.expect(desc.len > 0);
}
