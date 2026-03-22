//! Cross-platform CSPRNG helper for Zig 0.16.
//!
//! Replaces the removed `std.crypto.random` with a DefaultCsprng
//! seeded from OS entropy via platform-specific APIs.

const std = @import("std");
const builtin = @import("builtin");

/// Get a seeded CSPRNG for cryptographic operations.
/// The seed is sourced from OS entropy (arc4random_buf on macOS/BSD,
/// or getrandom on platforms where arc4random_buf is unavailable.
pub fn init() std.Random.DefaultCsprng {
    var seed: [std.Random.DefaultCsprng.secret_seed_length]u8 = undefined;
    fillEntropy(&seed);
    return std.Random.DefaultCsprng.init(seed);
}

/// Fill a buffer with OS-sourced entropy.
fn fillEntropy(buf: []u8) void {
    if (buf.len == 0) return;

    // Use arc4random_buf when available via libc (macOS/BSD).
    if (builtin.link_libc and @TypeOf(std.posix.system.arc4random_buf) != void) {
        std.posix.system.arc4random_buf(buf.ptr, buf.len);
        return;
    }

    // Fallback to getrandom(2) where available.
    if (@TypeOf(std.posix.system.getrandom) != void) {
        var i: usize = 0;
        while (i < buf.len) {
            const out = buf[i..];
            const rc = std.posix.system.getrandom(out.ptr, out.len, 0);
            switch (std.posix.errno(rc)) {
                .SUCCESS => {
                    const n: usize = @intCast(rc);
                    if (n == 0) @panic("secure entropy unavailable via getrandom");
                    i += n;
                },
                .INTR => continue,
                else => |err| std.debug.panic("secure entropy unavailable via getrandom: {t}", .{err}),
            }
        }
        return;
    }

    // Fail closed on unsupported targets instead of silently downgrading entropy.
    @panic("no secure OS entropy source available for this target");
}

/// Convenience: fill a buffer with cryptographically random bytes.
/// Avoids needing to store the CSPRNG instance for one-shot usage.
pub fn fillRandom(buf: []u8) void {
    var rng = init();
    rng.fill(buf);
}

/// Convenience: get a random integer less than `less_than`.
pub fn uintLessThan(comptime T: type, less_than: T) T {
    var rng = init();
    return rng.random().uintLessThan(T, less_than);
}

/// Convenience: shuffle a slice randomly.
pub fn shuffle(comptime T: type, buf: []T) void {
    var rng = init();
    rng.random().shuffle(T, buf);
}

/// Fixed-capacity list that replaces the removed `std.StaticArrayList`.
/// Stores up to `capacity` items in a fixed buffer with no heap allocation.
pub fn FixedList(comptime T: type, comptime capacity: usize) type {
    return struct {
        buffer: [capacity]T = undefined,
        len: usize = 0,

        const Self = @This();

        pub fn append(self: *Self, item: T) error{Overflow}!void {
            if (self.len >= capacity) return error.Overflow;
            self.buffer[self.len] = item;
            self.len += 1;
        }

        pub fn slice(self: *const Self) []const T {
            return self.buffer[0..self.len];
        }
    };
}

test "csprng produces non-zero output" {
    var rng = init();
    const r = rng.random();
    var buf: [32]u8 = undefined;
    r.bytes(&buf);
    // Extremely unlikely to be all zeros from a properly seeded CSPRNG
    var all_zero = true;
    for (buf) |b| {
        if (b != 0) {
            all_zero = false;
            break;
        }
    }
    try std.testing.expect(!all_zero);
}

test "csprng different instances produce different output" {
    var rng1 = init();
    var rng2 = init();
    var buf1: [32]u8 = undefined;
    var buf2: [32]u8 = undefined;
    rng1.random().bytes(&buf1);
    rng2.random().bytes(&buf2);
    // Two separately seeded CSPRNGs should produce different output
    try std.testing.expect(!std.mem.eql(u8, &buf1, &buf2));
}

test "fillRandom supports large buffer" {
    var buf: [4096]u8 = undefined;
    fillRandom(&buf);

    var all_zero = true;
    for (buf) |b| {
        if (b != 0) {
            all_zero = false;
            break;
        }
    }
    try std.testing.expect(!all_zero);
}

test {
    std.testing.refAllDecls(@This());
}
