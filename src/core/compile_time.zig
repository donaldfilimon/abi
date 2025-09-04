//! Compile-Time Utilities Module
//! Leverages Zig's comptime features for optimal performance and modern patterns

const std = @import("std");

/// Compile-time factorial calculation
pub fn factorial(comptime n: u32) u32 {
    return if (n == 0) 1 else n * factorial(n - 1);
}

/// Compile-time power calculation
pub fn power(comptime base: u32, comptime exponent: u32) u32 {
    return if (exponent == 0) 1 else base * power(base, exponent - 1);
}

/// Compile-time fibonacci sequence
pub fn fibonacci(comptime n: u32) u32 {
    return if (n <= 1) n else fibonacci(n - 1) + fibonacci(n - 2);
}

/// Compile-time prime number checker
pub fn isPrime(comptime n: u32) bool {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;

    var i: u32 = 3;
    while (i * i <= n) : (i += 2) {
        if (n % i == 0) return false;
    }
    return true;
}

/// Compile-time array generation with function
pub fn generateArray(comptime T: type, comptime size: usize, comptime generator: fn (usize) T) [size]T {
    var result: [size]T = undefined;
    for (0..size) |i| {
        result[i] = generator(i);
    }
    return result;
}

/// Compile-time lookup table generator
pub fn generateLookupTable(comptime T: type, comptime size: usize, comptime func: fn (usize) T) [size]T {
    return generateArray(T, size, func);
}

/// Compile-time string operations
pub const StringOps = struct {
    /// Compile-time string length
    pub fn length(comptime str: []const u8) usize {
        return str.len;
    }

    /// Compile-time string concatenation
    pub fn concat(comptime a: []const u8, comptime b: []const u8) [a.len + b.len]u8 {
        var result: [a.len + b.len]u8 = undefined;
        @memcpy(result[0..a.len], a);
        @memcpy(result[a.len..], b);
        return result;
    }

    /// Compile-time string repetition
    pub fn repeat(comptime str: []const u8, comptime count: usize) [str.len * count]u8 {
        var result: [str.len * count]u8 = undefined;
        var i: usize = 0;
        while (i < count) : (i += 1) {
            @memcpy(result[i * str.len ..][0..str.len], str);
        }
        return result;
    }
};

/// Compile-time math utilities
pub const MathUtils = struct {
    /// Compile-time greatest common divisor
    pub fn gcd(comptime a: u32, comptime b: u32) u32 {
        return if (b == 0) a else gcd(b, a % b);
    }

    /// Compile-time least common multiple
    pub fn lcm(comptime a: u32, comptime b: u32) u32 {
        return (a * b) / gcd(a, b);
    }

    /// Compile-time square root approximation
    pub fn sqrt(comptime n: u32) u32 {
        if (n == 0 or n == 1) return n;

        var x: u32 = n;
        var y: u32 = 1;
        while (x > y) {
            x = (x + y) / 2;
            y = n / x;
        }
        return x;
    }

    /// Compile-time log2 calculation
    pub fn log2(comptime n: u32) u32 {
        var result: u32 = 0;
        var temp: u32 = n;
        while (temp > 1) : (temp >>= 1) {
            result += 1;
        }
        return result;
    }
};

/// Compile-time bit manipulation utilities
pub const BitUtils = struct {
    /// Compile-time bit count
    pub fn bitCount(comptime n: u32) u32 {
        var count: u32 = 0;
        var temp: u32 = n;
        while (temp > 0) : (temp >>= 1) {
            count += temp & 1;
        }
        return count;
    }

    /// Compile-time next power of 2
    pub fn nextPowerOf2(comptime n: u32) u32 {
        if (n == 0) return 1;
        var result: u32 = 1;
        while (result < n) : (result <<= 1) {}
        return result;
    }

    /// Compile-time is power of 2 check
    pub fn isPowerOf2(comptime n: u32) bool {
        return n != 0 and (n & (n - 1)) == 0;
    }

    /// Compile-time bit reversal
    pub fn reverseBits(comptime n: u32) u32 {
        var result: u32 = 0;
        var temp: u32 = n;
        var i: u32 = 0;
        while (i < 32) : (i += 1) {
            result = (result << 1) | (temp & 1);
            temp >>= 1;
        }
        return result;
    }
};

/// Compile-time type utilities
pub const TypeUtils = struct {
    /// Check if type is numeric
    pub fn isNumeric(comptime T: type) bool {
        return switch (@typeInfo(T)) {
            .Int, .Float, .ComptimeInt, .ComptimeFloat => true,
            else => false,
        };
    }

    /// Check if type is integer
    pub fn isInteger(comptime T: type) bool {
        return switch (@typeInfo(T)) {
            .Int, .ComptimeInt => true,
            else => false,
        };
    }

    /// Check if type is float
    pub fn isFloat(comptime T: type) bool {
        return switch (@typeInfo(T)) {
            .Float, .ComptimeFloat => true,
            else => false,
        };
    }

    /// Get type size in bits
    pub fn bitSize(comptime T: type) usize {
        return @sizeOf(T) * 8;
    }
};

/// Compile-time validation utilities
pub const ValidationUtils = struct {
    /// Compile-time range validation
    pub fn inRange(comptime value: anytype, comptime min: anytype, comptime max: anytype) bool {
        return value >= min and value <= max;
    }

    /// Compile-time array bounds check
    pub fn validIndex(comptime index: usize, comptime array_size: usize) bool {
        return index < array_size;
    }

    /// Compile-time power of 2 validation
    pub fn isValidPowerOf2(comptime n: anytype) bool {
        return n > 0 and (n & (n - 1)) == 0;
    }
};

/// Compile-time constants
pub const Constants = struct {
    pub const PI = 3.141592653589793;
    pub const E = 2.718281828459045;
    pub const GOLDEN_RATIO = 1.618033988749895;
    pub const SQRT_2 = 1.4142135623730951;
    pub const SQRT_3 = 1.7320508075688772;

    // Precomputed factorials for common values
    pub const FACTORIALS = [_]u64{
        1,        1,         2,          6,           24,            120,            720,             5040,             40320,              362880,              3628800,
        39916800, 479001600, 6227020800, 87178291200, 1307674368000, 20922789888000, 355687428096000, 6402373705728000, 121645100408832000, 2432902008176640000,
    };

    // Precomputed powers of 2
    pub const POWERS_OF_2 = [_]u64{
        1,                2,                4,                 8,                 16,                32,                 64,                 128,                256,                 512,                 1024,                2048,                4096,
        8192,             16384,            32768,             65536,             131072,            262144,             524288,             1048576,            2097152,             4194304,             8388608,             16777216,            33554432,
        67108864,         134217728,        268435456,         536870912,         1073741824,        2147483648,         4294967296,         8589934592,         17179869184,         34359738368,         68719476736,         137438953472,        274877906944,
        549755813888,     1099511627776,    2199023255552,     4398046511104,     8796093022208,     17592186044416,     35184372088832,     70368744177664,     140737488355328,     281474976710656,     562949953421312,     1125899906842624,    2251799813685248,
        4503599627370496, 9007199254740992, 18014398509481984, 36028797018963968, 72057594037927936, 144115188075855872, 288230376151711744, 576460752303423488, 1152921504606846976, 2305843009213693952, 4611686018427387904, 9223372036854775808,
    };
};

test "compile-time utilities" {
    // Test factorial
    try std.testing.expectEqual(@as(u32, 120), factorial(5));
    try std.testing.expectEqual(@as(u32, 1), factorial(0));

    // Test power
    try std.testing.expectEqual(@as(u32, 8), power(2, 3));
    try std.testing.expectEqual(@as(u32, 1), power(5, 0));

    // Test fibonacci
    try std.testing.expectEqual(@as(u32, 8), fibonacci(6));
    try std.testing.expectEqual(@as(u32, 1), fibonacci(1));

    // Test prime checking
    try std.testing.expect(isPrime(2));
    try std.testing.expect(isPrime(3));
    try std.testing.expect(!isPrime(4));
    try std.testing.expect(isPrime(5));

    // Test array generation
    const squares = generateArray(u32, 5, struct {
        fn square(i: usize) u32 {
            return @intCast(i * i);
        }
    }.square);
    try std.testing.expectEqual(@as(u32, 0), squares[0]);
    try std.testing.expectEqual(@as(u32, 1), squares[1]);
    try std.testing.expectEqual(@as(u32, 4), squares[2]);
    try std.testing.expectEqual(@as(u32, 9), squares[3]);
    try std.testing.expectEqual(@as(u32, 16), squares[4]);

    // Test string operations
    const hello = "Hello";
    const world = "World";
    const hello_world = StringOps.concat(hello, world);
    try std.testing.expectEqualStrings("HelloWorld", &hello_world);

    // Test math utilities
    try std.testing.expectEqual(@as(u32, 6), MathUtils.gcd(12, 18));
    try std.testing.expectEqual(@as(u32, 36), MathUtils.lcm(12, 18));
    try std.testing.expectEqual(@as(u32, 4), MathUtils.sqrt(16));
    try std.testing.expectEqual(@as(u32, 3), MathUtils.log2(8));

    // Test bit utilities
    try std.testing.expectEqual(@as(u32, 2), BitUtils.bitCount(5));
    try std.testing.expectEqual(@as(u32, 8), BitUtils.nextPowerOf2(5));
    try std.testing.expect(BitUtils.isPowerOf2(8));
    try std.testing.expect(!BitUtils.isPowerOf2(5));

    // Test type utilities
    try std.testing.expect(TypeUtils.isNumeric(u32));
    try std.testing.expect(TypeUtils.isInteger(u32));
    try std.testing.expect(!TypeUtils.isFloat(u32));
    try std.testing.expectEqual(@as(usize, 32), TypeUtils.bitSize(u32));

    // Test validation utilities
    try std.testing.expect(ValidationUtils.inRange(5, 1, 10));
    try std.testing.expect(!ValidationUtils.inRange(15, 1, 10));
    try std.testing.expect(ValidationUtils.validIndex(2, 5));
    try std.testing.expect(!ValidationUtils.validIndex(5, 5));
    try std.testing.expect(ValidationUtils.isValidPowerOf2(8));
    try std.testing.expect(!ValidationUtils.isValidPowerOf2(5));

    // Test constants
    try std.testing.expectEqual(@as(u64, 120), Constants.FACTORIALS[5]);
    try std.testing.expectEqual(@as(u64, 8), Constants.POWERS_OF_2[3]);
}
