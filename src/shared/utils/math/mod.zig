//! Math Utilities Module
//! Contains mathematical functions, statistical calculations, and numerical utilities

const std = @import("std");

// =============================================================================
// BASIC MATH UTILITIES
// =============================================================================

/// Basic mathematical utility functions
pub const MathUtils = struct {
    /// Clamp value between min and max
    pub fn clamp(comptime T: type, value: T, min: T, max: T) T {
        if (value < min) return min;
        if (value > max) return max;
        return value;
    }

    /// Linear interpolation between a and b
    pub fn lerp(a: f64, b: f64, t: f64) f64 {
        return a + (b - a) * clamp(f64, t, 0.0, 1.0);
    }

    /// Calculate percentage (value / total * 100)
    pub fn percentage(value: f64, total: f64) f64 {
        if (total == 0.0) return 0.0;
        return (value / total) * 100.0;
    }

    /// Round to specified decimal places
    pub fn roundToDecimal(value: f64, decimals: usize) f64 {
        const multiplier = std.math.pow(f64, 10.0, @as(f64, @floatFromInt(decimals)));
        return @round(value * multiplier) / multiplier;
    }

    /// Check if number is power of 2
    pub fn isPowerOfTwo(value: usize) bool {
        return value != 0 and (value & (value - 1)) == 0;
    }

    /// Find next power of 2 greater than or equal to value
    pub fn nextPowerOfTwo(value: usize) usize {
        if (value == 0) return 1;
        var result = value - 1;
        result |= result >> 1;
        result |= result >> 2;
        result |= result >> 4;
        result |= result >> 8;
        result |= result >> 16;
        if (@sizeOf(usize) > 4) {
            result |= result >> 32;
        }
        return result + 1;
    }

    /// Calculate factorial
    pub fn factorial(n: u64) u64 {
        if (n == 0 or n == 1) return 1;
        var result: u64 = 1;
        var i: u64 = 2;
        while (i <= n) : (i += 1) {
            result *= i;
        }
        return result;
    }

    /// Calculate greatest common divisor (GCD)
    pub fn gcd(a: usize, b: usize) usize {
        var x = a;
        var y = b;
        while (y != 0) {
            const t = y;
            y = x % y;
            x = t;
        }
        return x;
    }

    /// Calculate least common multiple (LCM)
    pub fn lcm(a: usize, b: usize) usize {
        if (a == 0 or b == 0) return 0;
        return (a / gcd(a, b)) * b;
    }
};

// =============================================================================
// STATISTICAL FUNCTIONS
// =============================================================================

/// Statistical calculation functions
pub const Statistics = struct {
    /// Calculate mean (average)
    pub fn mean(values: []const f64) f64 {
        if (values.len == 0) return 0.0;
        var sum: f64 = 0.0;
        for (values) |v| sum += v;
        return sum / @as(f64, @floatFromInt(values.len));
    }

    /// Calculate standard deviation
    pub fn standardDeviation(values: []const f64) f64 {
        if (values.len < 2) return 0.0;

        const avg = mean(values);
        var sum_squares: f64 = 0.0;

        for (values) |v| {
            const diff = v - avg;
            sum_squares += diff * diff;
        }

        return std.math.sqrt(sum_squares / @as(f64, @floatFromInt(values.len - 1)));
    }

    /// Calculate median
    pub fn median(allocator: std.mem.Allocator, values: []const f64) !f64 {
        if (values.len == 0) return 0.0;

        // Create a copy to sort
        const sorted = try allocator.alloc(f64, values.len);
        defer allocator.free(sorted);
        @memcpy(sorted, values);
        std.mem.sort(f64, sorted, {}, std.sort.asc(f64));

        const mid = values.len / 2;
        if (values.len % 2 == 0) {
            return (sorted[mid - 1] + sorted[mid]) / 2.0;
        } else {
            return sorted[mid];
        }
    }

    /// Calculate variance
    pub fn variance(values: []const f64) f64 {
        if (values.len < 2) return 0.0;

        const avg = mean(values);
        var sum_squares: f64 = 0.0;

        for (values) |v| {
            const diff = v - avg;
            sum_squares += diff * diff;
        }

        return sum_squares / @as(f64, @floatFromInt(values.len - 1));
    }

    /// Calculate minimum value
    pub fn min(values: []const f64) f64 {
        if (values.len == 0) return 0.0;
        var result = values[0];
        for (values[1..]) |v| {
            if (v < result) result = v;
        }
        return result;
    }

    /// Calculate maximum value
    pub fn max(values: []const f64) f64 {
        if (values.len == 0) return 0.0;
        var result = values[0];
        for (values[1..]) |v| {
            if (v > result) result = v;
        }
        return result;
    }

    /// Calculate range (max - min)
    pub fn range(values: []const f64) f64 {
        return max(values) - min(values);
    }
};

// =============================================================================
// GEOMETRY AND DISTANCE
// =============================================================================

/// Geometric and distance calculation functions
pub const Geometry = struct {
    /// Calculate distance between two points (2D)
    pub fn distance2D(x1: f64, y1: f64, x2: f64, y2: f64) f64 {
        const dx = x2 - x1;
        const dy = y2 - y1;
        return std.math.sqrt(dx * dx + dy * dy);
    }

    /// Calculate distance between two points (3D)
    pub fn distance3D(x1: f64, y1: f64, z1: f64, x2: f64, y2: f64, z2: f64) f64 {
        const dx = x2 - x1;
        const dy = y2 - y1;
        const dz = z2 - z1;
        return std.math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    /// Calculate Euclidean distance (N-dimensional)
    pub fn distance(a: []const f64, b: []const f64) f64 {
        if (a.len != b.len) return 0.0;

        var sum: f64 = 0.0;
        for (a, b) |x, y| {
            const diff = x - y;
            sum += diff * diff;
        }
        return std.math.sqrt(sum);
    }

    /// Calculate Manhattan distance (L1 norm)
    pub fn manhattanDistance(a: []const f64, b: []const f64) f64 {
        if (a.len != b.len) return 0.0;

        var sum: f64 = 0.0;
        for (a, b) |x, y| {
            sum += @abs(x - y);
        }
        return sum;
    }

    /// Calculate cosine similarity
    pub fn cosineSimilarity(a: []const f64, b: []const f64) f64 {
        if (a.len != b.len) return 0.0;

        var dot_product: f64 = 0.0;
        var norm_a: f64 = 0.0;
        var norm_b: f64 = 0.0;

        for (a, b) |x, y| {
            dot_product += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }

        norm_a = std.math.sqrt(norm_a);
        norm_b = std.math.sqrt(norm_b);

        if (norm_a == 0.0 or norm_b == 0.0) return 0.0;
        return dot_product / (norm_a * norm_b);
    }
};

// =============================================================================
// ANGULAR CONVERSIONS
// =============================================================================

/// Angular conversion utilities
pub const Angles = struct {
    /// Convert degrees to radians
    pub fn degreesToRadians(degrees: f64) f64 {
        return degrees * (std.math.pi / 180.0);
    }

    /// Convert radians to degrees
    pub fn radiansToDegrees(radians: f64) f64 {
        return radians * (180.0 / std.math.pi);
    }

    /// Normalize angle to [0, 2π) range
    pub fn normalizeRadians(angle: f64) f64 {
        const two_pi = 2.0 * std.math.pi;
        var result = @mod(angle, two_pi);
        if (result < 0.0) result += two_pi;
        return result;
    }

    /// Normalize angle to [-180, 180] degree range
    pub fn normalizeDegrees(angle: f64) f64 {
        const result = @mod(angle + 180.0, 360.0) - 180.0;
        return result;
    }
};

// =============================================================================
// RANDOM NUMBER UTILITIES
// =============================================================================

/// Random number generation utilities
pub const Random = struct {
    /// Generate random integer in range [min, max)
    pub fn intRange(random: std.rand.Random, min: i64, max: i64) i64 {
        if (min >= max) return min;
        return min + @as(i64, @intCast(random.uintLessThan(@as(u64, @intCast(max - min)))));
    }

    /// Generate random float in range [min, max)
    pub fn floatRange(random: std.rand.Random, min: f64, max: f64) f64 {
        if (min >= max) return min;
        return min + (max - min) * random.float(f64);
    }

    /// Generate random boolean with given probability
    pub fn boolean(random: std.rand.Random, probability: f64) bool {
        return random.float(f64) < MathUtils.clamp(f64, probability, 0.0, 1.0);
    }

    /// Select random element from slice
    pub fn choice(comptime T: type, random: std.rand.Random, items: []const T) ?T {
        if (items.len == 0) return null;
        return items[random.uintLessThan(items.len)];
    }

    /// Shuffle slice in place
    pub fn shuffle(comptime T: type, random: std.rand.Random, items: []T) void {
        for (0..items.len) |i| {
            const j = random.uintLessThan(items.len - i) + i;
            std.mem.swap(T, &items[i], &items[j]);
        }
    }
};

// =============================================================================
// TESTS
// =============================================================================

test "MathUtils basic operations" {
    // Test clamp
    try std.testing.expectEqual(@as(i32, 5), MathUtils.clamp(i32, 10, 0, 5));
    try std.testing.expectEqual(@as(i32, 0), MathUtils.clamp(i32, -5, 0, 5));
    try std.testing.expectEqual(@as(i32, 3), MathUtils.clamp(i32, 3, 0, 5));

    // Test lerp
    try std.testing.expectApproxEqAbs(@as(f64, 7.5), MathUtils.lerp(5.0, 10.0, 0.5), 0.001);

    // Test power of two
    try std.testing.expect(MathUtils.isPowerOfTwo(8));
    try std.testing.expect(!MathUtils.isPowerOfTwo(9));

    // Test next power of two
    try std.testing.expectEqual(@as(usize, 8), MathUtils.nextPowerOfTwo(5));
    try std.testing.expectEqual(@as(usize, 16), MathUtils.nextPowerOfTwo(9));
}

test "Statistics basic operations" {
    const values = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };

    // Test mean
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), Statistics.mean(&values), 0.001);

    // Test min/max
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), Statistics.min(&values), 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), Statistics.max(&values), 0.001);

    // Test range
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), Statistics.range(&values), 0.001);
}

test "Geometry distance calculations" {
    // Test 2D distance
    const dist2d = Geometry.distance2D(0, 0, 3, 4);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), dist2d, 0.001);

    // Test 3D distance
    const dist3d = Geometry.distance3D(0, 0, 0, 1, 1, 1);
    try std.testing.expectApproxEqAbs(@as(f64, 1.732), dist3d, 0.1);

    // Test N-dimensional distance
    const a = [_]f64{ 0, 0 };
    const b = [_]f64{ 3, 4 };
    const dist = Geometry.distance(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), dist, 0.001);
}

test "Angles conversions" {
    // Test degree/radian conversion
    const ninety_deg = Angles.degreesToRadians(90.0);
    try std.testing.expectApproxEqAbs(std.math.pi / 2.0, ninety_deg, 0.001);

    const ninety_rad = Angles.radiansToDegrees(std.math.pi / 2.0);
    try std.testing.expectApproxEqAbs(@as(f64, 90.0), ninety_rad, 0.001);
}
