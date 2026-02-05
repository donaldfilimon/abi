//! Standard Generators for Property-Based Testing
//!
//! Provides composable generators for common types including:
//! - Primitive types (integers, floats, booleans)
//! - Vectors and arrays
//! - Strings and byte slices
//! - Composite types (pairs, optionals, enums)
//!
//! All generators are designed for composition and shrinking support.

const std = @import("std");
const property = @import("mod.zig");
const Generator = property.Generator;

// ============================================================================
// Primitive Generators
// ============================================================================

/// Generate random integers in range [min, max]
pub fn intRange(comptime T: type, min: T, max: T) Generator(T) {
    const GenState = struct {
        var min_val: T = undefined;
        var max_val: T = undefined;

        fn generate(prng: *std.Random.DefaultPrng, _: usize) T {
            return prng.random().intRangeAtMost(T, min_val, max_val);
        }

        fn shrink(value: T, _: std.mem.Allocator) ?T {
            if (value == min_val) return null;
            // Shrink towards min
            const mid = @divFloor(value - min_val, 2) + min_val;
            if (mid != value) return mid;
            return null;
        }
    };

    GenState.min_val = min;
    GenState.max_val = max;

    return .{
        .generateFn = GenState.generate,
        .shrinkFn = GenState.shrink,
    };
}

/// Generate random unsigned integers in range [0, max]
pub fn uintMax(comptime T: type, max: T) Generator(T) {
    return intRange(T, 0, max);
}

/// Generate random booleans
pub fn boolean() Generator(bool) {
    const GenFn = struct {
        fn generate(prng: *std.Random.DefaultPrng, _: usize) bool {
            return prng.random().boolean();
        }
    };

    return .{
        .generateFn = GenFn.generate,
        .shrinkFn = null,
    };
}

/// Generate random f32 in range [0, 1]
pub fn float01() Generator(f32) {
    const GenFn = struct {
        fn generate(prng: *std.Random.DefaultPrng, _: usize) f32 {
            return prng.random().float(f32);
        }
    };

    return .{
        .generateFn = GenFn.generate,
        .shrinkFn = null,
    };
}

/// Generate random f32 in range [-1, 1]
pub fn floatSymmetric() Generator(f32) {
    const GenFn = struct {
        fn generate(prng: *std.Random.DefaultPrng, _: usize) f32 {
            return prng.random().float(f32) * 2.0 - 1.0;
        }
    };

    return .{
        .generateFn = GenFn.generate,
        .shrinkFn = null,
    };
}

/// Generate random f32 in range [min, max]
pub fn floatRange(min: f32, max: f32) Generator(f32) {
    const GenState = struct {
        var min_val: f32 = undefined;
        var max_val: f32 = undefined;

        fn generate(prng: *std.Random.DefaultPrng, _: usize) f32 {
            const t = prng.random().float(f32);
            return min_val + t * (max_val - min_val);
        }
    };

    GenState.min_val = min;
    GenState.max_val = max;

    return .{
        .generateFn = GenState.generate,
        .shrinkFn = null,
    };
}

/// Generate positive f32 values scaled by size
pub fn positiveFloat() Generator(f32) {
    const GenFn = struct {
        fn generate(prng: *std.Random.DefaultPrng, size: usize) f32 {
            const scale = @as(f32, @floatFromInt(@min(size, 1000)));
            return prng.random().float(f32) * scale;
        }
    };

    return .{
        .generateFn = GenFn.generate,
        .shrinkFn = null,
    };
}

/// Generate small positive f32 values for epsilon testing
pub fn smallPositiveFloat() Generator(f32) {
    const GenFn = struct {
        fn generate(prng: *std.Random.DefaultPrng, _: usize) f32 {
            const exp = prng.random().intRangeAtMost(i32, -10, -1);
            return std.math.pow(f32, 10.0, @as(f32, @floatFromInt(exp)));
        }
    };

    return .{
        .generateFn = GenFn.generate,
        .shrinkFn = null,
    };
}

// ============================================================================
// Vector Generators
// ============================================================================

/// Generate random f32 vector of fixed dimension with values in [-1, 1]
pub fn vectorF32(comptime dim: usize) Generator([dim]f32) {
    const GenFn = struct {
        fn generate(prng: *std.Random.DefaultPrng, _: usize) [dim]f32 {
            var result: [dim]f32 = undefined;
            for (&result) |*v| {
                v.* = prng.random().float(f32) * 2.0 - 1.0;
            }
            return result;
        }
    };

    return .{
        .generateFn = GenFn.generate,
        .shrinkFn = null,
    };
}

/// Generate normalized unit vectors of fixed dimension
pub fn unitVector(comptime dim: usize) Generator([dim]f32) {
    const GenFn = struct {
        fn generate(prng: *std.Random.DefaultPrng, _: usize) [dim]f32 {
            var result: [dim]f32 = undefined;
            var sum_sq: f32 = 0.0;

            for (&result) |*v| {
                v.* = prng.random().float(f32) * 2.0 - 1.0;
                sum_sq += v.* * v.*;
            }

            // Normalize
            const mag = @sqrt(sum_sq);
            if (mag > 1e-6) {
                for (&result) |*v| {
                    v.* /= mag;
                }
            } else {
                // Degenerate case: use basis vector
                result[0] = 1.0;
                for (result[1..]) |*v| {
                    v.* = 0.0;
                }
            }

            return result;
        }
    };

    return .{
        .generateFn = GenFn.generate,
        .shrinkFn = null,
    };
}

/// Generate non-zero vectors (for division safety)
pub fn nonZeroVector(comptime dim: usize) Generator([dim]f32) {
    const GenFn = struct {
        fn generate(prng: *std.Random.DefaultPrng, _: usize) [dim]f32 {
            var result: [dim]f32 = undefined;
            var sum_sq: f32 = 0.0;

            for (&result) |*v| {
                v.* = prng.random().float(f32) * 2.0 - 1.0;
                sum_sq += v.* * v.*;
            }

            // Ensure non-zero
            if (sum_sq < 1e-10) {
                result[0] = 1.0;
            }

            return result;
        }
    };

    return .{
        .generateFn = GenFn.generate,
        .shrinkFn = null,
    };
}

/// Generate pair of vectors for binary operation tests
pub fn vectorPairF32(comptime dim: usize) Generator(property.VectorPair(dim)) {
    const GenFn = struct {
        fn generate(prng: *std.Random.DefaultPrng, size: usize) property.VectorPair(dim) {
            const gen = vectorF32(dim);
            return .{
                .a = gen.generate(prng, size),
                .b = gen.generate(prng, size),
            };
        }
    };

    return .{
        .generateFn = GenFn.generate,
        .shrinkFn = null,
    };
}

/// Generate triple of vectors for associativity tests
pub fn vectorTripleF32(comptime dim: usize) Generator(property.VectorTriple(dim)) {
    const GenFn = struct {
        fn generate(prng: *std.Random.DefaultPrng, size: usize) property.VectorTriple(dim) {
            const gen = vectorF32(dim);
            return .{
                .a = gen.generate(prng, size),
                .b = gen.generate(prng, size),
                .c = gen.generate(prng, size),
            };
        }
    };

    return .{
        .generateFn = GenFn.generate,
        .shrinkFn = null,
    };
}

/// Generate scalar-vector pair
pub fn scalarVectorF32(comptime dim: usize) Generator(property.ScalarVectorPair(dim)) {
    const GenFn = struct {
        fn generate(prng: *std.Random.DefaultPrng, size: usize) property.ScalarVectorPair(dim) {
            const vec_gen = vectorF32(dim);
            return .{
                .scalar = prng.random().float(f32) * 2.0 - 1.0,
                .vector = vec_gen.generate(prng, size),
            };
        }
    };

    return .{
        .generateFn = GenFn.generate,
        .shrinkFn = null,
    };
}

// ============================================================================
// Byte and String Generators
// ============================================================================

/// Generate random bytes of variable length up to max_len
/// Note: Returns a slice allocated from page_allocator. Not freed during test.
pub fn bytes(max_len: usize) Generator([]u8) {
    const GenState = struct {
        var max_length: usize = undefined;

        fn generate(prng: *std.Random.DefaultPrng, size: usize) []u8 {
            const len = prng.random().intRangeAtMost(usize, 1, @min(size + 1, max_length));
            const data = std.heap.page_allocator.alloc(u8, len) catch return &.{};
            prng.random().bytes(data);
            return data;
        }
    };

    GenState.max_length = max_len;

    return .{
        .generateFn = GenState.generate,
        .shrinkFn = null,
    };
}

/// Generate printable ASCII strings
/// Note: Returns a slice allocated from page_allocator. Not freed during test.
pub fn asciiString(max_len: usize) Generator([]const u8) {
    const GenState = struct {
        var max_length: usize = undefined;

        fn generate(prng: *std.Random.DefaultPrng, size: usize) []const u8 {
            const len = prng.random().intRangeAtMost(usize, 1, @min(size + 1, max_length));
            const data = std.heap.page_allocator.alloc(u8, len) catch return "";
            for (data) |*c| {
                // Printable ASCII: 32-126
                c.* = @as(u8, @intCast(prng.random().intRangeAtMost(u8, 32, 126)));
            }
            return data;
        }
    };

    GenState.max_length = max_len;

    return .{
        .generateFn = GenState.generate,
        .shrinkFn = null,
    };
}

/// Generate alphanumeric strings (a-z, A-Z, 0-9)
pub fn alphanumericString(max_len: usize) Generator([]const u8) {
    const GenState = struct {
        var max_length: usize = undefined;

        const chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

        fn generate(prng: *std.Random.DefaultPrng, size: usize) []const u8 {
            const len = prng.random().intRangeAtMost(usize, 1, @min(size + 1, max_length));
            const data = std.heap.page_allocator.alloc(u8, len) catch return "";
            for (data) |*c| {
                const idx = prng.random().intRangeLessThan(usize, 0, chars.len);
                c.* = chars[idx];
            }
            return data;
        }
    };

    GenState.max_length = max_len;

    return .{
        .generateFn = GenState.generate,
        .shrinkFn = null,
    };
}

// ============================================================================
// Composite Generators
// ============================================================================

/// Generate one of the given constant values
pub fn oneOf(comptime T: type, values: []const T) Generator(T) {
    const GenState = struct {
        var vals: []const T = undefined;

        fn generate(prng: *std.Random.DefaultPrng, _: usize) T {
            const idx = prng.random().intRangeLessThan(usize, 0, vals.len);
            return vals[idx];
        }
    };

    GenState.vals = values;

    return .{
        .generateFn = GenState.generate,
        .shrinkFn = null,
    };
}

/// Generate optional values (null with 20% probability)
pub fn optional(comptime T: type, inner: Generator(T)) Generator(?T) {
    const GenState = struct {
        var inner_gen: Generator(T) = undefined;

        fn generate(prng: *std.Random.DefaultPrng, size: usize) ?T {
            // 20% chance of null
            if (prng.random().intRangeLessThan(u8, 0, 5) == 0) {
                return null;
            }
            return inner_gen.generate(prng, size);
        }
    };

    GenState.inner_gen = inner;

    return .{
        .generateFn = GenState.generate,
        .shrinkFn = null,
    };
}

/// Generate all enum values with equal probability
pub fn enumValue(comptime E: type) Generator(E) {
    const GenFn = struct {
        fn generate(prng: *std.Random.DefaultPrng, _: usize) E {
            const fields = std.meta.fields(E);
            const idx = prng.random().intRangeLessThan(usize, 0, fields.len);
            return @enumFromInt(fields[idx].value);
        }
    };

    return .{
        .generateFn = GenFn.generate,
        .shrinkFn = null,
    };
}

/// Generate array of fixed size
pub fn array(comptime T: type, comptime N: usize, element_gen: Generator(T)) Generator([N]T) {
    const GenState = struct {
        var elem_gen: Generator(T) = undefined;

        fn generate(prng: *std.Random.DefaultPrng, size: usize) [N]T {
            var result: [N]T = undefined;
            for (&result) |*elem| {
                elem.* = elem_gen.generate(prng, size);
            }
            return result;
        }
    };

    GenState.elem_gen = element_gen;

    return .{
        .generateFn = GenState.generate,
        .shrinkFn = null,
    };
}

// ============================================================================
// Database-Specific Generators
// ============================================================================

/// Generate a batch of vectors with IDs for database testing
pub fn VectorBatch(comptime dim: usize) type {
    return struct {
        vectors: [][dim]f32,
        ids: []u64,
    };
}

/// Generate vector batch for database insertion tests
pub fn vectorBatch(comptime dim: usize, max_count: usize) Generator(VectorBatch(dim)) {
    const GenState = struct {
        var max_c: usize = undefined;

        fn generate(prng: *std.Random.DefaultPrng, size: usize) VectorBatch(dim) {
            const count = prng.random().intRangeAtMost(usize, 1, @min(size + 1, max_c));

            const vectors = std.heap.page_allocator.alloc([dim]f32, count) catch return .{ .vectors = &.{}, .ids = &.{} };
            const ids = std.heap.page_allocator.alloc(u64, count) catch return .{ .vectors = &.{}, .ids = &.{} };

            const vec_gen = unitVector(dim);
            for (vectors, 0..) |*v, i| {
                v.* = vec_gen.generate(prng, size);
                ids[i] = @intCast(i + 1);
            }

            return .{ .vectors = vectors, .ids = ids };
        }
    };

    GenState.max_c = max_count;

    return .{
        .generateFn = GenState.generate,
        .shrinkFn = null,
    };
}

/// Search parameter configuration
pub const SearchParams = struct {
    k: u32,
    ef: u32,
};

/// Generate search parameters (k, ef) for database queries
pub fn searchParams() Generator(SearchParams) {
    const GenFn = struct {
        fn generate(prng: *std.Random.DefaultPrng, size: usize) SearchParams {
            const k = prng.random().intRangeAtMost(u32, 1, @intCast(@min(size + 1, 100)));
            const ef = prng.random().intRangeAtMost(u32, k, k * 4);
            return .{ .k = k, .ef = ef };
        }
    };

    return .{
        .generateFn = GenFn.generate,
        .shrinkFn = null,
    };
}

// ============================================================================
// Concurrency-Specific Generators
// ============================================================================

/// Generate work item for queue tests
pub const WorkItem = struct {
    id: u64,
    priority: u8,
    payload: u64,
};

/// Generate work items for concurrency tests
pub fn workItem() Generator(WorkItem) {
    const GenFn = struct {
        fn generate(prng: *std.Random.DefaultPrng, _: usize) WorkItem {
            return .{
                .id = prng.random().int(u64),
                .priority = prng.random().int(u8),
                .payload = prng.random().int(u64),
            };
        }
    };

    return .{
        .generateFn = GenFn.generate,
        .shrinkFn = null,
    };
}

/// Generate batch of operations for state machine testing
pub const OperationType = enum {
    insert,
    delete,
    search,
    update,
};

/// Generate random operation type
pub fn operationType() Generator(OperationType) {
    return enumValue(OperationType);
}

// ============================================================================
// Tests
// ============================================================================

test "intRange generator" {
    const gen = intRange(i32, 0, 100);
    var prng = std.Random.DefaultPrng.init(42);

    for (0..100) |_| {
        const value = gen.generate(&prng, 50);
        try std.testing.expect(value >= 0 and value <= 100);
    }
}

test "intRange shrinking" {
    const gen = intRange(i32, 0, 100);

    // Test shrink towards minimum
    const shrunk = gen.shrink(50, std.testing.allocator);
    try std.testing.expect(shrunk != null);
    try std.testing.expect(shrunk.? < 50);
    try std.testing.expect(shrunk.? >= 0);

    // Minimum doesn't shrink
    const no_shrink = gen.shrink(0, std.testing.allocator);
    try std.testing.expect(no_shrink == null);
}

test "vectorF32 generator" {
    const gen = vectorF32(8);
    var prng = std.Random.DefaultPrng.init(42);

    for (0..20) |_| {
        const vec = gen.generate(&prng, 50);
        for (vec) |v| {
            try std.testing.expect(v >= -1.0 and v <= 1.0);
        }
    }
}

test "unitVector generator produces normalized vectors" {
    const gen = unitVector(8);
    var prng = std.Random.DefaultPrng.init(42);

    for (0..20) |_| {
        const vec = gen.generate(&prng, 50);

        // Check magnitude is approximately 1.0
        var sum_sq: f32 = 0.0;
        for (vec) |v| {
            sum_sq += v * v;
        }
        const mag = @sqrt(sum_sq);
        try std.testing.expect(@abs(mag - 1.0) < 0.01);
    }
}

test "boolean generator produces both values" {
    const gen = boolean();
    var prng = std.Random.DefaultPrng.init(42);

    var true_count: usize = 0;
    var false_count: usize = 0;

    for (0..100) |_| {
        if (gen.generate(&prng, 0)) {
            true_count += 1;
        } else {
            false_count += 1;
        }
    }

    // Both should appear with reasonable frequency
    try std.testing.expect(true_count > 10);
    try std.testing.expect(false_count > 10);
}

test "oneOf generator" {
    const values = [_]i32{ 1, 2, 3, 4, 5 };
    const gen = oneOf(i32, &values);
    var prng = std.Random.DefaultPrng.init(42);

    for (0..50) |_| {
        const value = gen.generate(&prng, 0);
        var found = false;
        for (values) |v| {
            if (value == v) {
                found = true;
                break;
            }
        }
        try std.testing.expect(found);
    }
}

test "optional generator produces nulls" {
    const inner = intRange(i32, 0, 100);
    const gen = optional(i32, inner);
    var prng = std.Random.DefaultPrng.init(42);

    var null_count: usize = 0;
    var value_count: usize = 0;

    for (0..100) |_| {
        if (gen.generate(&prng, 50)) |_| {
            value_count += 1;
        } else {
            null_count += 1;
        }
    }

    // Should have both nulls and values
    try std.testing.expect(null_count > 0);
    try std.testing.expect(value_count > 0);
}

test "searchParams generator constraints" {
    const gen = searchParams();
    var prng = std.Random.DefaultPrng.init(42);

    for (0..50) |_| {
        const params = gen.generate(&prng, 50);
        // k should be >= 1
        try std.testing.expect(params.k >= 1);
        // ef should be >= k
        try std.testing.expect(params.ef >= params.k);
    }
}
