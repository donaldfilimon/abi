//! Property-Based Testing Module
//!
//! Comprehensive property-based testing infrastructure with:
//! - Composable generators for arbitrary test data
//! - Automatic shrinking to find minimal failing cases
//! - Reproducible tests via seeded random number generation
//! - Domain-specific generators for vectors, databases, and concurrency
//!
//! ## Quick Start
//!
//! ```zig
//! const proptest = @import("property");
//!
//! test "vector dot product is commutative" {
//!     const result = try proptest.forAll(
//!         proptest.generators.vectorPairF32(8),
//!         .{ .iterations = 100, .seed = 42 },
//!         struct {
//!             fn check(pair: proptest.VectorPair(8)) bool {
//!                 const dot_ab = vectorDot(pair.a, pair.b);
//!                 const dot_ba = vectorDot(pair.b, pair.a);
//!                 return @abs(dot_ab - dot_ba) < 1e-6;
//!             }
//!         }.check,
//!     );
//!     try std.testing.expect(result.passed);
//! }
//! ```
//!
//! ## Architecture
//!
//! - `mod.zig` - Main entry point and test runner
//! - `generators.zig` - Standard generators for primitive types
//! - `vector_properties.zig` - Vector math property tests
//! - `database_properties.zig` - Database operation property tests
//! - `serialization_properties.zig` - Serialization roundtrip tests
//! - `concurrency_properties.zig` - Lock-free structure property tests

const std = @import("std");
const abi = @import("abi");
const time = abi.shared.time;

// Re-export submodules
pub const generators = @import("generators.zig");
pub const vector_properties = @import("vector_properties.zig");
pub const database_properties = @import("database_properties.zig");
pub const serialization_properties = @import("serialization_properties.zig");
pub const concurrency_properties = @import("concurrency_properties.zig");

// Note: Legacy proptest available at src/tests/proptest.zig
// Import via @import("proptest.zig") from src/tests/mod.zig

// ============================================================================
// Configuration
// ============================================================================

/// Property test configuration
pub const PropertyConfig = struct {
    /// Number of test iterations
    iterations: u32 = 100,
    /// Seed for random number generator (null for time-based seed)
    seed: ?u64 = null,
    /// Maximum shrink iterations when finding minimal failing case
    shrink_iterations: u32 = 100,
    /// Print progress and debug information
    verbose: bool = false,
    /// Maximum size hint for generators (controls complexity)
    max_size: usize = 100,
};

/// Result of a property test run
pub const PropertyResult = struct {
    /// Whether all tests passed
    passed: bool,
    /// Number of iterations completed
    iterations_run: u32,
    /// Seed used for reproducibility
    seed_used: u64,
    /// Shrunk failing example (if any)
    shrunk_example: ?[]const u8 = null,
    /// Failure message (if any)
    failure_message: ?[]const u8 = null,
    /// Allocator used for cleanup
    allocator: ?std.mem.Allocator = null,

    pub fn deinit(self: *PropertyResult) void {
        if (self.allocator) |alloc| {
            if (self.shrunk_example) |ex| alloc.free(ex);
            if (self.failure_message) |msg| alloc.free(msg);
        }
        self.* = undefined;
    }

    pub fn success(iterations: u32, seed: u64) PropertyResult {
        return .{
            .passed = true,
            .iterations_run = iterations,
            .seed_used = seed,
        };
    }

    pub fn failure(iterations: u32, seed: u64, allocator: std.mem.Allocator, example: ?[]const u8, message: ?[]const u8) PropertyResult {
        return .{
            .passed = false,
            .iterations_run = iterations,
            .seed_used = seed,
            .shrunk_example = example,
            .failure_message = message,
            .allocator = allocator,
        };
    }
};

// ============================================================================
// Generator Interface
// ============================================================================

/// Generic generator interface for producing random values
pub fn Generator(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Function to generate a value
        generateFn: *const fn (*std.Random.DefaultPrng, usize) T,
        /// Optional function to shrink a value
        shrinkFn: ?*const fn (T, std.mem.Allocator) ?T = null,

        /// Generate a random value
        pub fn generate(self: Self, prng: *std.Random.DefaultPrng, size: usize) T {
            return self.generateFn(prng, size);
        }

        /// Try to shrink a value to a simpler failing case
        pub fn shrink(self: Self, value: T, allocator: std.mem.Allocator) ?T {
            if (self.shrinkFn) |f| {
                return f(value, allocator);
            }
            return null;
        }

        /// Map a generator to a different type
        pub fn map(self: Self, comptime U: type, comptime f: fn (T) U) Generator(U) {
            const Mapper = struct {
                var inner_gen: Self = undefined;

                fn generateMapped(prng: *std.Random.DefaultPrng, size: usize) U {
                    const t_val = inner_gen.generate(prng, size);
                    return f(t_val);
                }
            };
            Mapper.inner_gen = self;

            return .{
                .generateFn = Mapper.generateMapped,
                .shrinkFn = null,
            };
        }
    };
}

// ============================================================================
// Property Test Runner
// ============================================================================

/// Run a property test with the given generator and predicate
pub fn forAll(
    comptime T: type,
    generator: Generator(T),
    config: PropertyConfig,
    predicate: *const fn (T) bool,
) PropertyResult {
    const seed = config.seed orelse @as(u64, @intCast(time.unixMilliseconds()));
    var prng = std.Random.DefaultPrng.init(seed);

    var iterations: u32 = 0;
    while (iterations < config.iterations) : (iterations += 1) {
        const size = @min(iterations + 1, config.max_size);
        const value = generator.generate(&prng, size);

        if (!predicate(value)) {
            // Attempt to shrink
            var shrunk_value = value;
            var shrink_count: u32 = 0;

            while (shrink_count < config.shrink_iterations) : (shrink_count += 1) {
                if (generator.shrink(shrunk_value, std.heap.page_allocator)) |smaller| {
                    if (!predicate(smaller)) {
                        shrunk_value = smaller;
                    }
                } else {
                    break;
                }
            }

            if (config.verbose) {
                std.debug.print("Property failed at iteration {d} (seed: {d})\n", .{ iterations, seed });
            }

            return PropertyResult.failure(
                iterations,
                seed,
                std.heap.page_allocator,
                std.fmt.allocPrint(std.heap.page_allocator, "{any}", .{shrunk_value}) catch null,
                std.fmt.allocPrint(std.heap.page_allocator, "Failed at iteration {d}", .{iterations}) catch null,
            );
        }

        if (config.verbose and (iterations + 1) % 10 == 0) {
            std.debug.print("Progress: {d}/{d} passed\n", .{ iterations + 1, config.iterations });
        }
    }

    return PropertyResult.success(iterations, seed);
}

/// Run a property test with an allocator-aware predicate
pub fn forAllWithAllocator(
    comptime T: type,
    allocator: std.mem.Allocator,
    generator: Generator(T),
    config: PropertyConfig,
    predicate: *const fn (T, std.mem.Allocator) bool,
) PropertyResult {
    const seed = config.seed orelse @as(u64, @intCast(time.unixMilliseconds()));
    var prng = std.Random.DefaultPrng.init(seed);

    var iterations: u32 = 0;
    while (iterations < config.iterations) : (iterations += 1) {
        const size = @min(iterations + 1, config.max_size);
        const value = generator.generate(&prng, size);

        if (!predicate(value, allocator)) {
            return PropertyResult.failure(
                iterations,
                seed,
                allocator,
                std.fmt.allocPrint(allocator, "{any}", .{value}) catch null,
                std.fmt.allocPrint(allocator, "Failed at iteration {d}", .{iterations}) catch null,
            );
        }
    }

    return PropertyResult.success(iterations, seed);
}

// ============================================================================
// Assertion Helpers
// ============================================================================

/// Assertion utilities for property tests
pub const assert = struct {
    /// Assert two values are equal
    pub fn equal(comptime T: type, a: T, b: T) bool {
        return a == b;
    }

    /// Assert two floats are approximately equal
    pub fn approxEqual(a: f32, b: f32, epsilon: f32) bool {
        return @abs(a - b) < epsilon;
    }

    /// Assert two f64s are approximately equal
    pub fn approxEqual64(a: f64, b: f64, epsilon: f64) bool {
        return @abs(a - b) < epsilon;
    }

    /// Assert a value is within a range [min, max]
    pub fn inRange(comptime T: type, value: T, min: T, max: T) bool {
        return value >= min and value <= max;
    }

    /// Assert a float is finite (not NaN or Inf)
    pub fn isFinite(value: anytype) bool {
        const T = @TypeOf(value);
        return switch (@typeInfo(T)) {
            .float => !std.math.isNan(value) and !std.math.isInf(value),
            else => true,
        };
    }

    /// Assert all elements of a slice are finite
    pub fn allFinite(slice: []const f32) bool {
        for (slice) |v| {
            if (std.math.isNan(v) or std.math.isInf(v)) return false;
        }
        return true;
    }

    /// Assert a slice is sorted (ascending)
    pub fn isSorted(comptime T: type, slice: []const T) bool {
        if (slice.len < 2) return true;
        for (slice[0 .. slice.len - 1], slice[1..]) |a, b| {
            if (a > b) return false;
        }
        return true;
    }

    /// Assert two slices are equal
    pub fn slicesEqual(comptime T: type, a: []const T, b: []const T) bool {
        if (a.len != b.len) return false;
        for (a, b) |x, y| {
            if (x != y) return false;
        }
        return true;
    }

    /// Assert two f32 slices are approximately equal
    pub fn slicesApproxEqual(a: []const f32, b: []const f32, epsilon: f32) bool {
        if (a.len != b.len) return false;
        for (a, b) |x, y| {
            if (@abs(x - y) >= epsilon) return false;
        }
        return true;
    }
};

// ============================================================================
// Common Type Definitions
// ============================================================================

/// Pair of vectors for property testing
pub fn VectorPair(comptime dim: usize) type {
    return struct {
        a: [dim]f32,
        b: [dim]f32,
    };
}

/// Triple of vectors for associativity tests
pub fn VectorTriple(comptime dim: usize) type {
    return struct {
        a: [dim]f32,
        b: [dim]f32,
        c: [dim]f32,
    };
}

/// Scalar-vector pair for commutativity tests
pub fn ScalarVectorPair(comptime dim: usize) type {
    return struct {
        scalar: f32,
        vector: [dim]f32,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "property test basic pass" {
    const gen = generators.intRange(i32, 0, 100);

    const isPositive = struct {
        fn check(x: i32) bool {
            return x >= 0;
        }
    }.check;

    const result = forAll(i32, gen, .{ .iterations = 50, .seed = 42 }, isPositive);
    try std.testing.expect(result.passed);
    try std.testing.expectEqual(@as(u32, 50), result.iterations_run);
}

test "property test detects failure" {
    const gen = generators.intRange(i32, -100, 100);

    const alwaysPositive = struct {
        fn check(x: i32) bool {
            return x > 0;
        }
    }.check;

    var result = forAll(i32, gen, .{ .iterations = 1000, .seed = 42 }, alwaysPositive);
    defer result.deinit();

    try std.testing.expect(!result.passed);
    try std.testing.expect(result.iterations_run < 1000);
}

test "property test reproducibility" {
    const gen = generators.intRange(i32, 0, 1000);

    const property = struct {
        fn check(x: i32) bool {
            return x < 500;
        }
    }.check;

    // Run twice with same seed
    var result1 = forAll(i32, gen, .{ .iterations = 1000, .seed = 12345 }, property);
    defer result1.deinit();
    var result2 = forAll(i32, gen, .{ .iterations = 1000, .seed = 12345 }, property);
    defer result2.deinit();

    // Should fail at same iteration
    try std.testing.expectEqual(result1.iterations_run, result2.iterations_run);
}

test "assertions work correctly" {
    try std.testing.expect(assert.approxEqual(1.0, 1.0001, 0.001));
    try std.testing.expect(!assert.approxEqual(1.0, 2.0, 0.1));

    try std.testing.expect(assert.inRange(i32, 5, 0, 10));
    try std.testing.expect(!assert.inRange(i32, 15, 0, 10));

    try std.testing.expect(assert.isFinite(@as(f32, 1.0)));
    try std.testing.expect(!assert.isFinite(std.math.nan(f32)));

    const sorted = [_]i32{ 1, 2, 3, 4, 5 };
    const unsorted = [_]i32{ 1, 3, 2, 4, 5 };
    try std.testing.expect(assert.isSorted(i32, &sorted));
    try std.testing.expect(!assert.isSorted(i32, &unsorted));
}

// Force-include submodule tests
comptime {
    _ = generators;
    _ = vector_properties;
    _ = database_properties;
    _ = serialization_properties;
    _ = concurrency_properties;
}
