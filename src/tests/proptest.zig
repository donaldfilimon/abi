//! Property-based testing infrastructure for ABI framework.
//!
//! Provides generators, shrinkers, and test runners for property-based testing,
//! enabling discovery of edge cases through randomized input generation.

const std = @import("std");
const abi = @import("abi");
const time = abi.shared.time;

/// Configuration for property-based tests.
pub const PropTestConfig = struct {
    /// Number of test cases to generate
    num_tests: usize = 100,
    /// Maximum number of shrink iterations
    max_shrink_iterations: usize = 1000,
    /// Seed for random number generator (null for random seed)
    seed: ?u64 = null,
    /// Whether to print progress
    verbose: bool = false,
    /// Maximum size hint for generators
    max_size: usize = 100,
};

/// Result of a property test run.
pub const PropTestResult = struct {
    passed: usize,
    failed: usize,
    shrunk_example: ?[]const u8,
    failure_message: ?[]const u8,
    seed_used: u64,

    pub fn success(passed: usize, seed: u64) PropTestResult {
        return .{
            .passed = passed,
            .failed = 0,
            .shrunk_example = null,
            .failure_message = null,
            .seed_used = seed,
        };
    }

    pub fn failure(passed: usize, example: ?[]const u8, message: ?[]const u8, seed: u64) PropTestResult {
        return .{
            .passed = passed,
            .failed = 1,
            .shrunk_example = example,
            .failure_message = message,
            .seed_used = seed,
        };
    }
};

/// Generator interface for producing random values.
pub fn Generator(comptime T: type) type {
    return struct {
        const Self = @This();

        generateFn: *const fn (*std.Random.DefaultPrng, usize) T,
        shrinkFn: ?*const fn (T, std.mem.Allocator) []T,

        pub fn generate(self: Self, prng: *std.Random.DefaultPrng, size: usize) T {
            return self.generateFn(prng, size);
        }

        pub fn shrink(self: Self, value: T, allocator: std.mem.Allocator) []T {
            if (self.shrinkFn) |f| {
                return f(value, allocator);
            }
            return &.{};
        }
    };
}

/// Built-in generators for common types.
pub const Generators = struct {
    /// Generate random integers in range [min, max].
    pub fn intRange(comptime T: type, min: T, max: T) Generator(T) {
        const GenState = struct {
            var min_val: T = undefined;
            var max_val: T = undefined;

            fn generate(prng: *std.Random.DefaultPrng, _: usize) T {
                return prng.random().intRangeAtMost(T, min_val, max_val);
            }

            fn shrink(value: T, allocator: std.mem.Allocator) []T {
                if (value == min_val) return &.{};

                var candidates = std.ArrayListUnmanaged(T){};
                candidates.append(allocator, min_val) catch return &.{};

                const mid = @divFloor(value - min_val, 2) + min_val;
                if (mid != min_val and mid != value) {
                    candidates.append(allocator, mid) catch {};
                }

                if (value > min_val) {
                    candidates.append(allocator, value - 1) catch {};
                }

                return candidates.toOwnedSlice(allocator) catch &.{};
            }
        };

        GenState.min_val = min;
        GenState.max_val = max;

        return .{
            .generateFn = GenState.generate,
            .shrinkFn = GenState.shrink,
        };
    }

    /// Generate random booleans.
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

    /// Generate random floating point numbers in [0, 1].
    pub fn float01() Generator(f64) {
        const GenFn = struct {
            fn generate(prng: *std.Random.DefaultPrng, _: usize) f64 {
                return prng.random().float(f64);
            }
        };

        return .{
            .generateFn = GenFn.generate,
            .shrinkFn = null,
        };
    }

    /// Generate random bytes of variable length.
    pub fn bytes(allocator: std.mem.Allocator) Generator([]u8) {
        _ = allocator;
        const GenFn = struct {
            fn generate(prng: *std.Random.DefaultPrng, size: usize) []u8 {
                const len = prng.random().intRangeAtMost(usize, 0, size);
                const data = std.heap.page_allocator.alloc(u8, len) catch return &.{};
                prng.random().bytes(data);
                return data;
            }
        };

        return .{
            .generateFn = GenFn.generate,
            .shrinkFn = null,
        };
    }

    /// Generate random f32 vectors of fixed dimension.
    pub fn vector(comptime dim: usize) Generator([dim]f32) {
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

    /// Generate one of the provided values.
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
};

/// Property test runner.
pub fn PropTest(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        config: PropTestConfig,
        generator: Generator(T),
        prng: std.Random.DefaultPrng,

        pub fn init(allocator: std.mem.Allocator, generator: Generator(T), config: PropTestConfig) Self {
            const seed = config.seed orelse @as(u64, @intCast(time.unixMilliseconds()));
            return .{
                .allocator = allocator,
                .config = config,
                .generator = generator,
                .prng = std.Random.DefaultPrng.init(seed),
            };
        }

        /// Run the property test with the given predicate.
        pub fn run(self: *Self, predicate: *const fn (T) bool) PropTestResult {
            const seed = self.prng.s;

            var passed: usize = 0;
            var failed_value: ?T = null;

            var i: usize = 0;
            while (i < self.config.num_tests) : (i += 1) {
                const size = @min(i + 1, self.config.max_size);
                const value = self.generator.generate(&self.prng, size);

                if (!predicate(value)) {
                    failed_value = value;
                    break;
                }

                passed += 1;

                if (self.config.verbose and (passed % 10 == 0)) {
                    std.debug.print("PropTest: {d}/{d} passed\n", .{ passed, self.config.num_tests });
                }
            }

            if (failed_value == null) {
                return PropTestResult.success(passed, seed);
            }

            // Attempt to shrink the failing case
            const shrunk = self.shrinkFailure(failed_value.?, predicate);

            return PropTestResult.failure(
                passed,
                shrunk,
                "Property failed",
                seed,
            );
        }

        fn shrinkFailure(self: *Self, value: T, predicate: *const fn (T) bool) ?[]const u8 {
            var current = value;
            var iterations: usize = 0;

            while (iterations < self.config.max_shrink_iterations) : (iterations += 1) {
                const candidates = self.generator.shrink(current, self.allocator);
                defer self.allocator.free(candidates);

                var found_smaller = false;
                for (candidates) |candidate| {
                    if (!predicate(candidate)) {
                        current = candidate;
                        found_smaller = true;
                        break;
                    }
                }

                if (!found_smaller) break;
            }

            // Format the shrunk value
            return std.fmt.allocPrint(self.allocator, "{any}", .{current}) catch null;
        }
    };
}

/// Run a property test with a simple predicate.
pub fn forAll(
    comptime T: type,
    allocator: std.mem.Allocator,
    generator: Generator(T),
    predicate: *const fn (T) bool,
    config: PropTestConfig,
) PropTestResult {
    var test_runner = PropTest(T).init(allocator, generator, config);
    return test_runner.run(predicate);
}

/// Assertion helpers for property tests.
pub const Assertions = struct {
    /// Assert that two values are equal.
    pub fn assertEqual(comptime T: type, actual: T, expected: T) bool {
        return actual == expected;
    }

    /// Assert that a value is within epsilon of expected.
    pub fn assertApproxEqual(actual: f64, expected: f64, epsilon: f64) bool {
        return @abs(actual - expected) < epsilon;
    }

    /// Assert that a slice is sorted.
    pub fn assertSorted(comptime T: type, slice: []const T, comptime lessThan: fn (T, T) bool) bool {
        if (slice.len < 2) return true;
        for (slice[0 .. slice.len - 1], slice[1..]) |a, b| {
            if (lessThan(b, a)) return false;
        }
        return true;
    }

    /// Assert that all elements satisfy a predicate.
    pub fn assertAll(comptime T: type, slice: []const T, predicate: fn (T) bool) bool {
        for (slice) |item| {
            if (!predicate(item)) return false;
        }
        return true;
    }

    /// Assert that at least one element satisfies a predicate.
    pub fn assertAny(comptime T: type, slice: []const T, predicate: fn (T) bool) bool {
        for (slice) |item| {
            if (predicate(item)) return true;
        }
        return false;
    }
};

/// Fuzzing support for finding edge cases.
pub const Fuzzer = struct {
    allocator: std.mem.Allocator,
    prng: std.Random.DefaultPrng,
    corpus: std.ArrayListUnmanaged([]const u8),
    max_corpus_size: usize,

    pub fn init(allocator: std.mem.Allocator, seed: u64) Fuzzer {
        return .{
            .allocator = allocator,
            .prng = std.Random.DefaultPrng.init(seed),
            .corpus = std.ArrayListUnmanaged([]const u8){},
            .max_corpus_size = 1000,
        };
    }

    pub fn deinit(self: *Fuzzer) void {
        for (self.corpus.items) |item| {
            self.allocator.free(item);
        }
        self.corpus.deinit(self.allocator);
        self.* = undefined;
    }

    /// Add input to the corpus if it's interesting.
    pub fn addToCorpus(self: *Fuzzer, input: []const u8) !void {
        if (self.corpus.items.len >= self.max_corpus_size) {
            // Remove a random element
            const idx = self.prng.random().intRangeLessThan(usize, 0, self.corpus.items.len);
            self.allocator.free(self.corpus.swapRemove(idx));
        }

        try self.corpus.append(self.allocator, try self.allocator.dupe(u8, input));
    }

    /// Generate a new fuzzed input based on corpus.
    pub fn generate(self: *Fuzzer, max_len: usize) ![]u8 {
        if (self.corpus.items.len == 0 or self.prng.random().boolean()) {
            // Generate random input
            const len = self.prng.random().intRangeAtMost(usize, 1, max_len);
            const data = try self.allocator.alloc(u8, len);
            self.prng.random().bytes(data);
            return data;
        }

        // Mutate an existing corpus entry
        const base_idx = self.prng.random().intRangeLessThan(usize, 0, self.corpus.items.len);
        const base = self.corpus.items[base_idx];

        return try self.mutate(base, max_len);
    }

    fn mutate(self: *Fuzzer, input: []const u8, max_len: usize) ![]u8 {
        const mutation_type = self.prng.random().intRangeLessThan(u8, 0, 5);

        var result = try self.allocator.dupe(u8, input);
        errdefer self.allocator.free(result);

        switch (mutation_type) {
            0 => {
                // Bit flip
                if (result.len > 0) {
                    const pos = self.prng.random().intRangeLessThan(usize, 0, result.len);
                    const bit = @as(u8, 1) << @as(u3, @intCast(self.prng.random().intRangeLessThan(u8, 0, 8)));
                    result[pos] ^= bit;
                }
            },
            1 => {
                // Byte replacement
                if (result.len > 0) {
                    const pos = self.prng.random().intRangeLessThan(usize, 0, result.len);
                    result[pos] = self.prng.random().int(u8);
                }
            },
            2 => {
                // Insert byte
                if (result.len < max_len) {
                    const new_result = try self.allocator.alloc(u8, result.len + 1);
                    const pos = self.prng.random().intRangeAtMost(usize, 0, result.len);
                    @memcpy(new_result[0..pos], result[0..pos]);
                    new_result[pos] = self.prng.random().int(u8);
                    @memcpy(new_result[pos + 1 ..], result[pos..]);
                    self.allocator.free(result);
                    result = new_result;
                }
            },
            3 => {
                // Delete byte
                if (result.len > 1) {
                    const pos = self.prng.random().intRangeLessThan(usize, 0, result.len);
                    const new_result = try self.allocator.alloc(u8, result.len - 1);
                    @memcpy(new_result[0..pos], result[0..pos]);
                    @memcpy(new_result[pos..], result[pos + 1 ..]);
                    self.allocator.free(result);
                    result = new_result;
                }
            },
            4 => {
                // Swap bytes
                if (result.len >= 2) {
                    const pos1 = self.prng.random().intRangeLessThan(usize, 0, result.len);
                    const pos2 = self.prng.random().intRangeLessThan(usize, 0, result.len);
                    const tmp = result[pos1];
                    result[pos1] = result[pos2];
                    result[pos2] = tmp;
                }
            },
            else => {},
        }

        return result;
    }

    /// Run a fuzz test.
    pub fn fuzz(
        self: *Fuzzer,
        target: *const fn ([]const u8) bool,
        iterations: usize,
        max_len: usize,
    ) !?[]const u8 {
        var i: usize = 0;
        while (i < iterations) : (i += 1) {
            const input = try self.generate(max_len);
            defer self.allocator.free(input);

            if (!target(input)) {
                return try self.allocator.dupe(u8, input);
            }

            // Add interesting inputs to corpus
            if (self.prng.random().intRangeLessThan(u8, 0, 10) == 0) {
                try self.addToCorpus(input);
            }
        }

        return null;
    }
};

// Tests for property-based testing infrastructure

test "generator int range" {
    const gen = Generators.intRange(i32, 0, 100);
    var prng = std.Random.DefaultPrng.init(42);

    var i: usize = 0;
    while (i < 100) : (i += 1) {
        const value = gen.generate(&prng, 50);
        try std.testing.expect(value >= 0 and value <= 100);
    }
}

test "generator boolean" {
    const gen = Generators.boolean();
    var prng = std.Random.DefaultPrng.init(42);

    var true_count: usize = 0;
    var false_count: usize = 0;

    var i: usize = 0;
    while (i < 100) : (i += 1) {
        if (gen.generate(&prng, 0)) {
            true_count += 1;
        } else {
            false_count += 1;
        }
    }

    // Both should have some values
    try std.testing.expect(true_count > 0);
    try std.testing.expect(false_count > 0);
}

test "property test basic" {
    const gen = Generators.intRange(i32, 0, 1000);

    const isPositive = struct {
        fn check(x: i32) bool {
            return x >= 0;
        }
    }.check;

    const result = forAll(i32, std.testing.allocator, gen, isPositive, .{ .num_tests = 50 });

    try std.testing.expectEqual(@as(usize, 50), result.passed);
    try std.testing.expectEqual(@as(usize, 0), result.failed);
}

test "property test failure detection" {
    const gen = Generators.intRange(i32, -100, 100);

    const isPositive = struct {
        fn check(x: i32) bool {
            return x >= 0;
        }
    }.check;

    const result = forAll(i32, std.testing.allocator, gen, isPositive, .{ .num_tests = 1000 });

    // Should detect negative numbers
    try std.testing.expect(result.failed > 0 or result.passed < 1000);
}

test "fuzzer initialization" {
    var fuzzer = Fuzzer.init(std.testing.allocator, 42);
    defer fuzzer.deinit();

    try fuzzer.addToCorpus("test input");
    try std.testing.expectEqual(@as(usize, 1), fuzzer.corpus.items.len);
}

test "fuzzer mutation" {
    var fuzzer = Fuzzer.init(std.testing.allocator, 42);
    defer fuzzer.deinit();

    try fuzzer.addToCorpus("hello");

    const mutated = try fuzzer.generate(100);
    defer std.testing.allocator.free(mutated);

    try std.testing.expect(mutated.len > 0);
}

test "assertions sorted" {
    const sorted = [_]i32{ 1, 2, 3, 4, 5 };
    const unsorted = [_]i32{ 1, 3, 2, 4, 5 };

    const lessThan = struct {
        fn lt(a: i32, b: i32) bool {
            return a < b;
        }
    }.lt;

    try std.testing.expect(Assertions.assertSorted(i32, &sorted, lessThan));
    try std.testing.expect(!Assertions.assertSorted(i32, &unsorted, lessThan));
}

