//! Property-based testing infrastructure for ABI framework.
//!
//! Provides generators, shrinkers, and test runners for property-based testing,
//! enabling discovery of edge cases through randomized input generation.
//!
//! ## Memory Management
//!
//! Generators that produce heap-allocated data (e.g., `bytes`, `asciiString`,
//! `vectorBatch`) use `page_allocator` for simplicity in test contexts. This
//! memory is intentionally not freed during test runs to avoid complexity in
//! generator composition. For long-running fuzz tests, consider using the
//! `Fuzzer` type which properly manages its corpus memory.
//!
//! ## Thread Safety
//!
//! Generators use static state for configuration capture (a limitation of Zig's
//! comptime constraints). This means generators should not be used concurrently
//! across threads. Each test should create its own generator instance.

const std = @import("std");
const abi = @import("abi");
const time = abi.services.shared.time;

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

                var candidates = std.ArrayListUnmanaged(T).empty;
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
    /// Note: Allocates memory that is not freed. See module docs for details.
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

    /// Generate optional values (null or some value).
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

    /// Generate ASCII strings of variable length.
    /// Note: Allocates memory that is not freed. See module docs for details.
    pub fn asciiString(max_len: usize) Generator([]const u8) {
        const GenState = struct {
            var max_length: usize = undefined;

            fn generate(prng: *std.Random.DefaultPrng, size: usize) []const u8 {
                const len = prng.random().intRangeAtMost(usize, 0, @min(size, max_length));
                const data = std.heap.page_allocator.alloc(u8, len) catch return "";
                for (data) |*c| {
                    // Printable ASCII range
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

    /// Generate normalized unit vectors of fixed dimension.
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
                if (mag > 0.0001) {
                    for (&result) |*v| {
                        v.* /= mag;
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

    /// Generate positive floats for distances/weights.
    pub fn positiveFloat() Generator(f64) {
        const GenFn = struct {
            fn generate(prng: *std.Random.DefaultPrng, size: usize) f64 {
                const scale = @as(f64, @floatFromInt(@min(size, 1000)));
                return prng.random().float(f64) * scale;
            }
        };

        return .{
            .generateFn = GenFn.generate,
            .shrinkFn = null,
        };
    }

    /// Generate enum values.
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

    /// Generate pairs of values.
    pub fn pair(comptime A: type, comptime B: type, gen_a: Generator(A), gen_b: Generator(B)) Generator(struct { a: A, b: B }) {
        const Pair = struct { a: A, b: B };
        const GenState = struct {
            var ga: Generator(A) = undefined;
            var gb: Generator(B) = undefined;

            fn generate(prng: *std.Random.DefaultPrng, size: usize) Pair {
                return .{
                    .a = ga.generate(prng, size),
                    .b = gb.generate(prng, size),
                };
            }
        };

        GenState.ga = gen_a;
        GenState.gb = gen_b;

        return .{
            .generateFn = GenState.generate,
            .shrinkFn = null,
        };
    }
};

/// Domain-specific generators for database testing.
/// Note: These generators allocate memory that is not freed. See module docs.
pub const DatabaseGenerators = struct {
    /// Generate a batch of vectors for HNSW testing.
    /// Note: Allocates memory that is not freed. See module docs for details.
    pub fn vectorBatch(comptime dim: usize, max_count: usize) Generator(struct { vectors: [][dim]f32, ids: []u64 }) {
        const Batch = struct { vectors: [][dim]f32, ids: []u64 };
        const GenState = struct {
            var max_c: usize = undefined;

            fn generate(prng: *std.Random.DefaultPrng, size: usize) Batch {
                const count = prng.random().intRangeAtMost(usize, 1, @min(size + 1, max_c));

                const vectors = std.heap.page_allocator.alloc([dim]f32, count) catch return .{ .vectors = &.{}, .ids = &.{} };
                const ids = std.heap.page_allocator.alloc(u64, count) catch return .{ .vectors = &.{}, .ids = &.{} };

                for (vectors, 0..) |*v, i| {
                    var sum_sq: f32 = 0.0;
                    for (v) |*val| {
                        val.* = prng.random().float(f32) * 2.0 - 1.0;
                        sum_sq += val.* * val.*;
                    }
                    // Normalize
                    const mag = @sqrt(sum_sq);
                    if (mag > 0.0001) {
                        for (v) |*val| {
                            val.* /= mag;
                        }
                    }
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

    /// Generate search parameters (k, ef).
    pub fn searchParams() Generator(struct { k: u32, ef: u32 }) {
        const Params = struct { k: u32, ef: u32 };
        const GenFn = struct {
            fn generate(prng: *std.Random.DefaultPrng, size: usize) Params {
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
};

/// Stateful property testing for databases and state machines.
pub fn StatefulTest(comptime State: type, comptime Command: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        prng: std.Random.DefaultPrng,
        state: State,
        command_history: std.ArrayListUnmanaged(Command),
        max_commands: usize,

        pub fn init(allocator: std.mem.Allocator, initial_state: State, seed: u64) Self {
            return .{
                .allocator = allocator,
                .prng = std.Random.DefaultPrng.init(seed),
                .state = initial_state,
                .command_history = .{},
                .max_commands = 100,
            };
        }

        pub fn deinit(self: *Self) void {
            self.command_history.deinit(self.allocator);
            self.* = undefined;
        }

        /// Run stateful test with command generator and execution.
        pub fn run(
            self: *Self,
            num_runs: usize,
            generate_command: *const fn (*std.Random.DefaultPrng, *const State) Command,
            execute_command: *const fn (*State, Command) bool,
            invariant: *const fn (*const State) bool,
        ) StatefulTestResult {
            var run_idx: usize = 0;
            while (run_idx < num_runs) : (run_idx += 1) {
                self.command_history.clearRetainingCapacity();

                var cmd_idx: usize = 0;
                while (cmd_idx < self.max_commands) : (cmd_idx += 1) {
                    const cmd = generate_command(&self.prng, &self.state);
                    self.command_history.append(self.allocator, cmd) catch break;

                    const success = execute_command(&self.state, cmd);
                    if (!success) {
                        return .{
                            .success = false,
                            .runs_completed = run_idx,
                            .commands_executed = cmd_idx + 1,
                            .failure_reason = "Command execution failed",
                        };
                    }

                    if (!invariant(&self.state)) {
                        return .{
                            .success = false,
                            .runs_completed = run_idx,
                            .commands_executed = cmd_idx + 1,
                            .failure_reason = "Invariant violated",
                        };
                    }
                }
            }

            return .{
                .success = true,
                .runs_completed = num_runs,
                .commands_executed = num_runs * self.max_commands,
                .failure_reason = null,
            };
        }
    };
}

pub const StatefulTestResult = struct {
    success: bool,
    runs_completed: usize,
    commands_executed: usize,
    failure_reason: ?[]const u8,
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

    /// Assert that haystack contains needle.
    pub fn assertContains(haystack: []const u8, needle: []const u8) bool {
        return std.mem.indexOf(u8, haystack, needle) != null;
    }

    /// Assert that a < b.
    pub fn assertLessThan(comptime T: type, a: T, b: T) bool {
        return a < b;
    }

    /// Assert that a <= b.
    pub fn assertLessThanOrEqual(comptime T: type, a: T, b: T) bool {
        return a <= b;
    }

    /// Assert that a > b.
    pub fn assertGreaterThan(comptime T: type, a: T, b: T) bool {
        return a > b;
    }

    /// Assert that a >= b.
    pub fn assertGreaterThanOrEqual(comptime T: type, a: T, b: T) bool {
        return a >= b;
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
            .corpus = std.ArrayListUnmanaged([]const u8).empty,
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

test "generator unit vector normalization" {
    const gen = Generators.unitVector(4);
    var prng = std.Random.DefaultPrng.init(42);

    var i: usize = 0;
    while (i < 20) : (i += 1) {
        const vec = gen.generate(&prng, 50);

        // Check magnitude is approximately 1.0
        var sum_sq: f32 = 0.0;
        for (vec) |v| {
            sum_sq += v * v;
        }
        const mag = @sqrt(sum_sq);
        try std.testing.expect(mag > 0.99 and mag < 1.01);
    }
}

test "generator positive float" {
    const gen = Generators.positiveFloat();
    var prng = std.Random.DefaultPrng.init(42);

    var i: usize = 0;
    while (i < 100) : (i += 1) {
        const value = gen.generate(&prng, 100);
        try std.testing.expect(value >= 0.0);
    }
}

test "database generators search params" {
    const gen = DatabaseGenerators.searchParams();
    var prng = std.Random.DefaultPrng.init(42);

    var i: usize = 0;
    while (i < 50) : (i += 1) {
        const params = gen.generate(&prng, 50);
        // k should be >= 1
        try std.testing.expect(params.k >= 1);
        // ef should be >= k
        try std.testing.expect(params.ef >= params.k);
    }
}

test "stateful test basic" {
    const State = struct {
        counter: i32,
    };

    const Command = enum {
        increment,
        decrement,
    };

    var test_runner = StatefulTest(State, Command).init(
        std.testing.allocator,
        .{ .counter = 0 },
        42,
    );
    defer test_runner.deinit();

    const genCmd = struct {
        fn gen(prng: *std.Random.DefaultPrng, _: *const State) Command {
            return if (prng.random().boolean()) .increment else .decrement;
        }
    }.gen;

    const execCmd = struct {
        fn exec(state: *State, cmd: Command) bool {
            switch (cmd) {
                .increment => state.counter += 1,
                .decrement => state.counter -= 1,
            }
            return true;
        }
    }.exec;

    const invariant = struct {
        fn check(state: *const State) bool {
            // Counter should stay within reasonable bounds
            return state.counter >= -200 and state.counter <= 200;
        }
    }.check;

    test_runner.max_commands = 10;
    const result = test_runner.run(5, genCmd, execCmd, invariant);
    try std.testing.expect(result.success);
}
