//! Property-based testing framework
//!
//! Provides utilities for testing properties of functions and data structures
//! using randomized inputs and invariants.

const std = @import("std");

pub const PropertyTestConfig = struct {
    max_cases: usize = 100,
    max_size: usize = 100,
    seed: ?u64 = null,
};

pub const PropertyTest = struct {
    name: []const u8,
    config: PropertyTestConfig,
    passed: usize = 0,
    failed: usize = 0,
    cases: std.ArrayListUnmanaged(TestCase),

    const TestCase = struct {
        inputs: []const u8,
        expected: []const u8,
        passed: bool,
    };

    pub fn init(allocator: std.mem.Allocator, name: []const u8, config: PropertyTestConfig) !PropertyTest {
        return .{
            .name = try allocator.dupe(u8, name),
            .config = config,
            .cases = std.ArrayListUnmanaged(TestCase).empty,
        };
    }

    pub fn deinit(self: *PropertyTest, allocator: std.mem.Allocator) void {
        for (self.cases.items) |*case| {
            allocator.free(case.inputs);
            allocator.free(case.expected);
        }
        self.cases.deinit(allocator);
        allocator.free(self.name);
        self.* = undefined;
    }

    pub fn addCase(self: *PropertyTest, allocator: std.mem.Allocator, inputs: []const u8, expected: []const u8) !void {
        const inputs_copy = try allocator.dupe(u8, inputs);
        errdefer allocator.free(inputs_copy);

        const expected_copy = try allocator.dupe(u8, expected);
        errdefer allocator.free(expected_copy);

        try self.cases.append(allocator, .{
            .inputs = inputs_copy,
            .expected = expected_copy,
            .passed = false,
        });
    }

    pub fn run(self: *PropertyTest, allocator: std.mem.Allocator, property: anytype) !void {
        self.passed = 0;
        self.failed = 0;

        for (self.cases.items, 0..) |*case, i| {
            if (property(allocator, case.inputs, case.expected)) {
                case.passed = true;
                self.passed += 1;
            } else {
                self.failed += 1;
                std.debug.print(
                    "Property test '{s}' failed for case {d}: inputs={s}, expected={s}\n",
                    .{ self.name, i, case.inputs, case.expected },
                );
            }
        }
    }

    pub fn summary(self: *const PropertyTest) void {
        const total = self.passed + self.failed;
        const pass_rate: f64 = if (total > 0)
            @as(f64, @floatFromInt(self.passed)) / @as(f64, @floatFromInt(total)) * 100.0
        else
            0.0;

        std.debug.print(
            "Property test '{s}': {d}/{d} passed ({d:.1}%)\n",
            .{ self.name, self.passed, total, pass_rate },
        );

        if (self.failed > 0) {
            std.debug.print("  FAILED: {d} cases\n", .{self.failed});
        }
    }
};

pub fn checkProperty(
    allocator: std.mem.Allocator,
    comptime PropertyFn: type,
    config: PropertyTestConfig,
    name: []const u8,
) !void {
    var prop_test = try PropertyTest.init(allocator, name, config);
    defer prop_test.deinit(allocator);

    var rng = if (config.seed) |seed|
        std.Random.DefaultPrng.init(seed)
    else
        std.Random.DefaultPrng.init(@intCast(std.time.nanoTimestamp()));

    var i: usize = 0;
    while (i < config.max_cases) : (i += 1) {
        const input_size = rng.uintRangeAtMost(usize, config.max_size);

        const input = try allocator.alloc(u8, input_size);
        defer allocator.free(input);

        for (input) |*byte| {
            byte.* = @intCast(rng.uintAtMost(u8, 255));
        }

        try prop_test.run(allocator, PropertyFn);
    }

    prop_test.summary();
}

pub fn assertEq(comptime T: type, a: T, b: T) bool {
    if (T == []const u8) {
        const a_slice = @as([]const u8, @ptrCast(@alignCast(a)));
        const b_slice = @as([]const u8, @ptrCast(@alignCast(b)));
        return std.mem.eql(u8, a_slice, b_slice);
    }
    return std.meta.eql(a, b);
}

pub fn assertLessThan(comptime T: type, a: T, b: T) bool {
    return a < b;
}

pub fn assertLessThanOrEqual(comptime T: type, a: T, b: T) bool {
    return a <= b;
}

pub fn assertGreaterThan(comptime T: type, a: T, b: T) bool {
    return a > b;
}

pub fn assertGreaterThanOrEqual(comptime T: type, a: T, b: T) bool {
    return a >= b;
}

pub fn assertContains(haystack: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, haystack, needle) != null;
}

pub fn assertLength(comptime T: type, value: T, expected_len: usize) bool {
    const len = switch (@typeInfo(T)) {
        .Pointer => |info| if (info.size == 0) value.len else 1,
        else => 1,
    };
    return len == expected_len;
}

test "property test framework" {
    const allocator = std.testing.allocator;

    var prop_test = try PropertyTest.init(allocator, "test_property", .{
        .max_cases = 10,
        .max_size = 10,
        .seed = 12345,
    });
    defer prop_test.deinit(allocator);

    try prop_test.addCase(allocator, "hello", "hello");
    try prop_test.addCase(allocator, "world", "world");

    try prop_test.run(allocator, struct {
        fn fn_(allocator2: std.mem.Allocator, inputs: []const u8, expected: []const u8) bool {
            _ = allocator2;
            return std.mem.eql(u8, inputs, expected);
        }
    }.fn_);

    try std.testing.expectEqual(@as(usize, 2), prop_test.passed);
}

test "assertions work correctly" {
    try std.testing.expect(assertEq([]const u8, "test", "test"));
    try std.testing.expect(assertLessThan(usize, 5, 10));
    try std.testing.expect(assertGreaterThan(usize, 10, 5));
    try std.testing.expect(assertContains("hello world", "world"));
    try std.testing.expect(assertLength([]const u8, "test", 4));
}
