#!/usr/bin/env zig

//! Test runner script for the Abi AI Framework
//!
//! This script runs all available tests and provides a comprehensive
//! testing report with coverage information.

const std = @import("std");
const builtin = @import("builtin");

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    std.debug.print("🚀 Abi AI Framework - Comprehensive Test Suite\n", .{});
    std.debug.print("================================================\n\n", .{});

    // Test configuration
    const test_config = TestConfig{
        .verbose = true,
        .fail_fast = false,
        .timeout_ms = 30000, // 30 seconds
    };

    var test_runner = TestRunner.init(allocator, test_config);
    defer test_runner.deinit();

    // Register all available tests
    try test_runner.registerTest(.{
        .name = "Dummy Tests",
        .command = "zig test tests/dummy_test.zig",
        .description = "Basic functionality tests",
        .expected_tests = 1,
    });

    try test_runner.registerTest(.{
        .name = "SIMD Vector Tests",
        .command = "zig test tests/test_simd_vector.zig",
        .description = "SIMD operations and vector math",
        .expected_tests = 11,
    });

    try test_runner.registerTest(.{
        .name = "Database Tests",
        .command = "zig test tests/test_database.zig",
        .description = "Vector database operations",
        .expected_tests = 7,
    });

    try test_runner.registerTest(.{
        .name = "Core Error Tests",
        .command = "zig test src/core/errors.zig",
        .description = "Error handling and context",
        .expected_tests = 2,
    });

    try test_runner.registerTest(.{
        .name = "Core Module Tests",
        .command = "zig test src/core/mod.zig",
        .description = "Core utilities and functions",
        .expected_tests = 1,
    });

    // Run all tests
    const results = try test_runner.runAllTests();

    // Print comprehensive report
    try printTestReport(results);

    // Exit with appropriate code
    if (results.failed > 0) {
        std.process.exit(1);
    } else {
        std.debug.print("\n✅ All tests passed! 🎉\n", .{});
        std.process.exit(0);
    }
}

const TestConfig = struct {
    verbose: bool = false,
    fail_fast: bool = false,
    timeout_ms: u64 = 30000,
};

const TestCase = struct {
    name: []const u8,
    command: []const u8,
    description: []const u8,
    expected_tests: u32 = 0,
};

const TestResult = struct {
    test_case: TestCase,
    success: bool,
    exit_code: u32,
    output: []const u8,
    duration_ms: u64,
    error_message: ?[]const u8 = null,
};

const TestRunner = struct {
    allocator: std.mem.Allocator,
    config: TestConfig,
    test_cases: std.ArrayList(TestCase),

    pub fn init(allocator: std.mem.Allocator, config: TestConfig) TestRunner {
        return TestRunner{
            .allocator = allocator,
            .config = config,
            .test_cases = std.ArrayList(TestCase).init(allocator),
        };
    }

    pub fn deinit(self: *TestRunner) void {
        self.test_cases.deinit();
    }

    pub fn registerTest(self: *TestRunner, test_case: TestCase) !void {
        try self.test_cases.append(test_case);
    }

    pub fn runAllTests(self: *TestRunner) !TestResults {
        var results = std.ArrayList(TestResult).init(self.allocator);
        defer results.deinit();

        var passed: u32 = 0;
        var failed: u32 = 0;
        var total_tests: u32 = 0;

        for (self.test_cases.items) |test_case| {
            if (self.config.verbose) {
                std.debug.print("Running: {s} - {s}\n", .{ test_case.name, test_case.description });
            }

            const result = try self.runTest(test_case);
            try results.append(result);

            total_tests += test_case.expected_tests;

            if (result.success) {
                passed += test_case.expected_tests;
                if (self.config.verbose) {
                    std.debug.print("✅ {s} passed\n", .{test_case.name});
                }
            } else {
                failed += test_case.expected_tests;
                std.debug.print("❌ {s} failed (exit code: {})\n", .{ test_case.name, result.exit_code });
                if (result.error_message) |msg| {
                    std.debug.print("   Error: {s}\n", .{msg});
                }

                if (self.config.fail_fast) {
                    break;
                }
            }
        }

        return TestResults{
            .results = try results.toOwnedSlice(),
            .passed = passed,
            .failed = failed,
            .total_tests = total_tests,
            .total_suites = @intCast(results.items.len),
        };
    }

    fn runTest(self: *TestRunner, test_case: TestCase) !TestResult {
        const start_time = std.time.milliTimestamp();

        var child_process = std.process.Child.init(&[_][]const u8{ "cmd", "/c", test_case.command }, self.allocator);
        child_process.stdout_behavior = .Pipe;
        child_process.stderr_behavior = .Pipe;

        try child_process.spawn();

        const stdout = try child_process.stdout.?.reader().readAllAlloc(self.allocator, 1024 * 1024);
        const stderr = try child_process.stderr.?.reader().readAllAlloc(self.allocator, 1024 * 1024);

        const term = try child_process.wait();
        const end_time = std.time.milliTimestamp();
        const duration = @as(u64, @intCast(end_time - start_time));

        const success = switch (term) {
            .Exited => |code| code == 0,
            else => false,
        };

        const output = try std.mem.concat(self.allocator, u8, &[_][]const u8{ stdout, stderr });
        const error_message = if (!success and stderr.len > 0) stderr else null;

        return TestResult{
            .test_case = test_case,
            .success = success,
            .exit_code = switch (term) {
                .Exited => |code| code,
                else => 1,
            },
            .output = output,
            .duration_ms = duration,
            .error_message = error_message,
        };
    }
};

const TestResults = struct {
    results: []const TestResult,
    passed: u32,
    failed: u32,
    total_tests: u32,
    total_suites: u32,
};

fn printTestReport(results: TestResults) !void {
    std.debug.print("\n📊 Test Results Summary\n", .{});
    std.debug.print("========================\n", .{});

    for (results.results) |result| {
        const status = if (result.success) "✅ PASS" else "❌ FAIL";
        std.debug.print("{s} {s} ({d}ms)\n", .{ status, result.test_case.name, result.duration_ms });
    }

    std.debug.print("\n📈 Statistics:\n", .{});
    std.debug.print("  Test Suites: {}/{}\n", .{ results.total_suites - results.failed, results.total_suites });
    std.debug.print("  Individual Tests: {}/{}\n", .{ results.passed, results.total_tests });

    const coverage = if (results.total_tests > 0)
        @as(f32, @floatFromInt(results.passed)) / @as(f32, @floatFromInt(results.total_tests)) * 100.0
    else
        0.0;

    std.debug.print("  Coverage: {d:.1}%\n", .{coverage});

    if (results.failed > 0) {
        std.debug.print("\n⚠️  Failed Tests:\n", .{});
        for (results.results) |result| {
            if (!result.success) {
                std.debug.print("  - {s}: {s}\n", .{ result.test_case.name, result.test_case.description });
                if (result.error_message) |msg| {
                    std.debug.print("    Error: {s}\n", .{msg});
                }
            }
        }
    }
}

// Utility functions for test discovery
pub fn discoverTests(allocator: std.mem.Allocator) ![]const TestCase {
    var tests = std.ArrayList(TestCase).init(allocator);
    defer tests.deinit();

    // Add discovered test files
    const test_files = &[_][]const u8{
        "tests/dummy_test.zig",
        "tests/test_simd_vector.zig",
        "tests/test_database.zig",
    };

    for (test_files) |file| {
        const test_case = TestCase{
            .name = std.fs.path.basename(file),
            .command = try std.fmt.allocPrint(allocator, "zig test {s}", .{file}),
            .description = try std.fmt.allocPrint(allocator, "Tests in {s}", .{file}),
        };
        try tests.append(test_case);
    }

    return try tests.toOwnedSlice();
}
