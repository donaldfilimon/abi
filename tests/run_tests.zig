//! Test runner for WDBX-AI

const std = @import("std");
const test_framework = @import("test_framework.zig");

// Import all test modules
const core_tests = @import("core_tests.zig");
const database_tests = @import("database_tests.zig");
const integration_tests = @import("integration_tests.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Parse command line arguments
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    
    var runner = test_framework.TestRunner.init(allocator);
    
    // Process arguments
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--filter")) {
            i += 1;
            if (i < args.len) {
                runner.filter = args[i];
            }
        } else if (std.mem.eql(u8, args[i], "--verbose") or std.mem.eql(u8, args[i], "-v")) {
            runner.verbose = true;
        } else if (std.mem.eql(u8, args[i], "--help") or std.mem.eql(u8, args[i], "-h")) {
            try printHelp();
            return;
        }
    }
    
    // Run tests
    try runner.run();
}

fn printHelp() !void {
    const stdout = std.io.getStdOut().writer();
    try stdout.writeAll(
        \\WDBX-AI Test Runner
        \\
        \\Usage: run_tests [options]
        \\
        \\Options:
        \\  --filter <pattern>  Run only tests matching pattern
        \\  --verbose, -v       Show detailed output
        \\  --help, -h          Show this help message
        \\
        \\Examples:
        \\  run_tests                    # Run all tests
        \\  run_tests --filter core      # Run only core tests
        \\  run_tests --verbose          # Run with detailed output
        \\
    );
}

test {
    // This references all test modules to ensure they're included
    _ = core_tests;
    _ = database_tests;
    _ = integration_tests;
    _ = test_framework;
}