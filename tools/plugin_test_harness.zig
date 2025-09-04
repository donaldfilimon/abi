//! Plugin Test Harness
//!
//! This tool provides comprehensive testing for Abi AI Framework plugins

const std = @import("std");
const plugin_interface = @import("../src/plugins/interface.zig");
const plugin_types = @import("../src/plugins/types.zig");
const plugin_loader = @import("../src/plugins/loader.zig");

const PluginInterface = plugin_interface.PluginInterface;
const Plugin = plugin_interface.Plugin;

const TestCase = struct {
    name: []const u8,
    description: []const u8,
    test_fn: *const fn (plugin: *Plugin, allocator: std.mem.Allocator) anyerror!void,
    required_capabilities: []const []const u8 = &.{},
};

const TestResult = struct {
    test_name: []const u8,
    passed: bool,
    duration_ns: u64,
    error_message: ?[]const u8 = null,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        try printUsage();
        return;
    }

    const plugin_path = args[1];
    var test_filter: ?[]const u8 = null;
    var stress_test = false;
    var benchmark = false;

    // Parse command line options
    var i: usize = 2;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--filter")) {
            if (i + 1 < args.len) {
                test_filter = args[i + 1];
                i += 1;
            }
        } else if (std.mem.eql(u8, args[i], "--stress")) {
            stress_test = true;
        } else if (std.mem.eql(u8, args[i], "--benchmark")) {
            benchmark = true;
        }
    }

    try runTestHarness(allocator, plugin_path, test_filter, stress_test, benchmark);
}

fn printUsage() !void {
    const stdout = std.io.getStdOut().writer();
    try stdout.print(
        \\Plugin Test Harness - Comprehensive plugin testing
        \\
        \\Usage: plugin_test_harness <plugin_path> [options]
        \\
        \\Options:
        \\  --filter <pattern>    Run only tests matching pattern
        \\  --stress             Run stress tests
        \\  --benchmark          Run performance benchmarks
        \\
        \\Example:
        \\  plugin_test_harness ./plugins/example_plugin.so --filter lifecycle
        \\  plugin_test_harness ./plugins/ai_plugin.so --stress --benchmark
        \\
    , .{});
}

fn runTestHarness(
    allocator: std.mem.Allocator,
    plugin_path: []const u8,
    filter: ?[]const u8,
    stress_test: bool,
    benchmark: bool,
) !void {
    const stdout = std.io.getStdOut().writer();

    try stdout.print("🧪 Plugin Test Harness\n", .{});
    try stdout.print("📦 Testing plugin: {s}\n\n", .{plugin_path});

    // Load the plugin
    var loader = try plugin_loader.PluginLoader.init(allocator);
    defer loader.deinit();

    const plugin_handle = try loader.loadPlugin(plugin_path);
    var plugin = try loader.getPlugin(plugin_handle);

    // Get plugin info
    const info = plugin.getInfo();
    try stdout.print("Plugin: {s} v{}\n", .{ info.name, info.version });
    try stdout.print("Type: {s}\n", .{info.plugin_type.toString()});
    try stdout.print("Author: {s}\n\n", .{info.author});

    // Initialize plugin
    var config = plugin_types.PluginConfig.init(allocator);
    defer config.deinit();

    try plugin.initialize(&config);
    defer plugin.deinit();

    // Define test cases
    const test_cases = [_]TestCase{
        .{
            .name = "lifecycle",
            .description = "Test plugin lifecycle transitions",
            .test_fn = testLifecycle,
        },
        .{
            .name = "configuration",
            .description = "Test configuration management",
            .test_fn = testConfiguration,
        },
        .{
            .name = "error_handling",
            .description = "Test error handling and recovery",
            .test_fn = testErrorHandling,
        },
        .{
            .name = "events",
            .description = "Test event handling",
            .test_fn = testEventHandling,
        },
        .{
            .name = "memory",
            .description = "Test memory management",
            .test_fn = testMemoryManagement,
        },
        .{
            .name = "api",
            .description = "Test extended API access",
            .test_fn = testExtendedApi,
        },
    };

    var results = std.ArrayList(TestResult).init(allocator);
    defer results.deinit();

    // Run test cases
    try stdout.print("Running tests...\n\n", .{});

    for (test_cases) |test_case| {
        if (filter) |f| {
            if (std.mem.indexOf(u8, test_case.name, f) == null) {
                continue;
            }
        }

        const result = try runTestCase(test_case, plugin, allocator);
        try results.append(result);

        const status = if (result.passed) "✅ PASS" else "❌ FAIL";
        try stdout.print("{s} {s}: {s}\n", .{ status, test_case.name, test_case.description });
        
        if (result.error_message) |msg| {
            try stdout.print("   Error: {s}\n", .{msg});
        }
        
        try stdout.print("   Duration: {d:.2}ms\n\n", .{@as(f64, @floatFromInt(result.duration_ns)) / 1_000_000.0});
    }

    // Run stress tests if requested
    if (stress_test) {
        try stdout.print("\n🔥 Running stress tests...\n\n", .{});
        try runStressTests(plugin, allocator);
    }

    // Run benchmarks if requested
    if (benchmark) {
        try stdout.print("\n📊 Running benchmarks...\n\n", .{});
        try runBenchmarks(plugin, allocator);
    }

    // Print summary
    try printSummary(results.items, stdout);
}

fn runTestCase(test_case: TestCase, plugin: *Plugin, allocator: std.mem.Allocator) !TestResult {
    const start_time = std.time.nanoTimestamp();
    var result = TestResult{
        .test_name = test_case.name,
        .passed = true,
        .duration_ns = 0,
    };

    test_case.test_fn(plugin, allocator) catch |err| {
        result.passed = false;
        result.error_message = try std.fmt.allocPrint(allocator, "{}", .{err});
    };

    result.duration_ns = @intCast(std.time.nanoTimestamp() - start_time);
    return result;
}

// Test implementations

fn testLifecycle(plugin: *Plugin, allocator: std.mem.Allocator) !void {
    _ = allocator;

    // Test state transitions
    try plugin.start();
    try std.testing.expectEqual(plugin_types.PluginState.running, plugin.getState());

    try plugin.pause();
    try std.testing.expectEqual(plugin_types.PluginState.paused, plugin.getState());

    try plugin.resumePlugin();
    try std.testing.expectEqual(plugin_types.PluginState.running, plugin.getState());

    try plugin.stop();
    try std.testing.expectEqual(plugin_types.PluginState.stopped, plugin.getState());
}

fn testConfiguration(plugin: *Plugin, allocator: std.mem.Allocator) !void {
    var config = plugin_types.PluginConfig.init(allocator);
    defer config.deinit();

    try config.setParameter("test_key", "test_value");
    try config.setParameter("batch_size", "256");

    try plugin.configure(&config);

    // Verify configuration was applied
    if (plugin.interface.get_config) |get_config_fn| {
        const current_config = get_config_fn(plugin.context.?);
        try std.testing.expect(current_config != null);
    }
}

fn testErrorHandling(plugin: *Plugin, allocator: std.mem.Allocator) !void {
    _ = allocator;

    // Test invalid state transitions
    if (plugin.getState() == .stopped) {
        // Try to pause when stopped (should fail)
        try std.testing.expectError(plugin_types.PluginError.InvalidParameters, plugin.pause());
    }

    // Test processing with null input (if supported)
    if (plugin.interface.process != null) {
        const result = plugin.process(null, null);
        // Should either succeed or return specific error
        _ = result;
    }
}

fn testEventHandling(plugin: *Plugin, allocator: std.mem.Allocator) !void {
    _ = allocator;

    // Test standard events
    const events = [_]struct { type: u32, name: []const u8 }{
        .{ .type = 1, .name = "system_startup" },
        .{ .type = 2, .name = "system_shutdown" },
        .{ .type = 3, .name = "config_update" },
        .{ .type = 4, .name = "memory_pressure" },
    };

    for (events) |event| {
        try plugin.onEvent(event.type, null);
    }
}

fn testMemoryManagement(plugin: *Plugin, allocator: std.mem.Allocator) !void {
    if (plugin.interface.process == null) return;

    // Allocate test data
    const test_sizes = [_]usize{ 1024, 4096, 16384, 65536 };
    
    for (test_sizes) |size| {
        const input = try allocator.alloc(u8, size);
        defer allocator.free(input);
        
        // Fill with test pattern
        for (input, 0..) |*byte, i| {
            byte.* = @truncate(i);
        }

        var output: []u8 = undefined;
        try plugin.process(&input, &output);
        
        // Ensure output was allocated
        if (output.len > 0) {
            allocator.free(output);
        }
    }
}

fn testExtendedApi(plugin: *Plugin, allocator: std.mem.Allocator) !void {
    _ = allocator;

    const info = plugin.getInfo();
    
    // Test each provided API
    for (info.provides) |api_name| {
        const api_ptr = plugin.getApi(api_name);
        // API might be null if not implemented
        _ = api_ptr;
    }
}

// Stress tests

fn runStressTests(plugin: *Plugin, allocator: std.mem.Allocator) !void {
    const stdout = std.io.getStdOut().writer();

    // Concurrent access test
    try stdout.print("Testing concurrent access...\n", .{});
    try testConcurrentAccess(plugin, allocator);

    // Rapid state changes
    try stdout.print("Testing rapid state changes...\n", .{});
    try testRapidStateChanges(plugin, allocator);

    // Memory pressure
    try stdout.print("Testing under memory pressure...\n", .{});
    try testMemoryPressure(plugin, allocator);
}

fn testConcurrentAccess(plugin: *Plugin, allocator: std.mem.Allocator) !void {
    _ = allocator;

    if (plugin.interface.process == null) return;

    const thread_count = 8;
    var threads: [thread_count]std.Thread = undefined;

    for (&threads) |*thread| {
        thread.* = try std.Thread.spawn(.{}, struct {
            fn run(p: *Plugin) void {
                var input: []const u8 = "test data";
                var output: []u8 = undefined;
                
                for (0..100) |_| {
                    p.process(&input, &output) catch {};
                }
            }
        }.run, .{plugin});
    }

    for (threads) |thread| {
        thread.join();
    }
}

fn testRapidStateChanges(plugin: *Plugin, allocator: std.mem.Allocator) !void {
    _ = allocator;

    for (0..100) |_| {
        try plugin.start();
        try plugin.stop();
    }
}

fn testMemoryPressure(plugin: *Plugin, allocator: std.mem.Allocator) !void {
    if (plugin.interface.process == null) return;

    // Allocate large buffers to simulate memory pressure
    var allocations = std.ArrayList([]u8).init(allocator);
    defer {
        for (allocations.items) |alloc| {
            allocator.free(alloc);
        }
        allocations.deinit();
    }

    // Allocate 80% of available memory (simulated)
    const pressure_size = 100 * 1024 * 1024; // 100MB
    for (0..10) |_| {
        const buffer = try allocator.alloc(u8, pressure_size / 10);
        try allocations.append(buffer);
    }

    // Try to use plugin under pressure
    var input: []const u8 = "test";
    var output: []u8 = undefined;
    
    // Send memory pressure event
    try plugin.onEvent(4, null);
    
    // Should still work
    try plugin.process(&input, &output);
}

// Benchmarks

fn runBenchmarks(plugin: *Plugin, allocator: std.mem.Allocator) !void {
    const stdout = std.io.getStdOut().writer();

    if (plugin.interface.process != null) {
        try stdout.print("Benchmarking process function...\n", .{});
        try benchmarkProcess(plugin, allocator);
    }

    try stdout.print("Benchmarking metrics collection...\n", .{});
    try benchmarkMetrics(plugin, allocator);
}

fn benchmarkProcess(plugin: *Plugin, allocator: std.mem.Allocator) !void {
    const stdout = std.io.getStdOut().writer();
    
    const iterations = 10000;
    const input_sizes = [_]usize{ 128, 1024, 8192, 65536 };

    for (input_sizes) |size| {
        const input = try allocator.alloc(u8, size);
        defer allocator.free(input);
        
        for (input, 0..) |*byte, i| {
            byte.* = @truncate(i);
        }

        const start_time = std.time.nanoTimestamp();
        
        for (0..iterations) |_| {
            var output: []u8 = undefined;
            try plugin.process(&input, &output);
            if (output.len > 0) {
                allocator.free(output);
            }
        }
        
        const duration_ns = std.time.nanoTimestamp() - start_time;
        const ops_per_sec = @as(f64, @floatFromInt(iterations)) * 1_000_000_000.0 / @as(f64, @floatFromInt(duration_ns));
        
        try stdout.print("  Size: {d}B - {d:.2} ops/sec\n", .{ size, ops_per_sec });
    }
}

fn benchmarkMetrics(plugin: *Plugin, allocator: std.mem.Allocator) !void {
    _ = allocator;
    const stdout = std.io.getStdOut().writer();
    
    if (plugin.interface.get_metrics == null) return;

    const iterations = 100000;
    var buffer: [1024]u8 = undefined;
    
    const start_time = std.time.nanoTimestamp();
    
    for (0..iterations) |_| {
        _ = try plugin.getMetrics(&buffer);
    }
    
    const duration_ns = std.time.nanoTimestamp() - start_time;
    const ops_per_sec = @as(f64, @floatFromInt(iterations)) * 1_000_000_000.0 / @as(f64, @floatFromInt(duration_ns));
    
    try stdout.print("  Metrics collection: {d:.2} ops/sec\n", .{ops_per_sec});
}

fn printSummary(results: []const TestResult, writer: anytype) !void {
    var passed: u32 = 0;
    var failed: u32 = 0;
    var total_duration_ns: u64 = 0;

    for (results) |result| {
        if (result.passed) {
            passed += 1;
        } else {
            failed += 1;
        }
        total_duration_ns += result.duration_ns;
    }

    try writer.print("\n📊 Test Summary\n", .{});
    try writer.print("─────────────────────────────\n", .{});
    try writer.print("Total tests: {d}\n", .{results.len});
    try writer.print("Passed: {d}\n", .{passed});
    try writer.print("Failed: {d}\n", .{failed});
    try writer.print("Total duration: {d:.2}ms\n", .{@as(f64, @floatFromInt(total_duration_ns)) / 1_000_000.0});
    
    if (failed == 0) {
        try writer.print("\n✅ All tests passed!\n", .{});
    } else {
        try writer.print("\n❌ Some tests failed.\n", .{});
    }
}