//! Advanced Features Example
//!
//! Demonstrates advanced usage patterns of the ABI framework

const std = @import("std");
const abi = @import("../lib/mod.zig");

pub fn main() !void {
    std.log.info("ABI Framework Advanced Features Example", .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create a custom framework configuration
    var framework = try abi.createFramework(allocator, .{
        .enabled_features = &[_]abi.features.FeatureTag{ .ai, .database, .web, .monitoring },
        .disabled_features = &[_]abi.features.FeatureTag{.gpu},
        .max_plugins = 64,
        .enable_hot_reload = true,
        .enable_profiling = true,
        .memory_limit_mb = 512,
        .log_level = .debug,
    });
    defer framework.deinit();

    // Demonstrate advanced feature management
    std.log.info("Advanced feature management:", .{});

    // Create feature flags programmatically
    const custom_features = [_]abi.features.FeatureTag{ .ai, .database, .web };
    const feature_flags = abi.features.config.createFlags(&custom_features);

    std.log.info("Custom feature flags: {d} features enabled", .{feature_flags.count()});
    for (custom_features, 0..) |feature, idx| {
        const enabled = feature_flags.isSet(idx);
        std.log.info("  - {s}: {s}", .{ abi.features.config.getName(feature), if (enabled) "enabled" else "disabled" });
    }

    // Demonstrate advanced component system
    std.log.info("\nAdvanced component system:", .{});

    // Create components with lifecycle functions
    const advanced_component = abi.framework.Component{
        .name = "advanced_component",
        .version = "2.0.0",
        .init_fn = componentInit,
        .deinit_fn = componentDeinit,
        .update_fn = componentUpdate,
    };

    try framework.registerComponent(advanced_component);

    // Register multiple components
    for (0..5) |i| {
        const component = abi.framework.Component{
            .name = std.fmt.allocPrint(allocator, "worker_{d}", .{i}) catch unreachable,
            .version = "1.0.0",
            .init_fn = workerInit,
            .update_fn = workerUpdate,
        };
        defer allocator.free(@constCast(component.name));

        try framework.registerComponent(component);
    }

    // Start framework and demonstrate runtime behavior
    try framework.start();
    defer framework.stop();

    std.log.info("Framework started with {d} components", .{framework.getStats().total_components});

    // Simulate runtime updates
    std.log.info("Simulating runtime updates...", .{});
    for (0..10) |i| {
        framework.update(0.016); // ~60 FPS

        if (i % 3 == 0) {
            const stats = framework.getStats();
            std.log.info("Update {d}: {d} updates completed", .{ i, stats.update_count });
        }
    }

    // Demonstrate error handling patterns
    std.log.info("\nError handling patterns:", .{});

    // Test error context creation
    const error_context = abi.core.errors.ErrorContext.initWithSource(.not_found, "Resource not found in database", "database_query");

    const formatted_error = try error_context.format(allocator);
    defer allocator.free(formatted_error);
    std.log.info("Formatted error: {s}", .{formatted_error});

    // Test generic result pattern
    const success_result = abi.core.types.GenericResult(u32).success(42);
    std.log.info("Success result: success={}, value={}", .{ success_result.success, success_result.value });

    const failure_result = abi.core.types.GenericResult(u32).failure("Operation failed");
    std.log.info("Failure result: success={}, error={s}", .{ failure_result.success, failure_result.error_message });

    // Demonstrate version management
    std.log.info("\nVersion management:", .{});

    const version1 = try abi.core.types.Version.fromString("1.2.3");
    const version2 = try abi.core.types.Version.fromString("2.0.0");

    const version_str1 = try version1.toString(allocator);
    const version_str2 = try version2.toString(allocator);
    defer allocator.free(version_str1);
    defer allocator.free(version_str2);

    std.log.info("Version 1: {s}", .{version_str1});
    std.log.info("Version 2: {s}", .{version_str2});
    std.log.info("Comparison: {d}", .{version1.compare(version2)});

    // Demonstrate advanced memory management
    std.log.info("\nAdvanced memory management:", .{});

    var tracked = abi.core.allocators.AllocatorFactory.createTracked(allocator, 2 * 1024 * 1024); // 2MB limit
    const tracked_allocator = tracked.allocator();

    // Test memory limit enforcement
    const large_allocation = tracked_allocator.alloc(u8, 1024 * 1024); // 1MB
    if (large_allocation) |memory| {
        defer tracked_allocator.free(memory);
        std.log.info("Large allocation successful: {d} bytes", .{memory.len});

        // Try to exceed memory limit
        const excess_allocation = tracked_allocator.alloc(u8, 2 * 1024 * 1024); // 2MB
        if (excess_allocation) |excess_memory| {
            defer tracked_allocator.free(excess_memory);
            std.log.info("Excess allocation unexpectedly succeeded", .{});
        } else {
            std.log.info("Memory limit enforced: excess allocation rejected", .{});
        }
    } else |err| {
        std.log.info("Large allocation failed: {}", .{err});
    }

    const final_stats = tracked.getStats();
    std.log.info("Final memory stats:", .{});
    std.log.info("  - Peak usage: {d} bytes", .{final_stats.peak_usage});
    std.log.info("  - Current usage: {d} bytes", .{final_stats.currentUsage()});
    std.log.info("  - Over limit: {}", .{final_stats.isOverLimit(1024 * 1024)});

    // Demonstrate framework summary
    std.log.info("\nFramework summary:", .{});
    try framework.writeSummary(std.io.getStdOut().writer());

    std.log.info("Advanced features example completed successfully!", .{});
}

// Component lifecycle functions
fn componentInit(allocator: std.mem.Allocator, config: *const abi.framework.RuntimeConfig) !void {
    _ = allocator;
    _ = config;
    std.log.info("Advanced component initialized", .{});
}

fn componentDeinit() !void {
    std.log.info("Advanced component deinitialized", .{});
}

fn componentUpdate(delta_time: f64) !void {
    std.log.info("Advanced component updated (delta: {d:.3}s)", .{delta_time});
}

fn workerInit(allocator: std.mem.Allocator, config: *const abi.framework.RuntimeConfig) !void {
    _ = allocator;
    _ = config;
    std.log.info("Worker component initialized", .{});
}

fn workerUpdate(delta_time: f64) !void {
    _ = delta_time;
    // Worker does some processing
}
