//! Basic Usage Example
//!
//! Demonstrates basic usage of the ABI framework

const std = @import("std");
const abi = @import("../src/mod.zig");

fn demoOperation() !u32 {
    // Simulate some operation that might fail
    return 42;
}

pub fn main() !void {
    std.log.info("ABI Framework Basic Usage Example", .{});

    // Create a general purpose allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize the framework with default configuration
    var framework = try abi.init(allocator, .{});
    defer abi.shutdown(framework);

    // Check which features are enabled
    std.log.info("Enabled features:", .{});
    const features = [_]abi.features.FeatureTag{ .ai, .gpu, .database, .web, .monitoring, .connectors };
    for (features) |feature| {
        const enabled = framework.isFeatureEnabled(feature);
        const status = if (enabled) "enabled" else "disabled";
        std.log.info("  - {s}: {s}", .{ abi.features.config.getName(feature), status });
    }

    // Start the framework
    try framework.start();
    std.log.info("Framework started", .{});

    // Get runtime statistics
    const stats = framework.getStats();
    std.log.info("Runtime stats:", .{});
    std.log.info("  - Components: {d}", .{stats.total_components});
    std.log.info("  - Active components: {d}", .{stats.active_components});
    std.log.info("  - Enabled features: {d}", .{stats.enabled_features});
    std.log.info("  - Uptime: {d}ms", .{stats.uptime()});

    // Demonstrate feature management
    std.log.info("\nFeature management demo:", .{});
    std.log.info("GPU feature is currently: {s}", .{if (framework.isFeatureEnabled(.gpu)) "enabled" else "disabled"});

    framework.enableFeature(.gpu);
    std.log.info("GPU feature after enabling: {s}", .{if (framework.isFeatureEnabled(.gpu)) "enabled" else "disabled"});

    framework.disableFeature(.gpu);
    std.log.info("GPU feature after disabling: {s}", .{if (framework.isFeatureEnabled(.gpu)) "enabled" else "disabled"});

    // Demonstrate component registration
    std.log.info("\nComponent registration demo:", .{});
    const test_component = abi.framework.Component{
        .name = "example_component",
        .version = "1.0.0",
    };

    try framework.registerComponent(test_component);
    std.log.info("Registered component: {s}", .{test_component.name});

    const component = framework.getComponent("example_component");
    if (component) |comp| {
        std.log.info("Retrieved component: {s} v{s}", .{ comp.name, comp.version });
    }

    // Demonstrate I/O abstraction
    std.log.info("\nI/O abstraction demo:", .{});
    const writer = abi.core.Writer.stdout();
    try writer.print("Using the new I/O abstraction layer!\n");
    
    // Demonstrate error handling
    std.log.info("\nError handling demo:", .{});
    const result = demoOperation() catch |err| {
        const ctx = abi.core.ErrorContext.init(err, "Demo operation failed")
            .withLocation(abi.core.here())
            .withContext("This is a demonstration of rich error context");
        
        std.log.err("Error occurred: {}", .{ctx});
        return err;
    };
    
    std.log.info("Operation result: {d}", .{result});

    // Demonstrate collections
    std.log.info("\nCollections demo:", .{});
    var list = std.ArrayList(u32).init(allocator);
    defer list.deinit();

    for (0..10) |i| {
        try list.append(@intCast(i * i));
    }

    std.log.info("Squares: {d}", .{list.items});

    // Stop the framework
    framework.stop();
    std.log.info("Framework stopped", .{});

    std.log.info("Basic usage example completed successfully!", .{});
}
