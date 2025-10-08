//! Basic Usage Example
//!
//! Demonstrates basic usage of the ABI framework

const std = @import("std");
const abi = @import("../lib/mod.zig");

pub fn main() !void {
    std.log.info("ABI Framework Basic Usage Example", .{});
    
    // Create a general purpose allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Initialize the framework with default configuration
    var framework = try abi.createDefaultFramework(allocator);
    defer framework.deinit();
    
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
    std.log.info("GPU feature is currently: {s}", .{
        if (framework.isFeatureEnabled(.gpu)) "enabled" else "disabled"
    });
    
    framework.enableFeature(.gpu);
    std.log.info("GPU feature after enabling: {s}", .{
        if (framework.isFeatureEnabled(.gpu)) "enabled" else "disabled"
    });
    
    framework.disableFeature(.gpu);
    std.log.info("GPU feature after disabling: {s}", .{
        if (framework.isFeatureEnabled(.gpu)) "enabled" else "disabled"
    });
    
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
    
    // Demonstrate memory management
    std.log.info("\nMemory management demo:", .{});
    var tracked = abi.core.allocators.AllocatorFactory.createTracked(allocator, 1024 * 1024); // 1MB limit
    const tracked_allocator = tracked.allocator();
    
    // Allocate some memory
    const memory = try tracked_allocator.alloc(u8, 1024);
    defer tracked_allocator.free(memory);
    
    const memory_stats = tracked.getStats();
    std.log.info("Memory stats:", .{});
    std.log.info("  - Allocated: {d} bytes", .{memory_stats.bytes_allocated});
    std.log.info("  - Freed: {d} bytes", .{memory_stats.bytes_freed});
    std.log.info("  - Current usage: {d} bytes", .{memory_stats.currentUsage()});
    std.log.info("  - Peak usage: {d} bytes", .{memory_stats.peak_usage});
    
    // Demonstrate collections
    std.log.info("\nCollections demo:", .{});
    var list = abi.core.utils.createArrayList(u32, allocator);
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