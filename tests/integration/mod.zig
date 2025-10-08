//! Integration Tests
//!
//! Integration tests for the ABI framework

const std = @import("std");
const abi = @import("abi");

test "framework full lifecycle" {
    var framework = try abi.createFramework(std.testing.allocator, .{
        .enabled_features = &[_]abi.features.FeatureTag{ .ai, .database, .web },
        .disabled_features = &[_]abi.features.FeatureTag{.gpu},
        .log_level = .debug,
    });
    defer framework.deinit();
    
    // Verify initial state
    try std.testing.expect(framework.isFeatureEnabled(.ai));
    try std.testing.expect(framework.isFeatureEnabled(.database));
    try std.testing.expect(framework.isFeatureEnabled(.web));
    try std.testing.expect(!framework.isFeatureEnabled(.gpu));
    
    // Start framework
    try framework.start();
    try std.testing.expect(framework.isRunning());
    
    // Verify stats
    const stats = framework.getStats();
    try std.testing.expectEqual(@as(usize, 3), stats.enabled_features);
    
    // Stop framework
    framework.stop();
    try std.testing.expect(!framework.isRunning());
}

test "feature runtime integration" {
    var framework = try abi.createDefaultFramework(std.testing.allocator);
    defer framework.deinit();
    
    // Test feature lifecycle
    try abi.features.lifecycle.initFeatures(std.testing.allocator, &[_]abi.features.FeatureTag{ .ai, .database });
    
    // Enable additional features
    framework.enableFeature(.web);
    framework.enableFeature(.monitoring);
    
    // Start framework
    try framework.start();
    
    // Verify all features are enabled
    try std.testing.expect(framework.isFeatureEnabled(.ai));
    try std.testing.expect(framework.isFeatureEnabled(.database));
    try std.testing.expect(framework.isFeatureEnabled(.web));
    try std.testing.expect(framework.isFeatureEnabled(.monitoring));
    
    // Cleanup
    abi.features.lifecycle.deinitFeatures(&[_]abi.features.FeatureTag{ .ai, .database, .web, .monitoring });
    framework.stop();
}

test "memory management integration" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    
    const allocator = gpa.allocator();
    
    // Test tracked allocator with framework
    var tracked = abi.core.allocators.AllocatorFactory.createTracked(allocator, 1024 * 1024); // 1MB limit
    const tracked_allocator = tracked.allocator();
    
    var framework = try abi.createFramework(tracked_allocator, .{
        .memory_limit_mb = 1,
    });
    defer framework.deinit();
    
    // Verify memory tracking
    const initial_stats = tracked.getStats();
    try std.testing.expectEqual(@as(usize, 0), initial_stats.bytes_allocated);
    
    // Start framework (should allocate memory)
    try framework.start();
    
    const after_start_stats = tracked.getStats();
    try std.testing.expect(after_start_stats.bytes_allocated > 0);
    
    // Stop and verify cleanup
    framework.stop();
    
    const after_stop_stats = tracked.getStats();
    // Memory should be freed (or at least not increased significantly)
    try std.testing.expect(after_stop_stats.bytes_allocated <= after_start_stats.bytes_allocated);
}

test "configuration persistence" {
    var framework1 = try abi.createFramework(std.testing.allocator, .{
        .enabled_features = &[_]abi.features.FeatureTag{ .ai, .gpu },
        .disabled_features = &[_]abi.features.FeatureTag{.database},
        .log_level = .debug,
    });
    defer framework1.deinit();
    
    // Verify configuration
    try std.testing.expect(framework1.isFeatureEnabled(.ai));
    try std.testing.expect(framework1.isFeatureEnabled(.gpu));
    try std.testing.expect(!framework1.isFeatureEnabled(.database));
    
    // Create another framework with same config
    var framework2 = try abi.createFramework(std.testing.allocator, .{
        .enabled_features = &[_]abi.features.FeatureTag{ .ai, .gpu },
        .disabled_features = &[_]abi.features.FeatureTag{.database},
        .log_level = .debug,
    });
    defer framework2.deinit();
    
    // Verify both have same configuration
    try std.testing.expectEqual(framework1.isFeatureEnabled(.ai), framework2.isFeatureEnabled(.ai));
    try std.testing.expectEqual(framework1.isFeatureEnabled(.gpu), framework2.isFeatureEnabled(.gpu));
    try std.testing.expectEqual(framework1.isFeatureEnabled(.database), framework2.isFeatureEnabled(.database));
}

test "component system integration" {
    var framework = try abi.createDefaultFramework(std.testing.allocator);
    defer framework.deinit();
    
    // Register multiple components
    const components = [_]abi.framework.Component{
        .{ .name = "component1", .version = "1.0.0" },
        .{ .name = "component2", .version = "2.0.0" },
        .{ .name = "component3", .version = "3.0.0" },
    };
    
    for (components) |component| {
        try framework.registerComponent(component);
    }
    
    try std.testing.expectEqual(@as(u32, 3), framework.getStats().total_components);
    
    // Start framework (should initialize all components)
    try framework.start();
    
    const stats = framework.getStats();
    try std.testing.expectEqual(@as(u32, 3), stats.active_components);
    
    // Verify components are accessible
    const component1 = framework.getComponent("component1");
    try std.testing.expect(component1 != null);
    try std.testing.expectEqualStrings("component1", component1.?.name);
    
    framework.stop();
}

test "error handling integration" {
    var framework = try abi.createDefaultFramework(std.testing.allocator);
    defer framework.deinit();
    
    // Test error propagation
    const result = abi.core.types.GenericResult(u32).success(42);
    try std.testing.expect(result.success);
    try std.testing.expectEqual(@as(u32, 42), result.value);
    
    const failure = abi.core.types.GenericResult(u32).failure("test error");
    try std.testing.expect(!failure.success);
    try std.testing.expectEqualStrings("test error", failure.error_message);
    
    // Test error context
    const context = abi.core.errors.ErrorContext.initWithSource(.not_found, "resource not found", "database");
    const formatted = try context.format(std.testing.allocator);
    defer std.testing.allocator.free(formatted);
    
    try std.testing.expect(std.mem.indexOf(u8, formatted, "not_found") != null);
    try std.testing.expect(std.mem.indexOf(u8, formatted, "database") != null);
    try std.testing.expect(std.mem.indexOf(u8, formatted, "resource not found") != null);
}

test "performance characteristics" {
    var framework = try abi.createDefaultFramework(std.testing.allocator);
    defer framework.deinit();
    
    // Test framework startup time
    const start_time = std.time.nanoTimestamp();
    try framework.start();
    const startup_time = std.time.nanoTimestamp() - start_time;
    
    // Startup should be reasonably fast (less than 100ms)
    try std.testing.expect(startup_time < 100_000_000); // 100ms in nanoseconds
    
    // Test feature toggling performance
    const toggle_start = std.time.nanoTimestamp();
    for (0..100) |_| {
        framework.enableFeature(.gpu);
        framework.disableFeature(.gpu);
    }
    const toggle_time = std.time.nanoTimestamp() - toggle_start;
    
    // Feature toggling should be very fast (less than 1ms for 100 toggles)
    try std.testing.expect(toggle_time < 1_000_000); // 1ms in nanoseconds
    
    framework.stop();
}