//! Unit Tests
//!
//! Comprehensive unit tests for the ABI framework

const std = @import("std");
const abi = @import("abi");

test "core collections" {
    var list = abi.core.utils.createArrayList(u32, std.testing.allocator);
    defer list.deinit();
    
    try list.append(42);
    try std.testing.expectEqual(@as(usize, 1), list.items.len);
    try std.testing.expectEqual(@as(u32, 42), list.items[0]);
}

test "core types" {
    const version = try abi.core.types.Version.fromString("1.2.3");
    try std.testing.expectEqual(@as(u32, 1), version.major);
    try std.testing.expectEqual(@as(u32, 2), version.minor);
    try std.testing.expectEqual(@as(u32, 3), version.patch);
}

test "core errors" {
    const context = abi.core.errors.ErrorContext.init(.invalid_request, "test error");
    const formatted = try context.format(std.testing.allocator);
    defer std.testing.allocator.free(formatted);
    
    try std.testing.expectEqualStrings("[invalid_request] test error", formatted);
}

test "core allocators" {
    var tracked = abi.core.allocators.AllocatorFactory.createTracked(std.testing.allocator, 1024);
    const allocator = tracked.allocator();
    
    const memory = try allocator.alloc(u8, 100);
    defer allocator.free(memory);
    
    const stats = tracked.getStats();
    try std.testing.expectEqual(@as(usize, 100), stats.bytes_allocated);
}

test "features configuration" {
    const enabled = [_]abi.features.FeatureTag{ .ai, .database, .web };
    const flags = abi.features.config.createFlags(&enabled);
    
    try std.testing.expect(flags.isSet(0)); // ai
    try std.testing.expect(!flags.isSet(1)); // gpu
    try std.testing.expect(flags.isSet(2)); // database
    try std.testing.expect(flags.isSet(3)); // web
}

test "framework initialization" {
    var framework = try abi.createDefaultFramework(std.testing.allocator);
    defer framework.deinit();
    
    try std.testing.expect(!framework.isRunning());
    try std.testing.expect(framework.isFeatureEnabled(.ai));
    try std.testing.expect(framework.isFeatureEnabled(.database));
}

test "framework feature management" {
    var framework = try abi.createDefaultFramework(std.testing.allocator);
    defer framework.deinit();
    
    try std.testing.expect(framework.isFeatureEnabled(.ai));
    
    framework.disableFeature(.ai);
    try std.testing.expect(!framework.isFeatureEnabled(.ai));
    
    framework.enableFeature(.ai);
    try std.testing.expect(framework.isFeatureEnabled(.ai));
}

test "framework runtime lifecycle" {
    var framework = try abi.createDefaultFramework(std.testing.allocator);
    defer framework.deinit();
    
    try std.testing.expect(!framework.isRunning());
    
    try framework.start();
    try std.testing.expect(framework.isRunning());
    
    framework.stop();
    try std.testing.expect(!framework.isRunning());
}

test "framework component registration" {
    var framework = try abi.createDefaultFramework(std.testing.allocator);
    defer framework.deinit();
    
    const test_component = abi.framework.Component{
        .name = "test",
        .version = "1.0.0",
    };
    
    try framework.registerComponent(test_component);
    try std.testing.expectEqual(@as(u32, 1), framework.getStats().total_components);
}

test "abi version" {
    const version = abi.version();
    try std.testing.expectEqualStrings("0.1.0a", version);
}

test "abi initialization" {
    var framework = try abi.init(std.testing.allocator, abi.framework.defaultConfig());
    defer abi.shutdown(&framework);
    
    try std.testing.expect(!framework.isRunning());
}

test "abi compatibility namespace" {
    // Test that the wdbx compatibility namespace is accessible
    _ = abi.wdbx.database;
    _ = abi.wdbx.helpers;
    _ = abi.wdbx.cli;
    _ = abi.wdbx.http;
}