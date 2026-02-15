//! Feature Registry Example
//!
//! Demonstrates the ABI feature registry system for managing
//! feature lifecycle with comptime and runtime toggle modes.

const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("ABI Feature Registry Example\n", .{});
    std.debug.print("============================\n\n", .{});

    // Initialize the feature registry
    var registry = abi.registry.Registry.init(allocator);
    defer registry.deinit();

    // Check compile-time feature availability
    std.debug.print("Compile-time Feature Status:\n", .{});
    const features = [_]abi.config.Feature{
        .gpu,
        .ai,
        .llm,
        .embeddings,
        .agents,
        .training,
        .database,
        .network,
        .observability,
        .web,
        .personas,
        .cloud,
    };

    inline for (features) |feature| {
        const enabled = comptime feature.isCompileTimeEnabled();
        const status = if (enabled) "enabled" else "disabled";
        std.debug.print("  {s}: {s} - {s}\n", .{
            feature.name(),
            status,
            feature.description(),
        });
    }

    std.debug.print("\nRegistering features...\n", .{});

    // Register comptime-only features (zero overhead)
    if (comptime abi.registry.isFeatureCompiledIn(.gpu)) {
        try registry.registerComptime(.gpu);
        std.debug.print("  Registered .gpu (comptime-only)\n", .{});
    }

    if (comptime abi.registry.isFeatureCompiledIn(.database)) {
        try registry.registerComptime(.database);
        std.debug.print("  Registered .database (comptime-only)\n", .{});
    }

    // Query registered features
    std.debug.print("\nRegistry Status:\n", .{});
    for (features) |feature| {
        if (registry.isRegistered(feature)) {
            const enabled_str = if (registry.isEnabled(feature)) "enabled" else "disabled";
            std.debug.print("  {s}: registered, {s}\n", .{ feature.name(), enabled_str });
        }
    }

    // Demonstrate feature hierarchy
    std.debug.print("\nFeature Hierarchy:\n", .{});
    const sub_features = [_]abi.config.Feature{ .llm, .embeddings, .agents, .training, .personas };
    for (sub_features) |feature| {
        if (abi.registry.getParentFeature(feature)) |parent| {
            std.debug.print("  {s} -> parent: {s}\n", .{ feature.name(), parent.name() });
        }
    }

    std.debug.print("\nRegistry example completed successfully!\n", .{});
}
