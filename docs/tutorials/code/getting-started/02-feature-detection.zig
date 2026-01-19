//! Getting Started Tutorial - Example 2: Feature Detection
//!
//! Run with: zig run docs/tutorials/code/getting-started/02-feature-detection.zig

const std = @import("std");
// In a real project, you would use: const abi = @import("abi");
// For tutorial purposes, we use a relative path.
const abi = @import("../../../../src/abi.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var framework = try abi.initDefault(allocator);
    defer framework.deinit();

    std.debug.print("\n=== ABI Feature Status ===\n\n", .{});

    const features = [_]struct { name: []const u8, enabled: bool }{
        .{ .name = "AI/LLM", .enabled = framework.isEnabled(.ai) },
        .{ .name = "GPU", .enabled = framework.isEnabled(.gpu) },
        .{ .name = "Database", .enabled = framework.isEnabled(.database) },
        .{ .name = "Network", .enabled = framework.isEnabled(.network) },
        .{ .name = "Web", .enabled = framework.isEnabled(.web) },
        .{ .name = "Observability", .enabled = framework.isEnabled(.observability) },
    };

    for (features) |f| {
        const status = if (f.enabled) "[ENABLED]" else "[DISABLED]";
        std.debug.print("  {s:<13} {s}\n", .{ f.name, status });
    }

    std.debug.print("\n", .{});
}
