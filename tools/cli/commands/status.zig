//! CLI command: abi status
//!
//! Shows framework health and component status.

const std = @import("std");
const abi = @import("abi");

pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    _ = args;

    var fw = abi.initDefault(allocator) catch |err| {
        std.debug.print("Framework initialization failed: {t}\n", .{err});
        std.debug.print("\nStatus: UNHEALTHY\n", .{});
        return;
    };
    defer abi.shutdown(&fw);

    std.debug.print("ABI Framework Status\n", .{});
    std.debug.print("====================\n\n", .{});
    std.debug.print("Version:  {s}\n", .{abi.version()});
    std.debug.print("State:    {s}\n\n", .{@tagName(fw.state)});
    std.debug.print("Features:\n", .{});

    const features = [_]struct { name: []const u8, enabled: bool }{
        .{ .name = "gpu", .enabled = fw.gpu != null },
        .{ .name = "ai", .enabled = fw.ai != null },
        .{ .name = "database", .enabled = fw.database != null },
        .{ .name = "network", .enabled = fw.network != null },
        .{ .name = "web", .enabled = fw.web != null },
        .{ .name = "observability", .enabled = fw.observability != null },
        .{ .name = "analytics", .enabled = fw.analytics != null },
        .{ .name = "cloud", .enabled = fw.cloud != null },
    };

    var enabled_count: usize = 0;
    for (features) |f| {
        const marker: []const u8 = if (f.enabled) "[ok]" else "[--]";
        std.debug.print("  {s} {s}\n", .{ marker, f.name });
        if (f.enabled) enabled_count += 1;
    }

    std.debug.print("\n{d}/{d} features active\n", .{ enabled_count, features.len });
    std.debug.print("Status: HEALTHY\n", .{});
}
