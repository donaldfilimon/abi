//! System information command.

const std = @import("std");
const abi = @import("abi");
const gpu = @import("gpu.zig");
const network = @import("network.zig");

/// Run the system-info command.
pub fn run(allocator: std.mem.Allocator, framework: *abi.Framework) !void {
    const platform = abi.platform.platform;
    const info = platform.PlatformInfo.detect();

    std.debug.print("System Info\n", .{});
    std.debug.print("  OS: {t}\n", .{info.os});
    std.debug.print("  Arch: {t}\n", .{info.arch});
    std.debug.print("  CPU Threads: {d}\n", .{info.max_threads});
    std.debug.print("  ABI Version: {s}\n", .{abi.version()});
    try gpu.printSummary(allocator);
    network.printSummary();

    std.debug.print("\nFeature Matrix:\n", .{});
    for (std.enums.values(abi.Feature)) |tag| {
        const status = if (framework.isFeatureEnabled(tag)) "enabled" else "disabled";
        std.debug.print("  {t}: {s}\n", .{ tag, status });
    }
}
