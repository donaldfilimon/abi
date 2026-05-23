const std = @import("std");
const Registry = @import("../../core/registry.zig").Registry;
const usage_mod = @import("../usage.zig");

pub fn handlePlugin(allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len != 3 or !std.mem.eql(u8, args[2], "list")) return usage_mod.usageError("usage: abi plugin list");

    var registry = Registry.init(allocator);
    defer registry.deinit();
    try registry.loadPlugins();

    std.debug.print("Installed Plugins:\n", .{});
    var it = registry.modules.iterator();
    while (it.next()) |entry| {
        std.debug.print("  - {s}: {s}\n", .{ entry.key_ptr.*, entry.value_ptr.* });
    }
    return 0;
}
