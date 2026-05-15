const std = @import("std");
const abi = @import("root.zig");

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.page_allocator;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) return;

    const cmd = args[1];

    if (std.mem.eql(u8, cmd, "train") and args.len >= 3) {
        const input = args[2];
        const response = try abi.features.ai.run(allocator, input);
        defer allocator.free(response);
        std.log.info("Pipeline Output: {s}", .{response});
    } else if (std.mem.eql(u8, cmd, "plugin") and args.len >= 3) {
        const sub_cmd = args[2];
        if (std.mem.eql(u8, sub_cmd, "list")) {
            var registry = abi.registry.Registry.init(allocator);
            defer registry.deinit();
            try registry.loadPlugins();

            std.debug.print("Installed Plugins:\n", .{});
            var it = registry.modules.iterator();
            while (it.next()) |entry| {
                std.debug.print("  - {s}: {s}\n", .{ entry.key_ptr.*, entry.value_ptr.* });
            }
        }
    } else {
        std.debug.print("Usage: abi [train <input> | plugin list]\n", .{});
    }
}
