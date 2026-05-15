const std = @import("std");
const os_config = @import("os_config.zig");

pub const OSController = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) OSController {
        return .{ .allocator = allocator };
    }

    pub fn execute(self: *OSController, command: os_config.Command) !void {
        _ = self;
        switch (command) {
            .list_processes => std.log.info("Executing list_processes...", .{}),
            .get_cpu_usage => std.log.info("Executing get_cpu_usage...", .{}),
            else => std.log.err("Command not implemented: {any}", .{command}),
        }
    }
};
