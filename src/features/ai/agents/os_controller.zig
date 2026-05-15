const std = @import("std");
const foundation = @import("../../../foundation/mod.zig");

pub const OSControllerAgent = struct {
    controller: foundation.os.OSController,

    pub fn init(allocator: std.mem.Allocator) OSControllerAgent {
        return .{ .controller = foundation.os.OSController.init(allocator) };
    }

    pub fn runCommand(self: *OSControllerAgent, command: foundation.os_config.Command) !void {
        try self.controller.execute(command);
    }
};
