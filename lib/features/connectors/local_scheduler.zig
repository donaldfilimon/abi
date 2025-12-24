const std = @import("std");
const T = @import("mod.zig");

fn init(_: std.mem.Allocator) !void {}

fn call(_: std.mem.Allocator, _: T.CallRequest) !T.CallResult {
    return .{ .ok = false, .content = "", .status_code = 501, .err_msg = "local scheduler connector not implemented" };
}

fn health() bool {
    return true;
}

pub fn get() T.Connector {
    return .{ .name = "local_scheduler", .init = init, .call = call, .health = health };
}
