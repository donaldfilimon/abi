const std = @import("std");
const T = @import("mod.zig");

fn init(_: std.mem.Allocator) !void {}

fn call(allocator: std.mem.Allocator, req: T.CallRequest) !T.CallResult {
    if (req.prompt.len == 0) {
        return .{ .ok = false, .content = "", .status_code = 400, .err_msg = "empty prompt" };
    }

    const content = try std.fmt.allocPrint(allocator, "MOCK: {s}", .{req.prompt});
    return .{
        .ok = true,
        .content = content,
        .tokens_in = @intCast(req.prompt.len),
        .tokens_out = @intCast(content.len),
    };
}

fn health() bool {
    return true;
}

pub fn get() T.Connector {
    return .{ .name = "mock", .init = init, .call = call, .health = health };
}
