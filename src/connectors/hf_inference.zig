const T = @import("mod.zig");

fn init(_: std.mem.Allocator) !void {}

fn call(_: std.mem.Allocator, _: T.CallRequest) !T.CallResult {
    return .{ .ok = false, .content = "", .status_code = 501, .err_msg = "hf inference connector not implemented" };
}

fn health() bool {
    return true;
}

pub fn get() T.Connector {
    return .{ .name = "hf_inference", .init = init, .call = call, .health = health };
}
