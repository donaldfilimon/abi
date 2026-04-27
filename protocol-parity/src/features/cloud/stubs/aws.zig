const std = @import("std");
const types = @import("types.zig");

pub const LambdaRuntime = struct {
    pub fn init(allocator: std.mem.Allocator, handler: types.CloudHandler) !LambdaRuntime {
        _ = allocator;
        _ = handler;
        return types.Error.CloudDisabled;
    }

    pub fn run(self: *LambdaRuntime) !void {
        _ = self;
        return types.Error.CloudDisabled;
    }
};

pub fn parseEvent(allocator: std.mem.Allocator, raw_event: []const u8, request_id: []const u8) !types.CloudEvent {
    _ = allocator;
    _ = raw_event;
    _ = request_id;
    return types.Error.CloudDisabled;
}

pub fn formatResponse(allocator: std.mem.Allocator, response: *const types.CloudResponse) ![]const u8 {
    _ = allocator;
    _ = response;
    return types.Error.CloudDisabled;
}

pub fn runHandler(allocator: std.mem.Allocator, handler: types.CloudHandler) !void {
    _ = allocator;
    _ = handler;
    return types.Error.CloudDisabled;
}

pub fn createTestHandler(comptime handler_fn: anytype) types.CloudHandler {
    _ = handler_fn;
    return struct {
        pub fn handle(_: *types.CloudEvent, _: std.mem.Allocator) anyerror!types.CloudResponse {
            return types.Error.CloudDisabled;
        }
    }.handle;
}

test {
    std.testing.refAllDecls(@This());
}
