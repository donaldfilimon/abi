const std = @import("std");

pub const TaskEnvelope = struct {
    task_id: []const u8 = "",
    payload: []const u8 = "",
};

pub const ResultEnvelope = struct {
    task_id: []const u8 = "",
    status: ResultStatus = .success,
    payload: []const u8 = "",
};

pub const ResultStatus = enum {
    success,
    failure,
    timeout,
};

pub fn encodeTask(_: std.mem.Allocator, _: TaskEnvelope) ![]const u8 {
    return error.NetworkDisabled;
}

pub fn decodeTask(_: std.mem.Allocator, _: []const u8) !TaskEnvelope {
    return error.NetworkDisabled;
}

pub fn encodeResult(_: std.mem.Allocator, _: ResultEnvelope) ![]const u8 {
    return error.NetworkDisabled;
}

pub fn decodeResult(_: std.mem.Allocator, _: []const u8) !ResultEnvelope {
    return error.NetworkDisabled;
}

test {
    std.testing.refAllDecls(@This());
}
