const std = @import("std");

pub const CallRequest = struct {
    model: []const u8,
    prompt: []const u8,
    max_tokens: u16,
    temperature: f32 = 0.2,
};

pub const CallResult = struct {
    ok: bool,
    content: []const u8,
    tokens_in: u32 = 0,
    tokens_out: u32 = 0,
    status_code: u16 = 200,
    err_msg: ?[]const u8 = null,
};

pub const Connector = struct {
    name: []const u8,
    init: *const fn (allocator: std.mem.Allocator) !void,
    call: *const fn (allocator: std.mem.Allocator, req: CallRequest) !CallResult,
    health: *const fn () bool,
};
