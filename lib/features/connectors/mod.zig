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

// Connector implementations
pub const hf_inference = @import("hf_inference.zig");
pub const local_scheduler = @import("local_scheduler.zig");
pub const mock = @import("mock.zig");
pub const ollama = @import("ollama.zig");
pub const openai = @import("openai.zig");
pub const plugin = @import("plugin.zig");

pub const Connector = struct {
    name: []const u8,
    init: *const fn (allocator: std.mem.Allocator) anyerror!void,
    call: *const fn (allocator: std.mem.Allocator, req: CallRequest) anyerror!CallResult,
    health: *const fn () bool,
};

/// Initialize the connectors feature module
pub fn init(allocator: std.mem.Allocator) !void {
    _ = allocator; // Currently no global connector state to initialize
}

/// Deinitialize the connectors feature module
pub fn deinit() void {
    // Currently no global connector state to cleanup
}
