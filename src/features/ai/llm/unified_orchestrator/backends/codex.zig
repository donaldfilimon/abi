//! Codex CLI backend stub.

const std = @import("std");
const types = @import("../protocol/types.zig");
const BackendInterface = @import("backend.zig").BackendInterface;

fn run(
    allocator: std.mem.Allocator,
    request: types.InferenceRequest,
) !types.InferenceResponse {
    _ = allocator;
    _ = request;
    return .{ .text = "", .finish_reason = "stop" };
}

pub const interface = BackendInterface{
    .run = run,
    .stream = null,
};
