//! Backend interface for the unified orchestrator.

const std = @import("std");
const types = @import("../protocol/types.zig");

pub const BackendInterface = struct {
    /// Run non-streaming inference.
    run: *const fn (
        allocator: std.mem.Allocator,
        request: types.InferenceRequest,
    ) anyerror!types.InferenceResponse,
    /// Run streaming inference; optional (null = not supported).
    stream: ?*const fn (
        allocator: std.mem.Allocator,
        request: types.InferenceRequest,
        callback: types.StreamCallback,
    ) anyerror!void = null,
};
