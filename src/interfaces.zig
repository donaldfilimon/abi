//! Cross-module contracts for the greenfield tree (behavior-compatible rewrite).
const std = @import("std");

pub const WdbxSearchResult = struct {
    id: u32,
    score: f32,
};

pub const AiRunRequest = struct {
    input: []const u8,
};

pub const AiCompleteRequest = struct {
    input: []const u8,
    model: []const u8 = "abi-local",
};

pub const AiTrainRequest = struct {
    profile: []const u8,
    dataset: []const u8,
    artifact_dir: []const u8 = "zig-cache/agents",
};

pub const ConnectorResponse = struct {
    body: []const u8,
    owned: bool = false,

    pub fn deinit(self: ConnectorResponse, allocator: std.mem.Allocator) void {
        if (self.owned) allocator.free(self.body);
    }
};

pub const BackendStatusLine = struct {
    name: []const u8,
    available: bool,
    accelerated: bool,
    native_kernels: bool,
    message: []const u8,
};

test {
    std.testing.refAllDecls(@This());
}
