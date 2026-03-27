//! Compute stub — disabled at compile time.

const std = @import("std");
pub const types = @import("types.zig");

pub const mesh = struct {
    pub const ComputeNode = struct {
        id: [16]u8 = [_]u8{0} ** 16,
        address: std.c.sockaddr.in = std.mem.zeroes(std.c.sockaddr.in),
        is_local: bool = false,
        available_vram_mb: u64 = 0,
        backend: BackendType = .cpu,
        last_seen_ms: i64 = 0,

        pub const BackendType = enum(u8) { metal = 0, cuda = 1, rocm = 2, vulkan = 3, cpu = 4 };
    };

    pub const MeshOrchestrator = struct {
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator, _: anytype) !MeshOrchestrator {
            _ = allocator;
            return error.MeshUnavailable;
        }

        pub fn deinit(_: *MeshOrchestrator) void {}
    };
};

pub const ComputeError = types.ComputeError;
pub const Error = types.Error;

pub const Context = struct {
    allocator: std.mem.Allocator,
    initialized: bool = false,

    pub fn init(allocator: std.mem.Allocator) Context {
        return .{ .allocator = allocator, .initialized = false };
    }

    pub fn deinit(self: *Context) void {
        _ = self;
    }
};

pub fn isEnabled() bool {
    return false;
}

pub fn isInitialized() bool {
    return false;
}

test {
    std.testing.refAllDecls(@This());
}
