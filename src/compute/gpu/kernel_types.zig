//! GPU kernel types and configuration primitives.
const std = @import("std");
const backend = @import("backend.zig");

pub const KernelError = error{
    CompilationFailed,
    LaunchFailed,
    InvalidArguments,
    UnsupportedBackend,
    MissingDevice,
};

pub const KernelSource = struct {
    name: []const u8,
    source: []const u8,
    entry_point: []const u8 = "main",
    backend: backend.Backend,

    pub fn deinit(self: *KernelSource, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.source);
        allocator.free(self.entry_point);
        self.* = undefined;
    }
};

pub const KernelConfig = struct {
    grid_dim: [3]u32 = .{ 1, 1, 1 },
    block_dim: [3]u32 = .{ 1, 1, 1 },
    shared_memory_bytes: u32 = 0,
    stream: ?*Stream = null,
};

pub const Stream = struct {
    id: u64,
    backend: backend.Backend,
    running: std.atomic.Value(bool),

    pub fn init(backend_id: backend.Backend) Stream {
        return .{
            .id = std.time.nanoTimestamp(),
            .backend = backend_id,
            .running = std.atomic.Value(bool).init(true),
        };
    }

    pub fn synchronize(self: *Stream) void {
        while (self.running.load(.acquire)) {
            std.atomic.spinLoopHint();
        }
    }

    pub fn stop(self: *Stream) void {
        self.running.store(false, .release);
    }
};
