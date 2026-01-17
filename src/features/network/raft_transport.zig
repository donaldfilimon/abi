//! Transport layer for Raft consensus.
//! Connects the Raft module to a TCP transport implementation.
//! This is a minimal stub to satisfy compilation; full networking is out of scope.

const std = @import("std");
const Raft = @import("raft.zig");

pub const RaftTransport = struct {
    allocator: std.mem.Allocator,
    // Placeholder for a TCP listener or client socket.
    // In a full implementation this would be a std.net.StreamSocket or similar.
    // Here we store a dummy handle for compile‑time type checking.
    dummy_handle: ?*anyopaque = null,

    pub fn init(allocator: std.mem.Allocator) RaftTransport {
        return .{ .allocator = allocator };
    }

    /// Bind to a local address (stub).
    pub fn bind(self: *RaftTransport, address: []const u8) !void {
        // TODO: Implement real socket binding.
        _ = address;
        // For now just set dummy_handle to a non-null value.
        self.dummy_handle = @ptrCast(@alignCast(&self));
    }

    /// Connect to a remote Raft peer (stub).
    pub fn connect(self: *RaftTransport, address: []const u8) !void {
        _ = self;
        _ = address;
        // No‑op stub.
    }

    /// Send a Raft message (stub).
    pub fn send(self: *RaftTransport, msg: Raft.Message) !void {
        _ = self;
        _ = msg;
        // No‑op stub.
    }
};
