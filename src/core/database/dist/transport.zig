//! Minimal TCP-based RPC transport for WDBX.
//!
//! Wraps `std.posix.socket_t` to send and receive `dist.rpc` frames. This transport
//! avoids full Raft semantics but allows current sync logic to cross socket boundaries.

const std = @import("std");
const rpc = @import("rpc");

pub const ConnectionError = error{
    ConnectionClosed,
    BufferTooSmall,
    InvalidMessageType,
    PayloadTooLarge,
};

/// Reads exactly `buffer.len` bytes from `socket` into `buffer`.
fn readExact(socket: std.posix.socket_t, buffer: []u8) ConnectionError!void {
    var total_read: usize = 0;
    while (total_read < buffer.len) {
        const amt = try std.posix.recv(socket, buffer[total_read..], 0);
        if (amt == 0) return error.ConnectionClosed;
        total_read += amt;
    }
}

/// A single incoming message with decoded header and raw payload slice.
pub const IncomingMessage = struct {
    header: rpc.Header,
    payload: []const u8,
};

/// Transport handle that abstracts connection and dispatch.
pub const TcpTransport = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) TcpTransport {
        return .{ .allocator = allocator };
    }

    /// Synchronously sends a message consisting of `msg_type` and `payload`.
    /// In a real async environment this would push to a send queue.
    pub fn sendMessage(self: *TcpTransport, socket: std.posix.socket_t, msg_type: rpc.MessageType, payload: []const u8) !void {
        if (payload.len > std.math.maxInt(u32)) return error.PayloadTooLarge;

        const need = rpc.Header.encoded_len + payload.len;

        // Stack-allocated buffer for short messages, dynamic for large.
        if (need <= 4096) {
            var buf: [4096]u8 = undefined;
            const size = try rpc.encodeMessage(msg_type, payload, &buf);
            var total_sent: usize = 0;
            while (total_sent < size) {
                const sent = try std.posix.send(socket, buf[total_sent..size], 0);
                total_sent += sent;
            }
        } else {
            // Unlikely to need this for control messages, but safe for block data
            var buf = try self.allocator.alloc(u8, need);
            defer self.allocator.free(buf);
            const size = try rpc.encodeMessage(msg_type, payload, buf);
            var total_sent: usize = 0;
            while (total_sent < size) {
                const sent = try std.posix.send(socket, buf[total_sent..size], 0);
                total_sent += sent;
            }
        }
    }

    /// Read a single message frame from `socket`.
    /// The returned payload buffer is allocated by `allocator` and MUST be freed by caller.
    pub fn recvMessageAlloc(self: *TcpTransport, socket: std.posix.socket_t) !IncomingMessage {
        var header_buf: [rpc.Header.encoded_len]u8 = undefined;
        try readExact(socket, &header_buf);

        const header = try rpc.decodeHeader(&header_buf);

        const payload_buf = try self.allocator.alloc(u8, header.payload_len);
        errdefer self.allocator.free(payload_buf);

        if (header.payload_len > 0) {
            try readExact(socket, payload_buf);
        }

        return IncomingMessage{
            .header = header,
            .payload = payload_buf,
        };
    }
};

test "TcpTransport syntactic test" {
    // Only syntactic compilation check to ensure types and symbols resolve,
    // avoiding OS socket creation for cross-platform unit test compat in -fno-emit-bin
    const allocator = std.testing.allocator;
    const tp = TcpTransport.init(allocator);
    _ = tp;
}

test {
    std.testing.refAllDecls(@This());
}
