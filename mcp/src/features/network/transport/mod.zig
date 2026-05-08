//! TCP Transport Layer for Distributed Systems
//!
//! Provides actual network I/O for distributed database queries, Raft consensus,
//! and general RPC communication. Uses connection pooling and health checks.
//!
//! Key features:
//! - TCP client and server with connection pooling
//! - Message framing with length-prefix protocol
//! - Request/response correlation with request IDs
//! - Automatic reconnection with exponential backoff
//! - Health checks and connection validation
//!
//! Usage:
//!   var transport = try TcpTransport.init(allocator, .{ .listen_port = 9000 });
//!   defer transport.deinit();
//!   try transport.start();
//!   const response = try transport.sendRequest("192.168.1.2:9000", request_data);

// Re-export all public symbols from sub-modules

// Protocol types
pub const NetworkAddress = @import("protocol.zig").NetworkAddress;
pub const TransportConfig = @import("protocol.zig").TransportConfig;
pub const MessageType = @import("protocol.zig").MessageType;
pub const MessageHeader = @import("protocol.zig").MessageHeader;
pub const TransportError = @import("protocol.zig").TransportError;
pub const parseAddress = @import("protocol.zig").parseAddress;

// Connection types
pub const PendingRequest = @import("connection.zig").PendingRequest;
pub const PeerConnection = @import("connection.zig").PeerConnection;

// TCP transport
pub const TcpTransport = @import("tcp.zig").TcpTransport;

// RPC serialization
pub const RpcSerializer = @import("rpc.zig").RpcSerializer;

// Test discovery
test {
    _ = @import("protocol.zig");
    _ = @import("connection.zig");
    _ = @import("tcp.zig");
    _ = @import("rpc.zig");
}
