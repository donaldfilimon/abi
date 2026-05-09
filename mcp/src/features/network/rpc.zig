pub const protocol = @import("protocol.zig");
pub const rpc_protocol = @import("rpc_protocol.zig");
pub const transport = @import("transport/mod.zig");
pub const connection_pool = @import("connection_pool.zig");

// Re-exports
pub const TaskEnvelope = protocol.TaskEnvelope;
pub const ResultEnvelope = protocol.ResultEnvelope;
pub const ResultStatus = protocol.ResultStatus;
pub const encodeTask = protocol.encodeTask;
pub const decodeTask = protocol.decodeTask;
pub const encodeResult = protocol.encodeResult;
pub const decodeResult = protocol.decodeResult;

pub const RpcMessageType = rpc_protocol.MessageType;
pub const RpcHeader = rpc_protocol.RpcHeader;
pub const ParsedFrame = rpc_protocol.ParsedFrame;
pub const RpcError = rpc_protocol.RpcError;
pub const VectorEntry = rpc_protocol.VectorEntry;
pub const BlockHeader = rpc_protocol.BlockHeader;
pub const frameMessage = rpc_protocol.frameMessage;
pub const parseRpcFrame = rpc_protocol.parseFrame;

pub const TcpTransport = transport.TcpTransport;
pub const TransportConfig = transport.TransportConfig;
pub const TransportError = transport.TransportError;
pub const TransportStats = transport.TcpTransport.TransportStats;
pub const MessageType = transport.MessageType;
pub const MessageHeader = transport.MessageHeader;
pub const PeerConnection = transport.PeerConnection;
pub const RpcSerializer = transport.RpcSerializer;
pub const parseAddress = transport.parseAddress;

pub const ConnectionPool = connection_pool.ConnectionPool;
pub const ConnectionPoolConfig = connection_pool.ConnectionPoolConfig;
pub const PooledConnection = connection_pool.PooledConnection;
pub const ConnectionState = connection_pool.ConnectionState;
pub const ConnectionStats = connection_pool.ConnectionStats;
pub const HostKey = connection_pool.HostKey;
pub const PoolStats = connection_pool.PoolStats;
pub const PoolBuilder = connection_pool.PoolBuilder;
