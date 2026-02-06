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

const std = @import("std");
const platform_time = @import("../../services/shared/time.zig");
const sync = @import("../../services/shared/sync.zig");
const platform_time = @import("../../services/shared/utils.zig");
const connection_pool = @import("connection_pool.zig");
const circuit_breaker = @import("circuit_breaker.zig");
const retry = @import("retry.zig");

/// Transport configuration.
pub const TransportConfig = struct {
    /// Local address to bind to.
    listen_address: []const u8 = "0.0.0.0",
    /// Local port to listen on.
    listen_port: u16 = 9000,
    /// Maximum concurrent connections.
    max_connections: u32 = 256,
    /// Connection timeout in milliseconds.
    connect_timeout_ms: u64 = 5000,
    /// Read/write timeout in milliseconds.
    io_timeout_ms: u64 = 30000,
    /// Maximum message size in bytes.
    max_message_size: usize = 16 * 1024 * 1024, // 16MB
    /// Enable TCP keepalive.
    enable_keepalive: bool = true,
    /// Keepalive interval in seconds.
    keepalive_interval_s: u32 = 30,
    /// Enable connection pooling.
    enable_pooling: bool = true,
    /// Maximum retries for failed requests.
    max_retries: u32 = 3,
    /// Base retry delay in milliseconds.
    retry_delay_ms: u64 = 100,
    /// Enable request/response logging.
    enable_logging: bool = true,
};

/// Message types for the transport protocol.
pub const MessageType = enum(u8) {
    // Raft consensus messages
    raft_vote_request = 1,
    raft_vote_response = 2,
    raft_append_entries = 3,
    raft_append_response = 4,
    raft_install_snapshot = 5,
    raft_snapshot_response = 6,

    // Database RPC messages
    db_search_request = 10,
    db_search_response = 11,
    db_insert_request = 12,
    db_insert_response = 13,
    db_delete_request = 14,
    db_delete_response = 15,
    db_update_request = 16,
    db_update_response = 17,
    db_batch_request = 18,
    db_batch_response = 19,

    // Cluster management messages
    cluster_join = 20,
    cluster_join_ack = 21,
    cluster_leave = 22,
    cluster_leave_ack = 23,
    cluster_heartbeat = 24,
    cluster_heartbeat_ack = 25,
    cluster_state_sync = 26,
    cluster_state_ack = 27,

    // Generic RPC
    rpc_request = 30,
    rpc_response = 31,
    rpc_error = 32,

    // Control messages
    ping = 100,
    pong = 101,
    close = 102,
};

/// Message header for framing.
pub const MessageHeader = extern struct {
    /// Magic number for validation.
    magic: u32 = MAGIC_NUMBER,
    /// Protocol version.
    version: u8 = PROTOCOL_VERSION,
    /// Message type.
    message_type: u8,
    /// Flags (reserved).
    flags: u16 = 0,
    /// Request ID for correlation.
    request_id: u64,
    /// Payload length.
    payload_length: u32,
    /// CRC32 checksum of payload (0 if disabled).
    checksum: u32 = 0,

    pub const SIZE: usize = @sizeOf(MessageHeader);
};

const MAGIC_NUMBER: u32 = 0x41424954; // "ABIT"
const PROTOCOL_VERSION: u8 = 1;

/// Errors for transport operations.
pub const TransportError = error{
    ConnectionFailed,
    ConnectionClosed,
    ConnectionTimeout,
    InvalidMessage,
    InvalidMagic,
    InvalidVersion,
    MessageTooLarge,
    ChecksumMismatch,
    RequestTimeout,
    SendFailed,
    ReceiveFailed,
    PoolExhausted,
    NotConnected,
    AlreadyStarted,
    NotStarted,
    AddressParseError,
    BindFailed,
    ListenFailed,
    AcceptFailed,
    OutOfMemory,
    Cancelled,
};

/// Request state for pending requests.
pub const PendingRequest = struct {
    request_id: u64,
    sent_at_ns: i64,
    timeout_ns: i64,
    response_data: ?[]u8 = null,
    completed: std.atomic.Value(bool),
    error_code: ?TransportError = null,
    condition: std.Thread.Condition,
    mutex: sync.Mutex,

    pub fn init(request_id: u64, timeout_ms: u64) PendingRequest {
        return .{
            .request_id = request_id,
            .sent_at_ns = @intCast(time.nowNanoseconds()),
            .timeout_ns = @intCast(timeout_ms * 1_000_000),
            .completed = std.atomic.Value(bool).init(false),
            .condition = .{},
            .mutex = .{},
        };
    }

    pub fn isTimedOut(self: *const PendingRequest) bool {
        const elapsed: i64 = @intCast(time.nowNanoseconds());
        return (elapsed - self.sent_at_ns) > self.timeout_ns;
    }

    pub fn complete(self: *PendingRequest, response: ?[]u8, err: ?TransportError) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.response_data = response;
        self.error_code = err;
        self.completed.store(true, .release);
        self.condition.signal();
    }

    pub fn waitForCompletion(self: *PendingRequest, timeout_ns: u64) !?[]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        while (!self.completed.load(.acquire)) {
            self.condition.timedWait(&self.mutex, timeout_ns) catch |err| {
                if (err == error.Timeout) {
                    return TransportError.RequestTimeout;
                }
                return TransportError.Cancelled;
            };
        }

        if (self.error_code) |e| {
            return e;
        }

        return self.response_data;
    }
};

/// Connection state for a remote peer.
pub const PeerConnection = struct {
    allocator: std.mem.Allocator,
    address: []const u8,
    port: u16,
    socket_fd: ?std.posix.socket_t = null,
    state: ConnectionState,
    last_activity_ns: i64,
    bytes_sent: u64,
    bytes_received: u64,
    requests_sent: u64,
    requests_failed: u64,
    consecutive_failures: u32,
    mutex: sync.Mutex,

    pub const ConnectionState = enum {
        disconnected,
        connecting,
        connected,
        error_state,
        closing,
    };

    pub fn init(allocator: std.mem.Allocator, address: []const u8, port: u16) !PeerConnection {
        const addr_copy = try allocator.dupe(u8, address);
        return .{
            .allocator = allocator,
            .address = addr_copy,
            .port = port,
            .state = .disconnected,
            .last_activity_ns = @intCast(time.nowNanoseconds()),
            .bytes_sent = 0,
            .bytes_received = 0,
            .requests_sent = 0,
            .requests_failed = 0,
            .consecutive_failures = 0,
            .mutex = .{},
        };
    }

    pub fn deinit(self: *PeerConnection) void {
        self.close();
        self.allocator.free(self.address);
        self.* = undefined;
    }

    pub fn connect(self: *PeerConnection, config: TransportConfig) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.state == .connected) return;

        self.state = .connecting;

        // Parse address
        const addr = std.net.Address.parseIp4(self.address, self.port) catch {
            // Try IPv6
            const addr6 = std.net.Address.parseIp6(self.address, self.port) catch {
                self.state = .error_state;
                return TransportError.AddressParseError;
            };
            return self.connectToAddress(addr6, config);
        };

        return self.connectToAddress(addr, config);
    }

    fn connectToAddress(self: *PeerConnection, addr: std.net.Address, config: TransportConfig) !void {
        const socket = std.posix.socket(
            addr.any.family,
            std.posix.SOCK.STREAM,
            std.posix.IPPROTO.TCP,
        ) catch {
            self.state = .error_state;
            return TransportError.ConnectionFailed;
        };
        errdefer std.posix.close(socket);

        // Set socket options
        if (config.enable_keepalive) {
            std.posix.setsockopt(socket, std.posix.SOL.SOCKET, std.posix.SO.KEEPALIVE, &std.mem.toBytes(@as(c_int, 1))) catch |err| {
                std.log.warn("Transport: Failed to set SO_KEEPALIVE on socket: {}", .{err});
            };
        }

        // Set connect timeout via SO_RCVTIMEO/SO_SNDTIMEO
        const timeout_s = config.connect_timeout_ms / 1000;
        const timeout_us = (config.connect_timeout_ms % 1000) * 1000;
        const timeval = std.posix.timeval{
            .sec = @intCast(timeout_s),
            .usec = @intCast(timeout_us),
        };
        std.posix.setsockopt(socket, std.posix.SOL.SOCKET, std.posix.SO.RCVTIMEO, std.mem.asBytes(&timeval)) catch |err| {
            std.log.warn("Transport: Failed to set SO_RCVTIMEO on socket: {}", .{err});
        };
        std.posix.setsockopt(socket, std.posix.SOL.SOCKET, std.posix.SO.SNDTIMEO, std.mem.asBytes(&timeval)) catch |err| {
            std.log.warn("Transport: Failed to set SO_SNDTIMEO on socket: {}", .{err});
        };

        // Connect
        std.posix.connect(socket, &addr.any, addr.getOsSockLen()) catch {
            std.posix.close(socket);
            self.state = .error_state;
            self.consecutive_failures += 1;
            return TransportError.ConnectionFailed;
        };

        self.socket_fd = socket;
        self.state = .connected;
        self.consecutive_failures = 0;
        self.last_activity_ns = @intCast(time.nowNanoseconds());
    }

    pub fn close(self: *PeerConnection) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.socket_fd) |fd| {
            self.state = .closing;
            std.posix.close(fd);
            self.socket_fd = null;
        }
        self.state = .disconnected;
    }

    pub fn isConnected(self: *const PeerConnection) bool {
        return self.state == .connected and self.socket_fd != null;
    }

    pub fn send(self: *PeerConnection, data: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const socket = self.socket_fd orelse return TransportError.NotConnected;

        var total_sent: usize = 0;
        while (total_sent < data.len) {
            const sent = std.posix.send(socket, data[total_sent..], 0) catch |err| {
                self.requests_failed += 1;
                self.consecutive_failures += 1;
                if (err == error.WouldBlock or err == error.ConnectionResetByPeer) {
                    self.state = .error_state;
                }
                return TransportError.SendFailed;
            };

            if (sent == 0) {
                self.state = .error_state;
                return TransportError.ConnectionClosed;
            }

            total_sent += sent;
        }

        self.bytes_sent += data.len;
        self.last_activity_ns = @intCast(time.nowNanoseconds());
    }

    pub fn receive(self: *PeerConnection, buffer: []u8) !usize {
        self.mutex.lock();
        defer self.mutex.unlock();

        const socket = self.socket_fd orelse return TransportError.NotConnected;

        const received = std.posix.recv(socket, buffer, 0) catch |err| {
            self.consecutive_failures += 1;
            if (err == error.WouldBlock) {
                return TransportError.ConnectionTimeout;
            }
            if (err == error.ConnectionResetByPeer) {
                self.state = .error_state;
            }
            return TransportError.ReceiveFailed;
        };

        if (received == 0) {
            self.state = .disconnected;
            return TransportError.ConnectionClosed;
        }

        self.bytes_received += received;
        self.last_activity_ns = @intCast(time.nowNanoseconds());
        return received;
    }

    pub fn receiveExact(self: *PeerConnection, buffer: []u8) !void {
        var total_received: usize = 0;
        while (total_received < buffer.len) {
            const remaining = buffer[total_received..];
            const received = try self.receive(remaining);
            total_received += received;
        }
    }
};

/// TCP Transport for distributed communication.
pub const TcpTransport = struct {
    allocator: std.mem.Allocator,
    config: TransportConfig,

    // Connection management
    peers: std.StringHashMapUnmanaged(*PeerConnection),
    pool: ?*connection_pool.ConnectionPool,

    // Request tracking
    pending_requests: std.AutoHashMapUnmanaged(u64, *PendingRequest),
    next_request_id: std.atomic.Value(u64),

    // Server state
    listener: ?std.posix.socket_t,
    running: std.atomic.Value(bool),
    accept_thread: ?std.Thread,

    // Message handlers
    handlers: std.AutoHashMapUnmanaged(u8, *const MessageHandler),

    // Synchronization
    mutex: sync.Mutex,

    // Statistics
    stats: TransportStats,

    pub const MessageHandler = fn (
        transport: *TcpTransport,
        peer_addr: []const u8,
        header: *const MessageHeader,
        payload: []const u8,
    ) TransportError![]u8;

    pub const TransportStats = struct {
        connections_active: u32 = 0,
        connections_total: u64 = 0,
        bytes_sent: u64 = 0,
        bytes_received: u64 = 0,
        messages_sent: u64 = 0,
        messages_received: u64 = 0,
        requests_pending: u32 = 0,
        requests_completed: u64 = 0,
        requests_failed: u64 = 0,
        requests_timeout: u64 = 0,
    };

    /// Initialize the transport layer.
    pub fn init(allocator: std.mem.Allocator, config: TransportConfig) !*TcpTransport {
        const transport = try allocator.create(TcpTransport);
        errdefer allocator.destroy(transport);

        var pool: ?*connection_pool.ConnectionPool = null;
        if (config.enable_pooling) {
            pool = try allocator.create(connection_pool.ConnectionPool);
            pool.?.* = connection_pool.ConnectionPool.init(allocator, .{
                .max_connections_per_host = 10,
                .max_total_connections = config.max_connections,
                .connect_timeout_ns = config.connect_timeout_ms * 1_000_000,
                .idle_timeout_ns = 60_000_000_000,
            });
        }

        transport.* = .{
            .allocator = allocator,
            .config = config,
            .peers = .{},
            .pool = pool,
            .pending_requests = .{},
            .next_request_id = std.atomic.Value(u64).init(1),
            .listener = null,
            .running = std.atomic.Value(bool).init(false),
            .accept_thread = null,
            .handlers = .{},
            .mutex = .{},
            .stats = .{},
        };

        return transport;
    }

    /// Clean up the transport layer.
    pub fn deinit(self: *TcpTransport) void {
        self.stop();

        // Clean up peers
        var peer_iter = self.peers.valueIterator();
        while (peer_iter.next()) |peer| {
            peer.*.deinit();
            self.allocator.destroy(peer.*);
        }
        self.peers.deinit(self.allocator);

        // Clean up pending requests
        var req_iter = self.pending_requests.valueIterator();
        while (req_iter.next()) |req| {
            if (req.*.response_data) |data| {
                self.allocator.free(data);
            }
            self.allocator.destroy(req.*);
        }
        self.pending_requests.deinit(self.allocator);

        // Clean up handlers
        self.handlers.deinit(self.allocator);

        // Clean up pool
        if (self.pool) |pool| {
            pool.deinit();
            self.allocator.destroy(pool);
        }

        self.allocator.destroy(self);
    }

    /// Start the transport server.
    pub fn start(self: *TcpTransport) !void {
        if (self.running.load(.acquire)) {
            return TransportError.AlreadyStarted;
        }

        // Create listener socket
        const addr = std.net.Address.parseIp4(
            self.config.listen_address,
            self.config.listen_port,
        ) catch {
            return TransportError.AddressParseError;
        };

        const listener = std.posix.socket(
            addr.any.family,
            std.posix.SOCK.STREAM,
            std.posix.IPPROTO.TCP,
        ) catch {
            return TransportError.BindFailed;
        };
        errdefer std.posix.close(listener);

        // Allow address reuse
        std.posix.setsockopt(
            listener,
            std.posix.SOL.SOCKET,
            std.posix.SO.REUSEADDR,
            &std.mem.toBytes(@as(c_int, 1)),
        ) catch |err| {
            std.log.warn("Transport: Failed to set SO_REUSEADDR on listener socket: {}", .{err});
        };

        // Bind
        std.posix.bind(listener, &addr.any, addr.getOsSockLen()) catch {
            return TransportError.BindFailed;
        };

        // Listen
        std.posix.listen(listener, 128) catch {
            return TransportError.ListenFailed;
        };

        self.listener = listener;
        self.running.store(true, .release);

        // Start accept thread
        self.accept_thread = std.Thread.spawn(.{}, acceptLoop, .{self}) catch {
            self.running.store(false, .release);
            std.posix.close(listener);
            self.listener = null;
            return TransportError.ListenFailed;
        };

        if (self.config.enable_logging) {
            std.log.info("Transport: Listening on {s}:{d}", .{
                self.config.listen_address,
                self.config.listen_port,
            });
        }
    }

    /// Stop the transport server.
    pub fn stop(self: *TcpTransport) void {
        if (!self.running.load(.acquire)) return;

        self.running.store(false, .release);

        // Close listener
        if (self.listener) |listener| {
            std.posix.close(listener);
            self.listener = null;
        }

        // Wait for accept thread
        if (self.accept_thread) |thread| {
            thread.join();
            self.accept_thread = null;
        }

        // Close all peer connections
        var peer_iter = self.peers.valueIterator();
        while (peer_iter.next()) |peer| {
            peer.*.close();
        }
    }

    /// Register a message handler.
    pub fn registerHandler(
        self: *TcpTransport,
        message_type: MessageType,
        handler: *const MessageHandler,
    ) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        try self.handlers.put(self.allocator, @intFromEnum(message_type), handler);
    }

    /// Get or create a connection to a peer.
    pub fn getOrCreatePeer(self: *TcpTransport, address: []const u8, port: u16) !*PeerConnection {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Create address key
        var key_buf: [280]u8 = undefined;
        const key = std.fmt.bufPrint(&key_buf, "{s}:{d}", .{ address, port }) catch {
            return TransportError.AddressParseError;
        };

        // Check if peer exists
        if (self.peers.get(key)) |peer| {
            return peer;
        }

        // Create new peer
        const peer = try self.allocator.create(PeerConnection);
        errdefer self.allocator.destroy(peer);

        peer.* = try PeerConnection.init(self.allocator, address, port);

        // Store with copied key
        const key_copy = try self.allocator.dupe(u8, key);
        try self.peers.put(self.allocator, key_copy, peer);

        self.stats.connections_total += 1;
        self.stats.connections_active += 1;

        return peer;
    }

    /// Send a request and wait for response.
    pub fn sendRequest(
        self: *TcpTransport,
        address: []const u8,
        port: u16,
        message_type: MessageType,
        payload: []const u8,
    ) ![]u8 {
        // Get or create peer connection
        const peer = try self.getOrCreatePeer(address, port);

        // Connect if needed
        if (!peer.isConnected()) {
            try peer.connect(self.config);
        }

        // Generate request ID
        const request_id = self.next_request_id.fetchAdd(1, .monotonic);

        // Create pending request
        const pending = try self.allocator.create(PendingRequest);
        pending.* = PendingRequest.init(request_id, self.config.io_timeout_ms);

        {
            self.mutex.lock();
            defer self.mutex.unlock();
            try self.pending_requests.put(self.allocator, request_id, pending);
        }
        errdefer {
            self.mutex.lock();
            defer self.mutex.unlock();
            _ = self.pending_requests.remove(request_id);
            self.allocator.destroy(pending);
        }

        // Build message
        const header = MessageHeader{
            .message_type = @intFromEnum(message_type),
            .request_id = request_id,
            .payload_length = @intCast(payload.len),
            .checksum = std.hash.crc.Crc32.hash(payload),
        };

        // Send header + payload
        const header_bytes = std.mem.asBytes(&header);
        try peer.send(header_bytes);
        if (payload.len > 0) {
            try peer.send(payload);
        }

        self.stats.messages_sent += 1;
        self.stats.bytes_sent += MessageHeader.SIZE + payload.len;
        peer.requests_sent += 1;

        // Wait for response
        const response = pending.waitForCompletion(self.config.io_timeout_ms * 1_000_000) catch |err| {
            self.stats.requests_failed += 1;
            if (err == TransportError.RequestTimeout) {
                self.stats.requests_timeout += 1;
            }
            return err;
        };

        // Clean up pending request
        {
            self.mutex.lock();
            defer self.mutex.unlock();
            _ = self.pending_requests.remove(request_id);
        }
        defer self.allocator.destroy(pending);

        if (response) |data| {
            self.stats.requests_completed += 1;
            return data;
        }

        return TransportError.ReceiveFailed;
    }

    /// Send a one-way message (no response expected).
    pub fn sendMessage(
        self: *TcpTransport,
        address: []const u8,
        port: u16,
        message_type: MessageType,
        payload: []const u8,
    ) !void {
        const peer = try self.getOrCreatePeer(address, port);

        if (!peer.isConnected()) {
            try peer.connect(self.config);
        }

        const request_id = self.next_request_id.fetchAdd(1, .monotonic);

        const header = MessageHeader{
            .message_type = @intFromEnum(message_type),
            .request_id = request_id,
            .payload_length = @intCast(payload.len),
            .checksum = std.hash.crc.Crc32.hash(payload),
        };

        const header_bytes = std.mem.asBytes(&header);
        try peer.send(header_bytes);
        if (payload.len > 0) {
            try peer.send(payload);
        }

        self.stats.messages_sent += 1;
        self.stats.bytes_sent += MessageHeader.SIZE + payload.len;
    }

    /// Send a response to a request.
    pub fn sendResponse(
        self: *TcpTransport,
        peer: *PeerConnection,
        request_id: u64,
        message_type: MessageType,
        payload: []const u8,
    ) !void {
        const header = MessageHeader{
            .message_type = @intFromEnum(message_type),
            .request_id = request_id,
            .payload_length = @intCast(payload.len),
            .checksum = std.hash.crc.Crc32.hash(payload),
        };

        const header_bytes = std.mem.asBytes(&header);
        try peer.send(header_bytes);
        if (payload.len > 0) {
            try peer.send(payload);
        }

        self.stats.messages_sent += 1;
        self.stats.bytes_sent += MessageHeader.SIZE + payload.len;
    }

    /// Get transport statistics.
    pub fn getStats(self: *TcpTransport) TransportStats {
        self.mutex.lock();
        defer self.mutex.unlock();

        var stats = self.stats;
        stats.requests_pending = @intCast(self.pending_requests.count());
        return stats;
    }

    /// Check if transport is running.
    pub fn isRunning(self: *TcpTransport) bool {
        return self.running.load(.acquire);
    }

    // Internal: Accept loop for incoming connections
    fn acceptLoop(self: *TcpTransport) void {
        const listener = self.listener orelse return;

        while (self.running.load(.acquire)) {
            var client_addr: std.net.Address = undefined;
            var addr_len: std.posix.socklen_t = @sizeOf(std.net.Address);

            const client_fd = std.posix.accept(
                listener,
                &client_addr.any,
                &addr_len,
                0,
            ) catch |err| {
                if (err == error.SocketNotListening or !self.running.load(.acquire)) {
                    break;
                }
                continue;
            };

            // Spawn handler thread for this connection
            _ = std.Thread.spawn(.{}, handleConnection, .{ self, client_fd, client_addr }) catch {
                std.posix.close(client_fd);
                continue;
            };
        }
    }

    // Internal: Handle a single client connection
    fn handleConnection(self: *TcpTransport, client_fd: std.posix.socket_t, addr: std.net.Address) void {
        defer std.posix.close(client_fd);

        var addr_buf: [64]u8 = undefined;
        const peer_addr = std.fmt.bufPrint(&addr_buf, "{}", .{addr}) catch "unknown";

        if (self.config.enable_logging) {
            std.log.debug("Transport: Connection from {s}", .{peer_addr});
        }

        var header_buf: [MessageHeader.SIZE]u8 = undefined;

        while (self.running.load(.acquire)) {
            // Read header
            var total_read: usize = 0;
            while (total_read < MessageHeader.SIZE) {
                const n = std.posix.recv(
                    client_fd,
                    header_buf[total_read..],
                    0,
                ) catch break;

                if (n == 0) break; // Connection closed
                total_read += n;
            }

            if (total_read < MessageHeader.SIZE) break;

            const header: *const MessageHeader = @ptrCast(@alignCast(&header_buf));

            // Validate header
            if (header.magic != MAGIC_NUMBER) {
                if (self.config.enable_logging) {
                    std.log.warn("Transport: Invalid magic from {s}", .{peer_addr});
                }
                break;
            }

            if (header.version != PROTOCOL_VERSION) {
                if (self.config.enable_logging) {
                    std.log.warn("Transport: Invalid version from {s}", .{peer_addr});
                }
                break;
            }

            if (header.payload_length > self.config.max_message_size) {
                if (self.config.enable_logging) {
                    std.log.warn("Transport: Message too large from {s}", .{peer_addr});
                }
                break;
            }

            // Read payload
            const payload = self.allocator.alloc(u8, header.payload_length) catch break;
            defer self.allocator.free(payload);

            if (header.payload_length > 0) {
                var payload_read: usize = 0;
                while (payload_read < header.payload_length) {
                    const n = std.posix.recv(
                        client_fd,
                        payload[payload_read..],
                        0,
                    ) catch break;

                    if (n == 0) break;
                    payload_read += n;
                }

                if (payload_read < header.payload_length) break;

                // Verify checksum
                if (header.checksum != 0) {
                    const computed = std.hash.crc.Crc32.hash(payload);
                    if (computed != header.checksum) {
                        if (self.config.enable_logging) {
                            std.log.warn("Transport: Checksum mismatch from {s}", .{peer_addr});
                        }
                        break;
                    }
                }
            }

            self.stats.messages_received += 1;
            self.stats.bytes_received += MessageHeader.SIZE + header.payload_length;

            // Check if this is a response to a pending request
            if (self.isResponseMessage(header.message_type)) {
                self.handleResponse(header, payload);
            } else {
                // Handle as incoming request
                self.handleRequest(client_fd, peer_addr, header, payload);
            }
        }

        if (self.config.enable_logging) {
            std.log.debug("Transport: Disconnected from {s}", .{peer_addr});
        }
    }

    fn isResponseMessage(self: *TcpTransport, msg_type: u8) bool {
        _ = self;
        return msg_type == @intFromEnum(MessageType.raft_vote_response) or
            msg_type == @intFromEnum(MessageType.raft_append_response) or
            msg_type == @intFromEnum(MessageType.raft_snapshot_response) or
            msg_type == @intFromEnum(MessageType.db_search_response) or
            msg_type == @intFromEnum(MessageType.db_insert_response) or
            msg_type == @intFromEnum(MessageType.db_delete_response) or
            msg_type == @intFromEnum(MessageType.db_update_response) or
            msg_type == @intFromEnum(MessageType.db_batch_response) or
            msg_type == @intFromEnum(MessageType.cluster_join_ack) or
            msg_type == @intFromEnum(MessageType.cluster_leave_ack) or
            msg_type == @intFromEnum(MessageType.cluster_heartbeat_ack) or
            msg_type == @intFromEnum(MessageType.cluster_state_ack) or
            msg_type == @intFromEnum(MessageType.rpc_response) or
            msg_type == @intFromEnum(MessageType.rpc_error) or
            msg_type == @intFromEnum(MessageType.pong);
    }

    fn handleResponse(self: *TcpTransport, header: *const MessageHeader, payload: []const u8) void {
        self.mutex.lock();
        const pending = self.pending_requests.get(header.request_id);
        self.mutex.unlock();

        if (pending) |req| {
            // Copy payload for the waiter
            const response_copy = self.allocator.dupe(u8, payload) catch {
                req.complete(null, TransportError.OutOfMemory);
                return;
            };
            req.complete(response_copy, null);
        }
    }

    fn handleRequest(
        self: *TcpTransport,
        client_fd: std.posix.socket_t,
        peer_addr: []const u8,
        header: *const MessageHeader,
        payload: []const u8,
    ) void {
        // Look up handler
        const handler = self.handlers.get(header.message_type) orelse {
            if (self.config.enable_logging) {
                std.log.warn("Transport: No handler for message type {d}", .{header.message_type});
            }
            return;
        };

        // Call handler
        const response = handler(self, peer_addr, header, payload) catch |err| {
            if (self.config.enable_logging) {
                std.log.warn("Transport: Handler error: {}", .{err});
            }
            return;
        };
        defer self.allocator.free(response);

        // Send response
        const response_type = self.getResponseType(header.message_type);
        const resp_header = MessageHeader{
            .message_type = response_type,
            .request_id = header.request_id,
            .payload_length = @intCast(response.len),
            .checksum = std.hash.crc.Crc32.hash(response),
        };

        const header_bytes = std.mem.asBytes(&resp_header);
        _ = std.posix.send(client_fd, header_bytes, 0) catch return;
        if (response.len > 0) {
            _ = std.posix.send(client_fd, response, 0) catch return;
        }
    }

    fn getResponseType(self: *TcpTransport, request_type: u8) u8 {
        _ = self;
        return switch (request_type) {
            @intFromEnum(MessageType.raft_vote_request) => @intFromEnum(MessageType.raft_vote_response),
            @intFromEnum(MessageType.raft_append_entries) => @intFromEnum(MessageType.raft_append_response),
            @intFromEnum(MessageType.raft_install_snapshot) => @intFromEnum(MessageType.raft_snapshot_response),
            @intFromEnum(MessageType.db_search_request) => @intFromEnum(MessageType.db_search_response),
            @intFromEnum(MessageType.db_insert_request) => @intFromEnum(MessageType.db_insert_response),
            @intFromEnum(MessageType.db_delete_request) => @intFromEnum(MessageType.db_delete_response),
            @intFromEnum(MessageType.db_update_request) => @intFromEnum(MessageType.db_update_response),
            @intFromEnum(MessageType.db_batch_request) => @intFromEnum(MessageType.db_batch_response),
            @intFromEnum(MessageType.cluster_join) => @intFromEnum(MessageType.cluster_join_ack),
            @intFromEnum(MessageType.cluster_leave) => @intFromEnum(MessageType.cluster_leave_ack),
            @intFromEnum(MessageType.cluster_heartbeat) => @intFromEnum(MessageType.cluster_heartbeat_ack),
            @intFromEnum(MessageType.cluster_state_sync) => @intFromEnum(MessageType.cluster_state_ack),
            @intFromEnum(MessageType.rpc_request) => @intFromEnum(MessageType.rpc_response),
            @intFromEnum(MessageType.ping) => @intFromEnum(MessageType.pong),
            else => @intFromEnum(MessageType.rpc_response),
        };
    }
};

/// RPC request/response serialization helpers.
pub const RpcSerializer = struct {
    /// Serialize a Raft RequestVote request.
    pub fn serializeVoteRequest(
        allocator: std.mem.Allocator,
        term: u64,
        candidate_id: []const u8,
        last_log_index: u64,
        last_log_term: u64,
    ) ![]u8 {
        const header_size = 8 + 4 + 8 + 8; // term + id_len + last_log_index + last_log_term
        const total_size = header_size + candidate_id.len;

        var buffer = try allocator.alloc(u8, total_size);
        var offset: usize = 0;

        std.mem.writeInt(u64, buffer[offset..][0..8], term, .little);
        offset += 8;
        std.mem.writeInt(u32, buffer[offset..][0..4], @intCast(candidate_id.len), .little);
        offset += 4;
        std.mem.writeInt(u64, buffer[offset..][0..8], last_log_index, .little);
        offset += 8;
        std.mem.writeInt(u64, buffer[offset..][0..8], last_log_term, .little);
        offset += 8;
        @memcpy(buffer[offset..][0..candidate_id.len], candidate_id);

        return buffer;
    }

    /// Deserialize a Raft RequestVote request.
    pub fn deserializeVoteRequest(data: []const u8) !struct {
        term: u64,
        candidate_id: []const u8,
        last_log_index: u64,
        last_log_term: u64,
    } {
        if (data.len < 28) return TransportError.InvalidMessage;

        var offset: usize = 0;
        const term = std.mem.readInt(u64, data[offset..][0..8], .little);
        offset += 8;
        const id_len = std.mem.readInt(u32, data[offset..][0..4], .little);
        offset += 4;
        const last_log_index = std.mem.readInt(u64, data[offset..][0..8], .little);
        offset += 8;
        const last_log_term = std.mem.readInt(u64, data[offset..][0..8], .little);
        offset += 8;

        if (offset + id_len > data.len) return TransportError.InvalidMessage;
        const candidate_id = data[offset..][0..id_len];

        return .{
            .term = term,
            .candidate_id = candidate_id,
            .last_log_index = last_log_index,
            .last_log_term = last_log_term,
        };
    }

    /// Serialize a Raft RequestVote response.
    pub fn serializeVoteResponse(
        allocator: std.mem.Allocator,
        term: u64,
        vote_granted: bool,
        voter_id: []const u8,
    ) ![]u8 {
        const header_size = 8 + 1 + 4; // term + vote_granted + id_len
        const total_size = header_size + voter_id.len;

        var buffer = try allocator.alloc(u8, total_size);
        var offset: usize = 0;

        std.mem.writeInt(u64, buffer[offset..][0..8], term, .little);
        offset += 8;
        buffer[offset] = if (vote_granted) 1 else 0;
        offset += 1;
        std.mem.writeInt(u32, buffer[offset..][0..4], @intCast(voter_id.len), .little);
        offset += 4;
        @memcpy(buffer[offset..][0..voter_id.len], voter_id);

        return buffer;
    }

    /// Serialize a database search request.
    pub fn serializeSearchRequest(
        allocator: std.mem.Allocator,
        query_vector: []const f32,
        top_k: u32,
        shard_id: u32,
    ) ![]u8 {
        const header_size = 4 + 4 + 4; // vector_len + top_k + shard_id
        const vector_bytes = query_vector.len * @sizeOf(f32);
        const total_size = header_size + vector_bytes;

        var buffer = try allocator.alloc(u8, total_size);
        var offset: usize = 0;

        std.mem.writeInt(u32, buffer[offset..][0..4], @intCast(query_vector.len), .little);
        offset += 4;
        std.mem.writeInt(u32, buffer[offset..][0..4], top_k, .little);
        offset += 4;
        std.mem.writeInt(u32, buffer[offset..][0..4], shard_id, .little);
        offset += 4;

        const vector_slice = std.mem.sliceAsBytes(query_vector);
        @memcpy(buffer[offset..][0..vector_bytes], vector_slice);

        return buffer;
    }

    /// Serialize a database insert request.
    pub fn serializeInsertRequest(
        allocator: std.mem.Allocator,
        id: u64,
        vector: []const f32,
        metadata: ?[]const u8,
    ) ![]u8 {
        const meta = metadata orelse "";
        const header_size = 8 + 4 + 4; // id + vector_len + meta_len
        const vector_bytes = vector.len * @sizeOf(f32);
        const total_size = header_size + vector_bytes + meta.len;

        var buffer = try allocator.alloc(u8, total_size);
        var offset: usize = 0;

        std.mem.writeInt(u64, buffer[offset..][0..8], id, .little);
        offset += 8;
        std.mem.writeInt(u32, buffer[offset..][0..4], @intCast(vector.len), .little);
        offset += 4;
        std.mem.writeInt(u32, buffer[offset..][0..4], @intCast(meta.len), .little);
        offset += 4;

        const vector_slice = std.mem.sliceAsBytes(vector);
        @memcpy(buffer[offset..][0..vector_bytes], vector_slice);
        offset += vector_bytes;

        if (meta.len > 0) {
            @memcpy(buffer[offset..][0..meta.len], meta);
        }

        return buffer;
    }
};

/// Address parsing utilities.
pub fn parseAddress(address: []const u8) !struct { host: []const u8, port: u16 } {
    // Handle IPv6 with brackets: [::1]:8080
    if (address[0] == '[') {
        const bracket_end = std.mem.indexOf(u8, address, "]") orelse return TransportError.AddressParseError;
        const host = address[1..bracket_end];

        if (address.len > bracket_end + 1 and address[bracket_end + 1] == ':') {
            const port_str = address[bracket_end + 2 ..];
            const port = std.fmt.parseInt(u16, port_str, 10) catch return TransportError.AddressParseError;
            return .{ .host = host, .port = port };
        }

        return .{ .host = host, .port = 9000 };
    }

    // Handle IPv4: 192.168.1.1:8080
    if (std.mem.lastIndexOf(u8, address, ":")) |colon_pos| {
        const host = address[0..colon_pos];
        const port_str = address[colon_pos + 1 ..];
        const port = std.fmt.parseInt(u16, port_str, 10) catch return TransportError.AddressParseError;
        return .{ .host = host, .port = port };
    }

    return .{ .host = address, .port = 9000 };
}

test "message header size" {
    try std.testing.expectEqual(@as(usize, 24), MessageHeader.SIZE);
}

test "address parsing ipv4" {
    const result = try parseAddress("192.168.1.1:8080");
    try std.testing.expectEqualStrings("192.168.1.1", result.host);
    try std.testing.expectEqual(@as(u16, 8080), result.port);
}

test "address parsing ipv4 default port" {
    const result = try parseAddress("192.168.1.1");
    try std.testing.expectEqualStrings("192.168.1.1", result.host);
    try std.testing.expectEqual(@as(u16, 9000), result.port);
}

test "rpc serialization vote request" {
    const allocator = std.testing.allocator;

    const data = try RpcSerializer.serializeVoteRequest(allocator, 5, "node-1", 10, 3);
    defer allocator.free(data);

    const parsed = try RpcSerializer.deserializeVoteRequest(data);
    try std.testing.expectEqual(@as(u64, 5), parsed.term);
    try std.testing.expectEqualStrings("node-1", parsed.candidate_id);
    try std.testing.expectEqual(@as(u64, 10), parsed.last_log_index);
    try std.testing.expectEqual(@as(u64, 3), parsed.last_log_term);
}

test "peer connection init and deinit" {
    const allocator = std.testing.allocator;

    var peer = try PeerConnection.init(allocator, "127.0.0.1", 9000);
    defer peer.deinit();

    try std.testing.expectEqualStrings("127.0.0.1", peer.address);
    try std.testing.expectEqual(@as(u16, 9000), peer.port);
    try std.testing.expect(!peer.isConnected());
}
