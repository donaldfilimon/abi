//! TCP Transport
//!
//! TcpTransport provides TCP client and server with connection pooling,
//! message framing, and request/response correlation.

const std = @import("std");
const protocol = @import("protocol.zig");
const connection = @import("connection.zig");
const connection_pool = @import("../connection_pool.zig");
const sync = @import("../../../foundation/mod.zig").sync;

const NetworkAddress = protocol.NetworkAddress;
const TransportConfig = protocol.TransportConfig;
const MessageType = protocol.MessageType;
const MessageHeader = protocol.MessageHeader;
const TransportError = protocol.TransportError;
const MAGIC_NUMBER = protocol.MAGIC_NUMBER;
const PROTOCOL_VERSION = protocol.PROTOCOL_VERSION;
const PeerConnection = connection.PeerConnection;
const PendingRequest = connection.PendingRequest;

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
        const addr = NetworkAddress.parseIp4(
            self.config.listen_address,
            self.config.listen_port,
        ) catch {
            return TransportError.AddressParseError;
        };

        const sock = std.c.socket(addr.family(), std.c.SOCK.STREAM, 0);
        if (sock < 0) {
            return TransportError.BindFailed;
        }
        const listener: std.posix.fd_t = @intCast(sock);
        errdefer _ = std.c.close(listener);

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
            _ = std.c.close(listener);
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
            _ = std.c.close(listener);
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
        errdefer self.allocator.free(key_copy);
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
        var header_bytes: [MessageHeader.SIZE]u8 = undefined;
        try header.encode(&header_bytes);
        try peer.send(&header_bytes);
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

        var header_bytes: [MessageHeader.SIZE]u8 = undefined;
        try header.encode(&header_bytes);
        try peer.send(&header_bytes);
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

        var header_bytes: [MessageHeader.SIZE]u8 = undefined;
        try header.encode(&header_bytes);
        try peer.send(&header_bytes);
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
            var client_addr: std.posix.sockaddr = undefined;
            var addr_len: std.posix.socklen_t = @sizeOf(std.posix.sockaddr);

            const client_fd = std.posix.accept(
                listener,
                &client_addr,
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
                _ = std.c.close(client_fd);
                continue;
            };
        }
    }

    // Internal: Handle a single client connection
    fn handleConnection(self: *TcpTransport, client_fd: std.posix.socket_t, addr: std.posix.sockaddr) void {
        defer _ = std.c.close(client_fd);

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

            const header = MessageHeader.decode(&header_buf) catch break;
            header.validate(self.config.max_message_size) catch |err| {
                if (self.config.enable_logging) {
                    switch (err) {
                        error.InvalidMagic => std.log.warn("Transport: Invalid magic from {s}", .{peer_addr}),
                        error.InvalidVersion => std.log.warn("Transport: Invalid version from {s}", .{peer_addr}),
                        error.MessageTooLarge => std.log.warn("Transport: Message too large from {s}", .{peer_addr}),
                        else => std.log.warn("Transport: Invalid header from {s}: {}", .{ peer_addr, err }),
                    }
                }
                break;
            };

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
                self.handleResponse(&header, payload);
            } else {
                // Handle as incoming request
                self.handleRequest(client_fd, peer_addr, &header, payload);
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

        var header_bytes: [MessageHeader.SIZE]u8 = undefined;
        resp_header.encode(&header_bytes) catch return;
        _ = std.posix.send(client_fd, &header_bytes, 0) catch return;
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
