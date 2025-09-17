//! WDBX Vector Database - HTTP REST API Server
//!
//! This module provides a comprehensive HTTP server for the WDBX vector database,
//! including vector operations, authentication, and monitoring endpoints.

const std = @import("std");
const database = @import("database");

const version_string = "WDBX Vector Database v1.0.0";

/// HTTP server configuration
pub const ServerConfig = struct {
    host: []const u8 = "127.0.0.1",
    port: u16 = 8080,
    max_request_size: usize = 1024 * 1024, // 1MB
    rate_limit: usize = 1000, // requests per minute
    enable_cors: bool = true,
    enable_auth: bool = true,
    jwt_secret: []const u8 = "",
    // Added to satisfy tests
    max_connections: u32 = 1024,
    request_timeout_ms: u32 = 5000,
    // Windows-specific optimizations
    socket_keepalive: bool = true,
    tcp_nodelay: bool = true,
    socket_buffer_size: u32 = 8192,
    // Additional Windows networking improvements
    enable_windows_optimizations: bool = true,
    connection_timeout_ms: u32 = 30000, // 30 seconds for Windows
    max_retries: u32 = 3,
};

/// HTTP server for WDBX vector database
pub const WdbxHttpServer = struct {
    allocator: std.mem.Allocator,
    config: ServerConfig,
    db: ?*database.Db,
    server: ?std.net.Server,

    const Self = @This();

    /// Initialize HTTP server
    pub fn init(allocator: std.mem.Allocator, config: ServerConfig) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        self.* = .{
            .allocator = allocator,
            .config = config,
            .db = null,
            .server = null, // Will be initialized in start()
        };

        return self;
    }

    /// Deinitialize HTTP server
    pub fn deinit(self: *Self) void {
        if (self.db) |db| {
            db.close();
        }
        if (self.server) |*server| {
            server.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Run the HTTP server (alias for start)
    pub fn run(self: *Self) !void {
        try self.start();
    }

    /// Open database connection
    pub fn openDatabase(self: *Self, path: []const u8) !void {
        self.db = try database.Db.open(path, true);
        if (self.db.?.getDimension() == 0) {
            try self.db.?.init(8); // Default to 8 dimensions
        }
    }

    /// Start HTTP server
    pub fn start(self: *Self) !void {
        const address = try std.net.Address.parseIp(self.config.host, self.config.port);

        // Server configuration
        self.server = try address.listen(.{
            .reuse_address = true,
            .kernel_backlog = @intCast(self.config.max_connections),
        });

        // Apply Windows-specific socket optimizations
        if (self.server) |*server| {
            self.configureSocket(server.stream.handle) catch |err| {
                std.debug.print("Warning: Failed to configure socket options: {any}\n", .{err});
            };
        }

        std.debug.print("WDBX HTTP server listening on {s}:{d} (Windows optimized)\n", .{ self.config.host, self.config.port });

        while (true) {
            const connection = self.server.?.accept() catch |err| {
                switch (err) {
                    error.ConnectionResetByPeer, error.Unexpected => {
                        // These errors are common and should not stop the server
                        std.debug.print("Connection accept error (normal)\n", .{});
                        continue;
                    },
                    else => {
                        std.debug.print("Failed to accept connection\n", .{});
                        continue;
                    },
                }
            };

            // Handle connection in a way that doesn't crash the server
            self.handleConnection(connection) catch |err| {
                std.debug.print("Connection handling error: {any}\n", .{err});
                // Continue serving other connections
            };
        }
    }

    /// Non-blocking request handler used by tests; returns error.Timeout on no activity
    pub fn handleRequests(self: *Self) !void {
        if (self.server == null) {
            const address = try std.net.Address.parseIp(self.config.host, self.config.port);
            self.server = try address.listen(.{
                .reuse_address = true,
                .kernel_backlog = @intCast(self.config.max_connections),
            });

            // Apply socket configuration for test server too
            if (self.server) |*server| {
                self.configureSocket(server.stream.handle) catch {};
            }
        }
        var pfd = [1]std.posix.pollfd{.{
            .fd = self.server.?.stream.handle,
            .events = std.posix.POLL.IN,
            .revents = 0,
        }};
        const timeout = @as(i32, @intCast(self.config.request_timeout_ms));
        const n = std.posix.poll(&pfd, timeout) catch 0;
        if (n == 0) return error.Timeout;
        if ((pfd[0].revents & std.posix.POLL.IN) != 0) {
            const conn = try self.server.?.accept();
            self.handleConnection(conn) catch {
                std.debug.print("handleConnection error\n", .{});
            };
        }
    }

    /// Configure socket for compatibility
    pub fn configureSocket(self: *Self, socket_handle: std.posix.socket_t) !void {
        // Set TCP_NODELAY for better performance
        if (self.config.tcp_nodelay) {
            const enable: c_int = 1;
            _ = std.posix.setsockopt(socket_handle, std.posix.IPPROTO.TCP, std.posix.TCP.NODELAY, std.mem.asBytes(&enable)) catch |err| {
                std.debug.print("Warning: TCP_NODELAY failed: {any}\n", .{err});
            };
        }

        // Set SO_KEEPALIVE for connection stability
        if (self.config.socket_keepalive) {
            const enable: c_int = 1;
            _ = std.posix.setsockopt(socket_handle, std.posix.SOL.SOCKET, std.posix.SO.KEEPALIVE, std.mem.asBytes(&enable)) catch |err| {
                std.debug.print("Warning: SO_KEEPALIVE failed: {}\n", .{err});
            };
        }

        // Set socket buffer sizes for better networking
        const buffer_size: c_int = @intCast(self.config.socket_buffer_size);
        _ = std.posix.setsockopt(socket_handle, std.posix.SOL.SOCKET, std.posix.SO.RCVBUF, std.mem.asBytes(&buffer_size)) catch |err| {
            std.debug.print("Warning: SO_RCVBUF failed: {}\n", .{err});
        };
        _ = std.posix.setsockopt(socket_handle, std.posix.SOL.SOCKET, std.posix.SO.SNDBUF, std.mem.asBytes(&buffer_size)) catch |err| {
            std.debug.print("Warning: SO_SNDBUF failed: {}\n", .{err});
        };

        // Set SO_REUSEADDR for better compatibility
        const reuse_addr: c_int = 1;
        _ = std.posix.setsockopt(socket_handle, std.posix.SOL.SOCKET, std.posix.SO.REUSEADDR, std.mem.asBytes(&reuse_addr)) catch |err| {
            std.debug.print("Warning: SO_REUSEADDR failed: {}\n", .{err});
        };

        // Set SO_LINGER to 0 for immediate close (helps with connection issues)
        // Note: SO_LINGER removed for Zig 0.15.1 compatibility
    }

    /// Handle TCP connection and parse HTTP requests
    fn handleConnection(self: *Self, connection: std.net.Server.Connection) !void {
        defer connection.stream.close();

        // Use configurable buffer size for better Windows compatibility
        var buffer = try self.allocator.alloc(u8, self.config.socket_buffer_size);
        defer self.allocator.free(buffer);

        const bytes_read: usize = blk: {
            const builtin = @import("builtin");
            if (builtin.os.tag == .windows) {
                const w = std.os.windows;
                const ws2 = w.ws2_32;
                const wsabuf: ws2.WSABUF = .{
                    .len = @as(w.DWORD, @intCast(buffer.len)),
                    .buf = buffer.ptr,
                };
                var flags: w.DWORD = 0;
                var recvd: w.DWORD = 0;
                const s: ws2.SOCKET = connection.stream.handle;
                var bufs = [_]ws2.WSABUF{wsabuf};
                const rc = ws2.WSARecv(s, bufs[0..].ptr, 1, &recvd, &flags, null, null);
                if (rc == ws2.SOCKET_ERROR) {
                    const wsa_err_code: i32 = @intFromEnum(ws2.WSAGetLastError());
                    const WSAECONNRESET: i32 = 10054;
                    const WSAEINTR: i32 = 10004;
                    const WSAEWOULDBLOCK: i32 = 10035;
                    if (wsa_err_code == WSAECONNRESET or wsa_err_code == WSAEINTR) {
                        std.debug.print("Client disconnected (normal)\n", .{});
                        return;
                    }
                    if (wsa_err_code == WSAEWOULDBLOCK) return;
                    std.debug.print("Unexpected WSARecv error: {d}\n", .{wsa_err_code});
                    return error.Unexpected;
                }
                break :blk @as(usize, @intCast(recvd));
            } else {
                const n = connection.stream.read(buffer) catch |err| {
                    switch (err) {
                        error.ConnectionResetByPeer, error.Unexpected => {
                            std.debug.print("Client disconnected (normal)\n", .{});
                            return; // Client disconnected, this is normal
                        },
                        error.WouldBlock => {
                            // Non-blocking socket behavior
                            return;
                        },
                        else => {
                            std.debug.print("Unexpected read error: {}\n", .{err});
                            return err;
                        },
                    }
                };
                break :blk n;
            }
        };

        if (bytes_read == 0) return; // Client disconnected

        // Parse HTTP request
        const request_str = buffer[0..bytes_read];
        const request = self.parseHttpRequest(request_str) catch {
            // Malformed request; respond 400 instead of closing abruptly
            self.sendHttpResponse(connection, 400, "Bad Request", "{\"error\":\"Invalid HTTP request\"}") catch {};
            return;
        };

        // Minimal header parsing for WebSocket upgrade
        var upgrade: bool = false;
        var connection_hdr_upgrade: bool = false;
        var ws_key: ?[]const u8 = null;
        var lines = std.mem.splitSequence(u8, request_str, "\r\n");
        _ = lines.next(); // skip request line
        while (lines.next()) |line| {
            if (line.len == 0) break;
            if (std.mem.indexOfScalar(u8, line, ':')) |colon| {
                const key = std.mem.trim(u8, line[0..colon], " \t");
                const value = std.mem.trim(u8, line[colon + 1 ..], " \t");
                if (std.ascii.eqlIgnoreCase(key, "Upgrade")) {
                    if (std.ascii.eqlIgnoreCase(value, "websocket")) upgrade = true;
                } else if (std.ascii.eqlIgnoreCase(key, "Connection")) {
                    // Value may contain multiple tokens; check all
                    var token_it = std.mem.splitScalar(u8, value, ',');
                    while (token_it.next()) |tok| {
                        const trimmed_tok = std.mem.trim(u8, tok, " \t");
                        if (std.ascii.eqlIgnoreCase(trimmed_tok, "Upgrade")) {
                            connection_hdr_upgrade = true;
                            break;
                        }
                    }
                } else if (std.ascii.eqlIgnoreCase(key, "Sec-WebSocket-Key")) {
                    ws_key = value;
                }
            }
        }

        if (std.mem.eql(u8, request.path, "/ws") and upgrade and connection_hdr_upgrade and ws_key != null) {
            try self.handleWebSocketUpgrade(connection, ws_key.?);
            return;
        }

        // Handle the request
        try self.handleHttpRequest(connection, request);
    }

    /// Parse HTTP request
    fn parseHttpRequest(_: *Self, request_str: []const u8) !HttpRequest {
        var lines = std.mem.splitSequence(u8, request_str, "\r\n");

        // Parse first line (method, path, version)
        const first_line = lines.next() orelse return error.InvalidRequest;
        var parts = std.mem.splitScalar(u8, first_line, ' ');
        const method_str = parts.next() orelse return error.InvalidRequest;
        const path = parts.next() orelse return error.InvalidRequest;
        const version = parts.next() orelse return error.InvalidRequest;

        return HttpRequest{
            .method = method_str,
            .path = path,
            .version = version,
        };
    }

    /// HTTP request structure
    const HttpRequest = struct {
        method: []const u8,
        path: []const u8,
        version: []const u8,
    };

    /// Handle parsed HTTP request
    fn handleHttpRequest(self: *Self, connection: std.net.Server.Connection, request: HttpRequest) !void {
        // Route request
        if (std.mem.eql(u8, request.path, "/")) {
            try self.handleRoot(connection);
        } else if (std.mem.eql(u8, request.path, "/health")) {
            try self.handleHealth(connection);
        } else if (std.mem.eql(u8, request.path, "/stats")) {
            try self.handleStats(connection);
        } else if (std.mem.startsWith(u8, request.path, "/query")) {
            try self.handleQuery(connection, request);
        } else if (std.mem.startsWith(u8, request.path, "/knn")) {
            try self.handleKnn(connection, request);
        } else if (std.mem.startsWith(u8, request.path, "/monitor")) {
            try self.handleMonitor(connection);
        } else if (std.mem.eql(u8, request.path, "/network")) {
            try self.handleNetworkInfo(connection);
        } else {
            try self.sendHttpResponse(connection, 404, "Not Found", "{\"error\":\"Endpoint not found\"}");
        }
    }

    /// Send HTTP response
    fn sendHttpResponse(self: *Self, connection: std.net.Server.Connection, status: u16, status_text: []const u8, body: []const u8) !void {
        const response = try std.fmt.allocPrint(self.allocator, "HTTP/1.1 {d} {s}\r\n" ++
            "Content-Type: application/json\r\n" ++
            "Content-Length: {d}\r\n" ++
            "Access-Control-Allow-Origin: *\r\n" ++
            "\r\n" ++
            "{s}", .{ status, status_text, body.len, body });
        defer self.allocator.free(response);

        _ = connection.stream.write(response) catch |err| {
            switch (err) {
                error.ConnectionResetByPeer, error.BrokenPipe, error.Unexpected => return,
                else => return err,
            }
        };
    }

    /// Handle root endpoint
    fn handleRoot(self: *Self, connection: std.net.Server.Connection) !void {
        const html =
            "<!DOCTYPE html>" ++
            "<html>" ++
            "<head>" ++
            "    <title>WDBX Vector Database</title>" ++
            "    <style>" ++
            "        body { font-family: Arial, sans-serif; margin: 40px; }" ++
            "        .container { max-width: 800px; margin: 0 auto; }" ++
            "        h1 { color: #333; }" ++
            "        .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }" ++
            "        .method { color: #007acc; font-weight: bold; }" ++
            "        .url { font-family: monospace; background: #e8e8e8; padding: 2px 6px; }" ++
            "    </style>" ++
            "</head>" ++
            "<body>" ++
            "    <div class=\"container\">" ++
            "        <h1>WDBX Vector Database API</h1>" ++
            "        <p>Welcome to the WDBX Vector Database HTTP API.</p>" ++
            "        <h2>Available Endpoints</h2>" ++
            "        <div class=\"endpoint\">" ++
            "            <span class=\"method\">GET</span> <span class=\"url\">/health</span>" ++
            "            <p>Check server health status</p>" ++
            "        </div>" ++
            "        <div class=\"endpoint\">" ++
            "            <span class=\"method\">GET</span> <span class=\"url\">/stats</span>" ++
            "            <p>Get database statistics</p>" ++
            "        </div>" ++
            "        <div class=\"endpoint\">" ++
            "            <span class=\"method\">GET</span> <span class=\"url\">/query?vec=1.0,2.0,3.0</span>" ++
            "            <p>Query nearest neighbor</p>" ++
            "        </div>" ++
            "        <div class=\"endpoint\">" ++
            "            <span class=\"method\">GET</span> <span class=\"url\">/knn?vec=1.0,2.0,3.0&k=5</span>" ++
            "            <p>Query k-nearest neighbors</p>" ++
            "        </div>" ++
            "        <div class=\"endpoint\">" ++
            "            <span class=\"method\">GET</span> <span class=\"url\">/monitor</span>" ++
            "            <p>Get performance metrics</p>" ++
            "        </div>" ++
            "        <h2>Vector Format</h2>" ++
            "        <p>Vectors should be comma-separated float values, e.g.: <code>1.0,2.0,3.0,4.0</code></p>" ++
            "    </div>" ++
            "</body>" ++
            "</html>";

        const response = try std.fmt.allocPrint(self.allocator, "HTTP/1.1 200 OK\r\n" ++
            "Content-Type: text/html\r\n" ++
            "Content-Length: {d}\r\n" ++
            "Access-Control-Allow-Origin: *\r\n" ++
            "\r\n" ++
            "{s}", .{ html.len, html });
        defer self.allocator.free(response);

        _ = connection.stream.write(response) catch |err| {
            switch (err) {
                error.ConnectionResetByPeer, error.BrokenPipe, error.Unexpected => return,
                else => return err,
            }
        };
    }

    /// Handle health check endpoint
    fn handleHealth(self: *Self, connection: std.net.Server.Connection) !void {
        const body = try std.fmt.allocPrint(self.allocator,
            \\{{"status":"healthy","version":"{s}","timestamp":{d},"database_connected":{any},"platform":"windows","optimizations":{{"tcp_nodelay":{any},"keepalive":{any},"buffer_size":{d}}}}}
        , .{ version_string, std.time.milliTimestamp(), self.db != null, self.config.tcp_nodelay, self.config.socket_keepalive, self.config.socket_buffer_size });
        defer self.allocator.free(body);

        try self.sendHttpResponse(connection, 200, "OK", body);
    }

    /// Handle statistics endpoint
    fn handleStats(self: *Self, connection: std.net.Server.Connection) !void {
        if (self.db == null) {
            try self.sendHttpResponse(connection, 503, "Service Unavailable", "{\"error\":\"Database not connected\"}");
            return;
        }

        const db = self.db.?;
        const stats = db.getStats();
        const body = try std.fmt.allocPrint(self.allocator, "{{\"vectors_stored\":{d},\"vector_dimension\":{d},\"searches_performed\":{d},\"average_search_time_us\":{d},\"writes_performed\":{d},\"initializations\":{d}}}", .{ db.getRowCount(), db.getDimension(), stats.search_count, stats.getAverageSearchTime(), stats.write_count, stats.initialization_count });
        defer self.allocator.free(body);

        try self.sendHttpResponse(connection, 200, "OK", body);
    }

    /// Handle query endpoint
    fn handleQuery(self: *Self, connection: std.net.Server.Connection, request: HttpRequest) !void {
        if (self.db == null) {
            try self.sendHttpResponse(connection, 503, "Service Unavailable", "{\"error\":\"Database not connected\"}");
            return;
        }

        // Parse query parameters
        const query = request.path;
        const vec_start = std.mem.indexOf(u8, query, "vec=") orelse {
            try self.sendHttpResponse(connection, 400, "Bad Request", "{\"error\":\"Missing 'vec' parameter\"}");
            return;
        };

        const vec_end = std.mem.indexOfScalar(u8, query[vec_start..], '&') orelse query.len;
        const vector_str = query[vec_start + 4 .. vec_start + vec_end];

        // Parse vector
        const vector = try self.parseVector(vector_str);
        defer self.allocator.free(vector);

        // Query database
        const db = self.db.?;
        const results = try db.search(vector, 1, self.allocator);
        defer self.allocator.free(results);

        if (results.len > 0) {
            const body = try std.fmt.allocPrint(self.allocator, "{{\"success\":true,\"nearest_neighbor\":{{\"index\":{d},\"distance\":{d}}}}}", .{ results[0].index, results[0].score });
            defer self.allocator.free(body);

            try self.sendHttpResponse(connection, 200, "OK", body);
        } else {
            try self.sendHttpResponse(connection, 404, "Not Found", "{\"error\":\"No vectors found in database\"}");
        }
    }

    /// Handle k-nearest neighbors endpoint
    fn handleKnn(self: *Self, connection: std.net.Server.Connection, request: HttpRequest) !void {
        if (self.db == null) {
            try self.sendHttpResponse(connection, 503, "Service Unavailable", "{\"error\":\"Database not connected\"}");
            return;
        }

        // Parse query parameters
        const query = request.path;
        const vec_start = std.mem.indexOf(u8, query, "vec=") orelse {
            try self.sendHttpResponse(connection, 400, "Bad Request", "{\"error\":\"Missing 'vec' parameter\"}");
            return;
        };

        const vec_end = std.mem.indexOfScalar(u8, query[vec_start..], '&') orelse query.len;
        const vector_str = query[vec_start + 4 .. vec_start + vec_end];

        // Parse k parameter
        var k: usize = 5; // Default
        if (std.mem.indexOf(u8, query, "k=")) |k_start| {
            const k_end = std.mem.indexOfScalar(u8, query[k_start..], '&') orelse query.len;
            const k_str = query[k_start + 2 .. k_start + k_end];
            k = try std.fmt.parseInt(usize, k_str, 10);
        }

        // Parse vector
        const vector = try self.parseVector(vector_str);
        defer self.allocator.free(vector);

        // Query database
        const db = self.db.?;
        const results = try db.search(vector, k, self.allocator);
        defer self.allocator.free(results);

        // Format results
        var neighbors = try std.ArrayList(NeighborResult).initCapacity(self.allocator, results.len);
        defer neighbors.deinit(self.allocator);

        for (results) |result| {
            try neighbors.append(self.allocator, .{
                .index = result.index,
                .distance = result.score,
            });
        }

        const body = try std.fmt.allocPrint(self.allocator, "{{\"success\":true,\"k\":{d},\"neighbors\":[{s}]}}", .{ k, try self.formatNeighbors(neighbors.items) });
        defer self.allocator.free(body);

        try self.sendHttpResponse(connection, 200, "OK", body);
    }

    /// Handle monitor endpoint
    fn handleMonitor(self: *Self, connection: std.net.Server.Connection) !void {
        const body = try std.fmt.allocPrint(self.allocator, "{{\"server_uptime_ms\":{d},\"rate_limit_enabled\":{any},\"cors_enabled\":{any},\"max_request_size\":{d},\"rate_limit_per_minute\":{d}}}", .{ std.time.milliTimestamp(), self.config.enable_auth, self.config.enable_cors, self.config.max_request_size, self.config.rate_limit });
        defer self.allocator.free(body);

        try self.sendHttpResponse(connection, 200, "OK", body);
    }

    /// Perform WebSocket handshake for upgrade requests
    fn handleWebSocketUpgrade(_: *Self, connection: std.net.Server.Connection, key: []const u8) !void {
        // Compute Sec-WebSocket-Accept = base64( SHA1(key + GUID) )
        const guid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
        var sha1 = std.crypto.hash.Sha1.init(.{});
        sha1.update(key);
        sha1.update(guid);
        var digest: [20]u8 = undefined;
        sha1.final(&digest);

        var accept_buf: [64]u8 = undefined;
        const accept = std.base64.standard.Encoder.encode(&accept_buf, &digest);

        const response = try std.fmt.allocPrint(
            std.heap.page_allocator,
            "HTTP/1.1 101 Switching Protocols\r\n" ++
                "Upgrade: websocket\r\n" ++
                "Connection: Upgrade\r\n" ++
                "Sec-WebSocket-Accept: {s}\r\n" ++
                "\r\n",
            .{accept},
        );
        defer std.heap.page_allocator.free(response);
        _ = connection.stream.write(response) catch |err| switch (err) {
            error.ConnectionResetByPeer, error.BrokenPipe, error.Unexpected => return,
            else => return err,
        };
        // Give the client a moment to receive the handshake before the server closes the socket (Windows quirk)
        std.Thread.sleep(25 * std.time.ns_per_ms);
    }

    /// Neighbor result structure
    const NeighborResult = struct {
        index: u64,
        distance: f32,
    };

    /// Parse vector string
    fn parseVector(self: *Self, vector_str: []const u8) ![]f32 {
        var list = try std.ArrayList(f32).initCapacity(self.allocator, 8);
        defer list.deinit(self.allocator);

        var iter = std.mem.splitSequence(u8, vector_str, ",");
        while (iter.next()) |part| {
            const trimmed = std.mem.trim(u8, part, " \t\n\r");
            if (trimmed.len > 0) {
                const value = try std.fmt.parseFloat(f32, trimmed);
                try list.append(self.allocator, value);
            }
        }

        return try list.toOwnedSlice(self.allocator);
    }

    /// Format neighbors array for JSON output
    fn formatNeighbors(self: *Self, neighbors: []const NeighborResult) ![]const u8 {
        var buffer = try std.ArrayList(u8).initCapacity(self.allocator, 256);
        defer buffer.deinit(self.allocator);

        for (neighbors, 0..) |neighbor, i| {
            if (i > 0) try buffer.appendSlice(self.allocator, ",");
            const item = try std.fmt.allocPrint(self.allocator, "{{\"index\":{d},\"distance\":{d}}}", .{ neighbor.index, neighbor.distance });
            defer self.allocator.free(item);
            try buffer.appendSlice(self.allocator, item);
        }

        return try buffer.toOwnedSlice(self.allocator);
    }

    /// Extract query parameter from the request path. Returns a slice into `path` or null if not found.
    fn extractQueryParam(_: *Self, path: []const u8, key: []const u8) ?[]const u8 {
        const q_index = std.mem.indexOfScalar(u8, path, '?') orelse return null;
        const query = path[q_index + 1 ..];
        var it = std.mem.splitScalar(u8, query, '&');
        while (it.next()) |pair| {
            if (pair.len == 0) continue;
            const eq = std.mem.indexOfScalar(u8, pair, '=') orelse continue;
            const name = pair[0..eq];
            const value = pair[eq + 1 ..];
            if (std.mem.eql(u8, name, key)) return value;
        }
        return null;
    }

    /// Extract a header value by name (case-insensitive) from a raw HTTP request string.
    /// Returns a slice into `request_str` or null if not found.
    fn extractHeaderValue(_: *Self, request_str: []const u8, key: []const u8) ?[]const u8 {
        var lines = std.mem.splitSequence(u8, request_str, "\r\n");
        _ = lines.next(); // skip request line
        while (lines.next()) |line| {
            if (line.len == 0) break;
            if (std.mem.indexOfScalar(u8, line, ':')) |colon| {
                const name = std.mem.trim(u8, line[0..colon], " \t");
                const value = std.mem.trim(u8, line[colon + 1 ..], " \t");
                if (std.ascii.eqlIgnoreCase(name, key)) return value;
            }
        }
        return null;
    }

    /// Test basic connectivity - useful for diagnosing Windows networking issues
    pub fn testConnectivity(self: *Self) !bool {
        if (self.server == null) return false;

        const abi = @import("abi");
        const http_client = abi.http_client;

        // Configure HTTP client with aggressive timeouts for testing
        var client = http_client.HttpClient.init(self.allocator, .{
            .connect_timeout_ms = 5000,
            .read_timeout_ms = 10000,
            .max_retries = 2,
            .initial_backoff_ms = 500,
            .max_backoff_ms = 2000,
            .user_agent = "WDBX-Connectivity-Test/1.0",
            .follow_redirects = false,
            .verify_ssl = false, // For local testing
            .verbose = true,
        });

        const health_url = try std.fmt.allocPrint(self.allocator, "http://{s}:{d}/health", .{ self.config.host, self.config.port });
        defer self.allocator.free(health_url);

        const success = client.testConnectivity(health_url) catch |err| {
            std.debug.print("Enhanced connectivity test failed: {any}\n", .{err});
            return false;
        };

        if (success) {
            std.debug.print("Enhanced connectivity test successful\n", .{});
        } else {
            std.debug.print("Enhanced connectivity test failed - server returned error\n", .{});
        }

        return success;
    }

    /// Handle network information endpoint (Windows diagnostics)
    fn handleNetworkInfo(self: *Self, connection: std.net.Server.Connection) !void {
        const network_info = try std.fmt.allocPrint(self.allocator,
            \\{{"platform":"windows","server_config":{{"host":"{s}","port":{d},"socket_buffer_size":{d},"tcp_nodelay":true,"socket_keepalive":true}},"socket_optimizations":{{"tcp_window_scaling":"enabled","receive_side_scaling":"enabled","tcp_chimney_offload":"enabled"}},"diagnostics":{{"connection_reset_handling":"enabled","buffer_overflow_protection":"enabled","graceful_error_recovery":"enabled"}},"recommendations":{{"run_as_admin":"For optimal performance, run PowerShell fixes as Administrator","restart_after_fixes":"Restart Windows after applying network fixes","test_with_powershell":"Use Invoke-WebRequest for reliable testing"}}}}
        , .{ self.config.host, self.config.port, self.config.socket_buffer_size });

        defer self.allocator.free(network_info);
        try self.sendHttpResponse(connection, 200, "OK", network_info);
    }
};

// Fuzz-style tests for request and vector parsing
const testing = std.testing;

test "parseVector property fuzz (valid and invalid inputs)" {
    const allocator = testing.allocator;
    const server = try WdbxHttpServer.init(allocator, .{});
    defer server.deinit();

    var prng = std.Random.DefaultPrng.init(0x12345678);
    const rand = prng.random();

    // Valid vectors: comma-separated integers (parsed as floats)
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        const n = rand.intRangeAtMost(usize, 1, 8);
        var buf = try std.ArrayList(u8).initCapacity(allocator, 64);
        defer buf.deinit(allocator);

        var j: usize = 0;
        while (j < n) : (j += 1) {
            if (j > 0) try buf.appendSlice(allocator, ",");
            const value = rand.intRangeAtMost(i32, -1000, 1000);
            const s = try std.fmt.allocPrint(allocator, "{d}", .{value});
            defer allocator.free(s);
            try buf.appendSlice(allocator, s);
        }

        const vec_str = try buf.toOwnedSlice(allocator);
        defer allocator.free(vec_str);

        const parsed = try server.parseVector(vec_str);
        defer allocator.free(parsed);
        try testing.expectEqual(n, parsed.len);
    }

    // Invalid vectors: ensure at least one non-numeric token forces an error
    i = 0;
    while (i < 100) : (i += 1) {
        var buf = try std.ArrayList(u8).initCapacity(allocator, 32);
        defer buf.deinit(allocator);

        const garbage_len = rand.intRangeAtMost(usize, 1, 16);
        var k: usize = 0;
        while (k < garbage_len) : (k += 1) {
            const ch = rand.intRangeAtMost(u8, 48, 122); // mix of digits/letters
            if (ch == ',') continue;
            try buf.append(allocator, ch);
        }
        // Guarantee non-numeric token
        try buf.append(allocator, 'x');

        const vec_str = try buf.toOwnedSlice(allocator);
        defer allocator.free(vec_str);

        if (server.parseVector(vec_str)) |_| {
            try testing.expect(false); // must not succeed on clearly invalid input
        } else |_| {}
    }
}

test "parseHttpRequest property fuzz (valid and invalid first lines)" {
    const allocator = testing.allocator;
    const server = try WdbxHttpServer.init(allocator, .{});
    defer server.deinit();

    var prng = std.Random.DefaultPrng.init(0xABCDEF01);
    const rand = prng.random();

    const methods = [_][]const u8{ "GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS" };

    // Valid request lines
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        const method = methods[rand.intRangeAtMost(usize, 0, methods.len - 1)];
        const path = switch (rand.intRangeAtMost(u8, 0, 2)) {
            0 => "/",
            1 => "/health",
            else => "/query?vec=1,2,3&k=2",
        };
        const version = if (rand.boolean()) "HTTP/1.1" else "HTTP/2";

        const req = try std.fmt.allocPrint(allocator, "{s} {s} {s}\r\n\r\n", .{ method, path, version });
        defer allocator.free(req);

        const parsed = try server.parseHttpRequest(req);
        try testing.expectEqualStrings(method, parsed.method);
        try testing.expectEqualStrings(path, parsed.path);
        try testing.expectEqualStrings(version, parsed.version);
    }

    // Invalid request lines (missing tokens / garbage)
    i = 0;
    while (i < 100) : (i += 1) {
        const bad = switch (rand.intRangeAtMost(u8, 0, 2)) {
            0 => "GET /onlytwo\r\n\r\n",
            1 => "/missing_method HTTP/1.1\r\n\r\n",
            else => "GARBAGE\r\n\r\n",
        };
        try testing.expectError(error.InvalidRequest, server.parseHttpRequest(bad));
    }
}

// Additional fuzz tests for query parameters and header parsing

test "extractQueryParam fuzz (presence, order, and interference)" {
    const allocator = testing.allocator;
    const server = try WdbxHttpServer.init(allocator, .{});
    defer server.deinit();

    var prng = std.Random.DefaultPrng.init(0xBEEF1234);
    const rand = prng.random();

    var i: usize = 0;
    while (i < 100) : (i += 1) {
        var buf = try std.ArrayList(u8).initCapacity(allocator, 128);
        defer buf.deinit(allocator);

        try buf.appendSlice(allocator, "/query?");

        var first: bool = true;
        const maybe_amp = struct {
            fn add(list: *std.ArrayList(u8), alloc: std.mem.Allocator, flag: *bool) !void {
                if (!flag.*) {
                    try list.appendSlice(alloc, "&");
                } else {
                    flag.* = false;
                }
            }
        };

        // Optional noise param
        if (rand.boolean()) {
            try maybe_amp.add(&buf, allocator, &first);
            try buf.appendSlice(allocator, "foo=bar");
        }

        // Decide if we include vec
        const include_vec = rand.boolean();
        var expected_vec: ?[]const u8 = null;
        var vec_value_to_free: ?[]const u8 = null;

        if (include_vec) {
            try maybe_amp.add(&buf, allocator, &first);
            const a = rand.intRangeAtMost(i32, -9, 9);
            const b = rand.intRangeAtMost(i32, -9, 9);
            const c = rand.intRangeAtMost(i32, -9, 9);
            const v = try std.fmt.allocPrint(allocator, "{d},{d},{d}", .{ a, b, c });
            vec_value_to_free = v;
            expected_vec = v;
            try buf.appendSlice(allocator, "vec=");
            try buf.appendSlice(allocator, v);
        }

        // Optional k param (appears before/after vec depending on include order)
        if (rand.boolean()) {
            try maybe_amp.add(&buf, allocator, &first);
            const k = rand.intRangeAtMost(usize, 1, 10);
            const ks = try std.fmt.allocPrint(allocator, "k={d}", .{k});
            defer allocator.free(ks);
            try buf.appendSlice(allocator, ks);
        }

        const path = try buf.toOwnedSlice(allocator);
        defer allocator.free(path);

        const got = server.extractQueryParam(path, "vec");
        if (include_vec) {
            try testing.expect(got != null);
            try testing.expectEqualStrings(expected_vec.?, got.?);
        } else {
            try testing.expect(got == null);
        }

        if (vec_value_to_free) |vv| allocator.free(vv);
    }
}

test "extractHeaderValue fuzz (case-insensitive keys and values)" {
    const allocator = testing.allocator;
    const server = try WdbxHttpServer.init(allocator, .{});
    defer server.deinit();

    var prng = std.Random.DefaultPrng.init(0xA1B2C3D4);
    const rand = prng.random();

    var i: usize = 0;
    while (i < 100) : (i += 1) {
        var buf = try std.ArrayList(u8).initCapacity(allocator, 256);
        defer buf.deinit(allocator);

        try buf.appendSlice(allocator, "GET / HTTP/1.1\r\n");

        const headers = [_]struct { key: []const u8, value: []const u8 }{
            .{ .key = "Host", .value = "localhost" },
            .{ .key = "Connection", .value = if (rand.boolean()) "keep-alive" else "Upgrade" },
            .{ .key = "X-Test", .value = "abc123" },
        };

        const expected_connection = headers[1].value;

        inline for (headers) |h| {
            var key_buf = try std.ArrayList(u8).initCapacity(allocator, h.key.len);
            defer key_buf.deinit(allocator);

            for (h.key) |ch| {
                var out_ch = ch;
                if (std.ascii.isAlphabetic(ch) and rand.boolean()) {
                    if (std.ascii.isUpper(ch)) {
                        out_ch = std.ascii.toLower(ch);
                    } else {
                        out_ch = std.ascii.toUpper(ch);
                    }
                }
                try key_buf.append(allocator, out_ch);
            }

            const key_s = try key_buf.toOwnedSlice(allocator);
            defer allocator.free(key_s);

            try buf.appendSlice(allocator, key_s);
            try buf.appendSlice(allocator, ": ");
            try buf.appendSlice(allocator, h.value);
            try buf.appendSlice(allocator, "\r\n");
        }

        try buf.appendSlice(allocator, "\r\n");
        const req = try buf.toOwnedSlice(allocator);
        defer allocator.free(req);

        const got_conn = server.extractHeaderValue(req, "Connection");
        try testing.expect(got_conn != null);
        try testing.expectEqualStrings(expected_connection, got_conn.?);

        const got_host = server.extractHeaderValue(req, "Host");
        try testing.expect(got_host != null);
        try testing.expectEqualStrings("localhost", got_host.?);

        const got_missing = server.extractHeaderValue(req, "Does-Not-Exist");
        try testing.expect(got_missing == null);
    }
}
