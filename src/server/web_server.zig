//! Web server for the Abi AI framework
//!
//! This module provides HTTP/HTTPS server capabilities including:
//! - RESTful API endpoints
//! - Static file serving
//! - WebSocket support
//! - Middleware system
//! - Request/response handling
//! - CORS support

const std = @import("std");
const builtin = @import("builtin");

/// Re-export commonly used types
pub const Allocator = std.mem.Allocator;

/// Web server configuration
pub const WebConfig = struct {
    port: u16 = 3000,
    host: []const u8 = "127.0.0.1",
    max_connections: u32 = 1000,
    enable_cors: bool = true,
    log_requests: bool = true,
    max_body_size: usize = 1024 * 1024, // 1MB
    timeout_seconds: u32 = 30,
    static_dir: ?[]const u8 = null,
};

/// Web server instance
pub const WebServer = struct {
    allocator: std.mem.Allocator,
    config: WebConfig,
    server: ?std.net.Server = null,

    pub fn init(allocator: std.mem.Allocator, config: WebConfig) !*WebServer {
        const self = try allocator.create(WebServer);
        self.* = .{
            .allocator = allocator,
            .config = config,
        };
        return self;
    }

    pub fn deinit(self: *WebServer) void {
        if (self.server) |*server| {
            server.deinit();
        }
        self.allocator.destroy(self);
    }

    pub fn start(self: *WebServer) !void {
        const address = try std.net.Address.parseIp(self.config.host, self.config.port);
        self.server = try address.listen(.{ .reuse_address = true });

        std.debug.print("Web server started on {s}:{}\n", .{ self.config.host, self.config.port });

        while (true) {
            const connection = self.server.?.accept() catch |err| {
                std.debug.print("Failed to accept connection: {any}\n", .{err});
                continue;
            };

            // Handle connection in background
            self.handleConnection(connection) catch |err| {
                std.debug.print("Connection handling error: {any}\n", .{err});
            };
        }
    }

    /// Handle HTTP connection
    fn handleConnection(self: *WebServer, connection: std.net.Server.Connection) !void {
        defer connection.stream.close();

        var buffer: [4096]u8 = undefined;
        var bytes_read: usize = 0;
        if (builtin.os.tag == .windows) {
            const windows = std.os.windows;
            const max_len: c_int = @intCast(@min(buffer.len, @as(usize, @intCast(std.math.maxInt(c_int)))));
            const n: c_int = windows.ws2_32.recv(connection.stream.handle, @ptrCast(&buffer[0]), max_len, 0);
            if (n == windows.ws2_32.SOCKET_ERROR) {
                // Treat as client disconnect or transient error; ignore
                return;
            }
            bytes_read = @intCast(n);
        } else {
            bytes_read = connection.stream.read(&buffer) catch |err| {
                switch (err) {
                    error.ConnectionResetByPeer, error.BrokenPipe, error.Unexpected => return,
                    else => return err,
                }
            };
        }

        if (bytes_read == 0) return;

        const request_str = buffer[0..bytes_read];

        // Check if this is a WebSocket upgrade request
        if (self.isWebSocketUpgrade(request_str)) {
            try self.handleWebSocketUpgrade(connection, request_str);
            try self.handleWebSocketProtocol(connection);
        } else {
            // Handle HTTP request
            try self.handleHttpRequest(connection, request_str);
        }
    }

    /// Check if request is a WebSocket upgrade
    fn isWebSocketUpgrade(_: *WebServer, request: []const u8) bool {
        var upgrade = false;
        var connection_upgrade = false;
        var ws_key = false;

        var lines = std.mem.splitSequence(u8, request, "\r\n");
        _ = lines.next(); // skip request line
        while (lines.next()) |line| {
            if (line.len == 0) break;
            if (std.mem.indexOfScalar(u8, line, ':')) |colon| {
                const key = std.mem.trim(u8, line[0..colon], " \t");
                const value = std.mem.trim(u8, line[colon + 1 ..], " \t");

                if (std.ascii.eqlIgnoreCase(key, "Upgrade")) {
                    if (std.ascii.eqlIgnoreCase(value, "websocket")) upgrade = true;
                } else if (std.ascii.eqlIgnoreCase(key, "Connection")) {
                    if (std.mem.indexOf(u8, value, "Upgrade") != null) connection_upgrade = true;
                } else if (std.ascii.eqlIgnoreCase(key, "Sec-WebSocket-Key")) {
                    ws_key = true;
                }
            }
        }

        return upgrade and connection_upgrade and ws_key;
    }

    /// Handle WebSocket upgrade handshake
    fn handleWebSocketUpgrade(self: *WebServer, connection: std.net.Server.Connection, request: []const u8) !void {
        // Extract WebSocket key
        var ws_key: ?[]const u8 = null;
        var lines = std.mem.splitSequence(u8, request, "\r\n");
        _ = lines.next(); // skip request line
        while (lines.next()) |line| {
            if (line.len == 0) break;
            if (std.mem.indexOfScalar(u8, line, ':')) |colon| {
                const key = std.mem.trim(u8, line[0..colon], " \t");
                const value = std.mem.trim(u8, line[colon + 1 ..], " \t");
                if (std.ascii.eqlIgnoreCase(key, "Sec-WebSocket-Key")) {
                    ws_key = value;
                    break;
                }
            }
        }

        if (ws_key == null) {
            try self.sendHttpResponse(connection, 400, "Bad Request", "Missing Sec-WebSocket-Key");
            return;
        }

        // Compute Sec-WebSocket-Accept
        const guid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
        var sha1 = std.crypto.hash.Sha1.init(.{});
        sha1.update(ws_key.?);
        sha1.update(guid);
        var digest: [20]u8 = undefined;
        sha1.final(&digest);

        var accept_buf: [64]u8 = undefined;
        const accept = std.base64.standard.Encoder.encode(&accept_buf, &digest);

        // Send upgrade response
        const response = try std.fmt.allocPrint(
            self.allocator,
            "HTTP/1.1 101 Switching Protocols\r\n" ++
                "Upgrade: websocket\r\n" ++
                "Connection: Upgrade\r\n" ++
                "Sec-WebSocket-Accept: {s}\r\n" ++
                "\r\n",
            .{accept},
        );
        defer self.allocator.free(response);

        _ = connection.stream.write(response) catch |err| {
            switch (err) {
                error.ConnectionResetByPeer, error.BrokenPipe, error.Unexpected => return,
                else => return err,
            }
        };
    }

    /// Handle WebSocket protocol after upgrade
    fn handleWebSocketProtocol(self: *WebServer, connection: std.net.Server.Connection) !void {
        var buffer: [4096]u8 = undefined;

        while (true) {
            const bytes_read = connection.stream.read(&buffer) catch |err| {
                switch (err) {
                    error.ConnectionResetByPeer, error.BrokenPipe, error.Unexpected => return,
                    else => return err,
                }
            };

            if (bytes_read == 0) return;

            // Parse WebSocket frame
            const frame = try self.parseWebSocketFrame(buffer[0..bytes_read]);
            if (frame.opcode == 0x8) { // Close frame
                try self.sendWebSocketClose(connection);
                return;
            } else if (frame.opcode == 0x9) { // Ping frame
                try self.sendWebSocketPong(connection);
            } else if (frame.opcode == 0x1) { // Text frame
                try self.handleWebSocketMessage(connection, frame.payload);
            }
        }
    }

    /// WebSocket frame structure
    const WebSocketFrame = struct {
        fin: bool,
        opcode: u4,
        payload: []const u8,
    };

    /// Parse WebSocket frame
    pub fn parseWebSocketFrame(_: *WebServer, data: []const u8) !WebSocketFrame {
        if (data.len < 2) return error.InvalidFrame;

        const first_byte = data[0];
        const second_byte = data[1];

        const fin = (first_byte & 0x80) != 0;
        const opcode: u4 = @intCast(first_byte & 0x0F);
        const masked = (second_byte & 0x80) != 0;
        var payload_len: usize = second_byte & 0x7F;

        var offset: usize = 2;

        // Extended payload length
        if (payload_len == 126) {
            if (data.len < 4) return error.InvalidFrame;
            payload_len = std.mem.readInt(u16, data[2..4], .big);
            offset += 2;
        } else if (payload_len == 127) {
            if (data.len < 10) return error.InvalidFrame;
            payload_len = std.mem.readInt(u64, data[2..10], .big);
            offset += 8;
        }

        // Mask key
        if (masked) {
            if (data.len < offset + 4) return error.InvalidFrame;
            offset += 4;
        }

        if (data.len < offset + payload_len) return error.InvalidFrame;

        const payload = data[offset .. offset + payload_len];

        return WebSocketFrame{
            .fin = fin,
            .opcode = opcode,
            .payload = payload,
        };
    }

    /// Handle WebSocket message
    fn handleWebSocketMessage(self: *WebServer, connection: std.net.Server.Connection, message: []const u8) !void {
        // Echo message back for now
        try self.sendWebSocketFrame(connection, 0x1, message);
    }

    /// Send WebSocket close frame
    fn sendWebSocketClose(self: *WebServer, connection: std.net.Server.Connection) !void {
        try self.sendWebSocketFrame(connection, 0x8, "");
    }

    /// Send WebSocket pong frame
    fn sendWebSocketPong(self: *WebServer, connection: std.net.Server.Connection) !void {
        try self.sendWebSocketFrame(connection, 0xA, "");
    }

    /// Send WebSocket frame
    fn sendWebSocketFrame(self: *WebServer, connection: std.net.Server.Connection, opcode: u4, payload: []const u8) !void {
        var frame = try std.ArrayList(u8).initCapacity(self.allocator, 2 + payload.len);
        defer frame.deinit(self.allocator);

        // First byte: FIN + RSV + Opcode
        const first_byte: u8 = 0x80 | @as(u8, @intCast(opcode));
        try frame.append(self.allocator, first_byte); // FIN = 1, RSV = 0, Opcode = opcode

        // Second byte: MASK + Payload length
        if (payload.len < 126) {
            try frame.append(self.allocator, @intCast(payload.len));
        } else if (payload.len < 65536) {
            try frame.append(self.allocator, 126);
            try frame.appendSlice(self.allocator, &std.mem.toBytes(@as(u16, @intCast(payload.len))));
        } else {
            try frame.append(self.allocator, 127);
            try frame.appendSlice(self.allocator, &std.mem.toBytes(@as(u64, @intCast(payload.len))));
        }

        // Payload
        try frame.appendSlice(self.allocator, payload);

        _ = connection.stream.write(frame.items) catch |err| {
            switch (err) {
                error.ConnectionResetByPeer, error.BrokenPipe, error.Unexpected => return,
                else => return err,
            }
        };
    }

    /// Handle HTTP request
    fn handleHttpRequest(self: *WebServer, connection: std.net.Server.Connection, request_str: []const u8) !void {
        // Parse request line
        var lines = std.mem.splitSequence(u8, request_str, "\r\n");
        const request_line = lines.next() orelse return;

        var parts = std.mem.splitScalar(u8, request_line, ' ');
        _ = parts.next() orelse return; // method
        const path = parts.next() orelse return;
        _ = parts.next() orelse return; // version

        // Route request
        if (std.mem.eql(u8, path, "/")) {
            try self.handleRoot(connection);
        } else if (std.mem.eql(u8, path, "/health")) {
            try self.handleHealth(connection);
        } else if (std.mem.eql(u8, path, "/api/status")) {
            try self.handleApiStatus(connection);
        } else if (self.config.static_dir != null and std.mem.startsWith(u8, path, "/static/")) {
            try self.handleStaticFile(connection, path);
        } else {
            try self.sendHttpResponse(connection, 404, "Not Found", "{\"error\":\"Not Found\"}");
        }
    }

    /// Start the server, accept a single connection, handle it, then stop.
    /// Intended for testing with a live socket.
    pub fn startOnce(self: *WebServer) !void {
        const address = try std.net.Address.parseIp(self.config.host, self.config.port);
        self.server = try address.listen(.{ .reuse_address = true });
        defer {
            if (self.server) |*srv| srv.deinit();
            self.server = null;
        }

        const connection = self.server.?.accept() catch |err| switch (err) {
            error.ConnectionAborted, error.WouldBlock, error.ConnectionResetByPeer => return,
            else => return err,
        };
        try self.handleConnection(connection);
    }

    // Test helper: route by path and return body string
    pub fn handlePathForTest(self: *WebServer, path: []const u8, allocator: std.mem.Allocator) ![]u8 {
        _ = self;
        if (std.mem.eql(u8, path, "/")) {
            const html =
                "<!DOCTYPE html>" ++
                "<html><head><title>Abi AI Framework</title></head><body>" ++
                "<h1>Abi AI Framework Web Server</h1>" ++
                "</body></html>";
            return allocator.dupe(u8, html);
        } else if (std.mem.eql(u8, path, "/health")) {
            return allocator.dupe(u8, "{\"status\":\"healthy\"}");
        } else if (std.mem.eql(u8, path, "/api/status")) {
            return allocator.dupe(u8, "{\"api\":\"running\"}");
        } else {
            return allocator.dupe(u8, "{\"error\":\"Not Found\"}");
        }
    }

    /// Handle root endpoint
    fn handleRoot(self: *WebServer, connection: std.net.Server.Connection) !void {
        const html =
            "<!DOCTYPE html>" ++
            "<html>" ++
            "<head><title>Abi AI Framework</title></head>" ++
            "<body>" ++
            "<h1>Abi AI Framework Web Server</h1>" ++
            "<p>Server is running successfully!</p>" ++
            "<ul>" ++
            "<li><a href='/health'>Health Check</a></li>" ++
            "<li><a href='/api/status'>API Status</a></li>" ++
            "</ul>" ++
            "</body></html>";

        try self.sendHttpResponse(connection, 200, "OK", html);
    }

    /// Handle health check
    fn handleHealth(self: *WebServer, connection: std.net.Server.Connection) !void {
        const body = "{\"status\":\"healthy\",\"timestamp\":\"2025-01-01T00:00:00Z\"}";
        try self.sendHttpResponse(connection, 200, "OK", body);
    }

    /// Handle API status
    fn handleApiStatus(self: *WebServer, connection: std.net.Server.Connection) !void {
        const body = "{\"api\":\"running\",\"version\":\"1.0.0\"}";
        try self.sendHttpResponse(connection, 200, "OK", body);
    }

    /// Handle static file serving
    fn handleStaticFile(self: *WebServer, connection: std.net.Server.Connection, path: []const u8) !void {
        if (self.config.static_dir == null) {
            try self.sendHttpResponse(connection, 404, "Not Found", "{\"error\":\"Static files not enabled\"}");
            return;
        }

        const static_dir = self.config.static_dir.?;
        const file_path = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ static_dir, path[7..] });
        defer self.allocator.free(file_path);

        const file = std.fs.cwd().openFile(file_path, .{}) catch |err| {
            try self.sendHttpResponse(connection, 404, "Not Found", "{\"error\":\"File not found\"}");
            return err;
        };
        defer file.close();

        // Zig 0.16-dev: replace deprecated readToEndAlloc with manual stat+read
        const st = file.stat() catch |err| {
            try self.sendHttpResponse(connection, 500, "Internal Server Error", "{\"error\":\"Failed to stat file\"}");
            return err;
        };
        const file_size_u64: u64 = st.size;
        const max_bytes: u64 = self.config.max_body_size;
        const to_read_u64: u64 = if (file_size_u64 > max_bytes) max_bytes else file_size_u64;
        const to_read: usize = @intCast(to_read_u64);

        var buf = try self.allocator.alloc(u8, to_read);
        defer self.allocator.free(buf);
        const n = file.readAll(buf) catch |err| {
            try self.sendHttpResponse(connection, 500, "Internal Server Error", "{\"error\":\"Failed to read file\"}");
            return err;
        };
        const body = buf[0..n];

        try self.sendHttpResponse(connection, 200, "OK", body);
    }

    /// Send HTTP response
    fn sendHttpResponse(self: *WebServer, connection: std.net.Server.Connection, status: u16, status_text: []const u8, body: []const u8) !void {
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
};
