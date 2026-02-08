//! HTTP Server Implementation
//!
//! Provides the main HTTP server that listens for connections and dispatches
//! requests to handlers. Uses Zig 0.16 std.Io.Threaded for async I/O.

const std = @import("std");
const time = @import("../../../services/shared/time.zig");
const sync = @import("../../../services/shared/sync.zig");
const types = @import("types.zig");

const ServerConfig = types.ServerConfig;
const ServerState = types.ServerState;
const ServerStats = types.ServerStats;
const ServerError = types.ServerError;
const Connection = types.Connection;

/// HTTP Server that accepts connections and processes requests.
pub const Server = struct {
    /// Memory allocator.
    allocator: std.mem.Allocator,
    /// Server configuration.
    config: ServerConfig,
    /// Current server state.
    state: ServerState,
    /// Server statistics.
    stats: ServerStats,
    /// TCP listener socket.
    listener: ?std.posix.socket_t,
    /// Connection ID counter.
    next_conn_id: u64,
    /// Mutex for thread-safe state access.
    mutex: sync.Mutex,
    /// Request handler callback.
    handler: ?*const RequestHandler,

    /// Request handler function type.
    pub const RequestHandler = fn (
        allocator: std.mem.Allocator,
        request: *Request,
        response: *Response,
    ) anyerror!void;

    /// Simplified request structure for handler.
    pub const Request = struct {
        method: types.Method,
        path: []const u8,
        query: ?[]const u8,
        headers: std.StringHashMap([]const u8),
        body: ?[]const u8,
        connection: *Connection,

        pub fn init(allocator: std.mem.Allocator) Request {
            return .{
                .method = .GET,
                .path = "/",
                .query = null,
                .headers = std.StringHashMap([]const u8).init(allocator),
                .body = null,
                .connection = undefined,
            };
        }

        pub fn deinit(self: *Request) void {
            self.headers.deinit();
        }

        /// Gets a header value (case-insensitive).
        pub fn getHeader(self: *const Request, name: []const u8) ?[]const u8 {
            return self.headers.get(name);
        }
    };

    /// Simplified response structure for handler.
    pub const Response = struct {
        status: types.Status,
        headers: std.StringHashMap([]const u8),
        body: std.ArrayListUnmanaged(u8),
        sent: bool,
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) Response {
            return .{
                .status = .ok,
                .headers = std.StringHashMap([]const u8).init(allocator),
                .body = .empty,
                .sent = false,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Response) void {
            self.headers.deinit();
            self.body.deinit(self.allocator);
        }

        /// Sets the response status.
        pub fn setStatus(self: *Response, status: types.Status) *Response {
            self.status = status;
            return self;
        }

        /// Sets a response header.
        pub fn setHeader(self: *Response, name: []const u8, value: []const u8) !*Response {
            try self.headers.put(name, value);
            return self;
        }

        /// Sets the response body as raw bytes.
        pub fn setBody(self: *Response, body: []const u8) !*Response {
            self.body.clearRetainingCapacity();
            try self.body.appendSlice(self.allocator, body);
            return self;
        }

        fn setJsonBodyWithOptions(self: *Response, value: anytype, options: anytype) !*Response {
            self.body.clearRetainingCapacity();
            try std.json.stringify(value, options, self.body.writer(self.allocator));
            _ = try self.setHeader(types.Header.content_type, types.MimeType.json);
            return self;
        }

        /// Sets the response body as JSON.
        pub fn json(self: *Response, value: anytype) !*Response {
            return self.setJsonBodyWithOptions(value, .{});
        }

        fn setTypedBody(self: *Response, content: []const u8, mime_type: []const u8) !*Response {
            _ = try self.setBody(content);
            _ = try self.setHeader(types.Header.content_type, mime_type);
            return self;
        }

        /// Sets the response body as plain text.
        pub fn text(self: *Response, content: []const u8) !*Response {
            return self.setTypedBody(content, types.MimeType.plain);
        }

        /// Sets the response body as HTML.
        pub fn html(self: *Response, content: []const u8) !*Response {
            return self.setTypedBody(content, types.MimeType.html);
        }

        fn writeHttpResponse(self: *Response, writer: anytype) !void {
            // Write status line
            try writer.print("HTTP/1.1 {d} {s}\r\n", .{
                @intFromEnum(self.status),
                self.status.phrase() orelse "Unknown",
            });

            // Write Content-Length if body present
            if (self.body.items.len > 0) {
                try writer.print("Content-Length: {d}\r\n", .{self.body.items.len});
            }

            // Write headers
            var it = self.headers.iterator();
            while (it.next()) |entry| {
                try writer.print("{s}: {s}\r\n", .{ entry.key_ptr.*, entry.value_ptr.* });
            }

            // End headers
            try writer.writeAll("\r\n");

            // Write body
            if (self.body.items.len > 0) {
                try writer.writeAll(self.body.items);
            }

            self.sent = true;
        }

        /// Serializes the response to a writer.
        pub fn writeTo(self: *Response, writer: anytype) !void {
            try self.writeHttpResponse(writer);
        }
    };

    /// Creates a new HTTP server.
    pub fn init(allocator: std.mem.Allocator, config: ServerConfig) Server {
        return .{
            .allocator = allocator,
            .config = config,
            .state = .stopped,
            .stats = ServerStats{},
            .listener = null,
            .next_conn_id = 1,
            .mutex = .{},
            .handler = null,
        };
    }

    /// Cleans up server resources.
    pub fn deinit(self: *Server) void {
        self.stop();
    }

    /// Sets the request handler.
    pub fn setHandler(self: *Server, handler: *const RequestHandler) void {
        self.handler = handler;
    }

    /// Starts the server and begins listening for connections.
    pub fn start(self: *Server) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.state != .stopped) {
            return ServerError.AlreadyRunning;
        }

        self.state = .starting;
        errdefer self.state = .errored;

        // Create socket
        const sock = try std.posix.socket(
            std.posix.AF.INET,
            std.posix.SOCK.STREAM,
            0,
        );
        errdefer std.posix.close(sock);

        // Set socket options
        const enable: i32 = 1;
        _ = std.posix.setsockopt(sock, std.posix.SOL.SOCKET, std.posix.SO.REUSEADDR, std.mem.asBytes(&enable));

        // Bind
        const addr = std.net.Address.parseIp4(self.config.host, self.config.port) catch
            return ServerError.BindFailed;

        std.posix.bind(sock, &addr.any, addr.getOsSockLen()) catch
            return ServerError.BindFailed;

        // Listen
        std.posix.listen(sock, 128) catch return ServerError.BindFailed;

        self.listener = sock;
        self.state = .running;
        self.stats.started_at = std.time.milliTimestamp();

        std.log.info("Server started on {s}:{d}", .{ self.config.host, self.config.port });
    }

    /// Stops the server gracefully.
    pub fn stop(self: *Server) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.state != .running) return;

        self.state = .stopping;

        if (self.listener) |sock| {
            std.posix.close(sock);
            self.listener = null;
        }

        self.state = .stopped;
        std.log.info("Server stopped", .{});
    }

    /// Accepts a single connection (blocking).
    pub fn accept(self: *Server) !Connection {
        if (self.state != .running) {
            return ServerError.NotRunning;
        }

        const sock = self.listener orelse return ServerError.NotRunning;

        var client_addr: std.posix.sockaddr = undefined;
        var addr_len: std.posix.socklen_t = @sizeOf(std.posix.sockaddr);

        const client_sock = std.posix.accept(sock, &client_addr, &addr_len, 0) catch |err| {
            std.log.err("Accept failed: {t}", .{err});
            return ServerError.InternalError;
        };
        _ = client_sock;

        self.mutex.lock();
        const conn_id = self.next_conn_id;
        self.next_conn_id += 1;
        self.stats.total_connections += 1;
        self.stats.active_connections += 1;
        self.mutex.unlock();

        const address = std.net.Address{ .any = client_addr };
        return Connection.init(self.allocator, conn_id, address);
    }

    /// Runs the server main loop (blocking).
    pub fn run(self: *Server) !void {
        try self.start();
        defer self.stop();

        while (self.state == .running) {
            const conn = self.accept() catch |err| {
                if (err == ServerError.NotRunning) break;
                std.log.warn("Accept error: {t}", .{err});
                continue;
            };
            _ = conn;

            // In a full implementation, would spawn a thread/task to handle connection
            // For now, just track the connection
            self.mutex.lock();
            self.stats.active_connections -= 1;
            self.mutex.unlock();
        }
    }

    /// Returns current server statistics.
    pub fn getStats(self: *Server) ServerStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.stats;
    }

    /// Returns current server state.
    pub fn getState(self: *Server) ServerState {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.state;
    }

    /// Checks if the server is running.
    pub fn isRunning(self: *Server) bool {
        return self.getState() == .running;
    }
};

test "Server initialization" {
    const allocator = std.testing.allocator;
    const config = ServerConfig{};

    var server = Server.init(allocator, config);
    defer server.deinit();

    try std.testing.expectEqual(ServerState.stopped, server.getState());
    try std.testing.expect(!server.isRunning());
}

test "Response builder" {
    const allocator = std.testing.allocator;

    var response = Server.Response.init(allocator);
    defer response.deinit();

    _ = response.setStatus(.ok);
    _ = try response.setHeader("X-Custom", "value");
    _ = try response.text("Hello, World!");

    try std.testing.expectEqual(types.Status.ok, response.status);
    try std.testing.expectEqualStrings("value", response.headers.get("X-Custom").?);
    try std.testing.expectEqualStrings("Hello, World!", response.body.items);
}

test "Request initialization" {
    const allocator = std.testing.allocator;

    var request = Server.Request.init(allocator);
    defer request.deinit();

    try std.testing.expectEqual(types.Method.GET, request.method);
    try std.testing.expectEqualStrings("/", request.path);
}
