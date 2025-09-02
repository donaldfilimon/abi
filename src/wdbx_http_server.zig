//! WDBX Vector Database - HTTP REST API Server
//!
//! This module provides a comprehensive HTTP server for the WDBX vector database,
//! including vector operations, authentication, and monitoring endpoints.

const std = @import("std");
const database = @import("database.zig");
const http = std.http;

const version_string = "WDBX Vector Database v1.0.0";

/// HTTP server configuration
pub const ServerConfig = struct {
    host: []const u8 = "127.0.0.1",
    port: u16 = 8080,
    max_request_size: usize = 1024 * 1024, // 1MB
    rate_limit: usize = 1000, // requests per minute
    enable_cors: bool = true,
    enable_auth: bool = true,
    jwt_secret: []const u8 = "wdbx-secret-key-change-in-production",
};

/// HTTP server for WDBX vector database
pub const WdbxHttpServer = struct {
    allocator: std.mem.Allocator,
    config: ServerConfig,
    db: ?*database.Db,
    server: http.Server,
    rate_limiter: RateLimiter,

    const Self = @This();

    /// Rate limiter for API endpoints
    const RateLimiter = struct {
        requests: std.AutoHashMap(u32, u64), // IP -> request count
        last_reset: i64,
        max_requests: usize,
        window_ms: i64 = 60 * 1000, // 1 minute

        pub fn init(allocator: std.mem.Allocator, max_requests: usize) RateLimiter {
            return .{
                .requests = std.AutoHashMap(u32, u64).init(allocator),
                .last_reset = std.time.milliTimestamp(),
                .max_requests = max_requests,
            };
        }

        pub fn deinit(self: *RateLimiter) void {
            self.requests.deinit();
        }

        pub fn checkLimit(self: *RateLimiter, ip: u32) !bool {
            const now = std.time.milliTimestamp();

            // Reset counter if window has passed
            if (now - self.last_reset > self.window_ms) {
                self.requests.clearRetainingCapacity();
                self.last_reset = now;
            }

            // Check current request count
            const current = self.requests.get(ip) orelse 0;
            if (current >= self.max_requests) {
                return false; // Rate limited
            }

            // Increment counter
            try self.requests.put(ip, current + 1);
            return true; // Allowed
        }
    };

    /// Initialize HTTP server
    pub fn init(allocator: std.mem.Allocator, config: ServerConfig) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        self.* = .{
            .allocator = allocator,
            .config = config,
            .db = null,
            .server = http.Server.init(.{
                .allocator = allocator,
                .reuse_address = true,
            }),
            .rate_limiter = RateLimiter.init(allocator, config.rate_limit),
        };

        return self;
    }

    /// Deinitialize HTTP server
    pub fn deinit(self: *Self) void {
        if (self.db) |db| {
            db.close();
        }
        self.server.deinit();
        self.rate_limiter.deinit();
        self.allocator.destroy(self);
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
        try self.server.listen(address);

        std.debug.print("WDBX HTTP server listening on {}:{}\n", .{ self.config.host, self.config.port });

        while (true) {
            var response = try self.server.accept(.{});
            defer response.deinit();

            try self.handleRequest(&response);
        }
    }

    /// Handle HTTP request
    fn handleRequest(self: *Self, response: *http.Server.Response) !void {
        // Parse request
        const request = response.request;
        _ = request.method; // Will be used for method-specific routing
        const uri = request.target;

        // Rate limiting
        const client_ip = self.getClientIP(response);
        if (!try self.rate_limiter.checkLimit(client_ip)) {
            try self.sendError(response, 429, "Too Many Requests");
            return;
        }

        // CORS headers
        if (self.config.enable_cors) {
            try self.addCorsHeaders(response);
        }

        // Route request
        if (std.mem.eql(u8, uri, "/")) {
            try self.handleRoot(response);
        } else if (std.mem.eql(u8, uri, "/health")) {
            try self.handleHealth(response);
        } else if (std.mem.eql(u8, uri, "/stats")) {
            try self.handleStats(response);
        } else if (std.mem.startsWith(u8, uri, "/add")) {
            try self.handleAdd(response);
        } else if (std.mem.startsWith(u8, uri, "/query")) {
            try self.handleQuery(response);
        } else if (std.mem.startsWith(u8, uri, "/knn")) {
            try self.handleKnn(response);
        } else if (std.mem.startsWith(u8, uri, "/monitor")) {
            try self.handleMonitor(response);
        } else {
            try self.sendError(response, 404, "Not Found");
        }
    }

    /// Handle root endpoint
    fn handleRoot(_: *Self, response: *http.Server.Response) !void {
        const html =
            \\<!DOCTYPE html>
            \\<html>
            \\<head>
            \\    <title>WDBX Vector Database</title>
            \\    <style>
            \\        body { font-family: Arial, sans-serif; margin: 40px; }
            \\        .container { max-width: 800px; margin: 0 auto; }
            \\        h1 { color: #333; }
            \\        .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            \\        .method { color: #007acc; font-weight: bold; }
            \\        .url { font-family: monospace; background: #e8e8e8; padding: 2px 6px; }
            \\    </style>
            \\</head>
            \\<body>
            \\    <div class="container">
            \\        <h1>WDBX Vector Database API</h1>
            \\        <p>Welcome to the WDBX Vector Database HTTP API.</p>
            \\
            \\        <h2>Available Endpoints</h2>
            \\
            \\        <div class="endpoint">
            \\            <span class="method">GET</span> <span class="url">/health</span>
            \\            <p>Check server health status</p>
            \\        </div>
            \\
            \\        <div class="endpoint">
            \\            <span class="method">GET</span> <span class="url">/stats</span>
            \\            <p>Get database statistics</p>
            \\        </div>
            \\
            \\        <div class="endpoint">
            \\            <span class="method">POST</span> <span class="url">/add</span>
            \\            <p>Add a vector to the database (requires admin token)</p>
            \\        </div>
            \\
            \\        <div class="endpoint">
            \\            <span class="method">GET</span> <span class="url">/query?vec=1.0,2.0,3.0</span>
            \\            <p>Query nearest neighbor</p>
            \\        </div>
            \\
            \\        <div class="endpoint">
            \\            <span class="method">GET</span> <span class="url">/knn?vec=1.0,2.0,3.0&k=5</span>
            \\            <p>Query k-nearest neighbors</p>
            \\        </div>
            \\
            \\        <div class="endpoint">
            \\            <span class="method">GET</span> <span class="url">/monitor</span>
            \\            <p>Get performance metrics</p>
            \\        </div>
            \\
            \\        <h2>Authentication</h2>
            \\        <p>Admin operations require a JWT token in the Authorization header:</p>
            \\        <code>Authorization: Bearer &lt;your-jwt-token&gt;</code>
            \\
            \\        <h2>Vector Format</h2>
            \\        <p>Vectors should be comma-separated float values, e.g.: <code>1.0,2.0,3.0,4.0</code></p>
            \\    </div>
            \\</body>
            \\</html>
        ;

        try response.headers.append("Content-Type", "text/html");
        try response.do();
        try response.writer().writeAll(html);
    }

    /// Handle health check endpoint
    fn handleHealth(self: *Self, response: *http.Server.Response) !void {
        const health = .{
            .status = "healthy",
            .version = version_string,
            .timestamp = std.time.milliTimestamp(),
            .database_connected = self.db != null,
        };

        try response.headers.append("Content-Type", "application/json");
        try response.do();
        try response.writer().print("{{\"status\":\"{s}\",\"version\":\"{s}\",\"timestamp\":{d},\"database_connected\":{any}}}", .{ health.status, health.version, health.timestamp, health.database_connected });
    }

    /// Handle statistics endpoint
    fn handleStats(self: *Self, response: *http.Server.Response) !void {
        if (self.db == null) {
            try self.sendError(response, 503, "Database not connected");
            return;
        }

        const db = self.db.?;
        const stats = db.getStats();
        const db_stats = .{
            .vectors_stored = db.getRowCount(),
            .vector_dimension = db.getDimension(),
            .searches_performed = stats.search_count,
            .average_search_time_us = stats.getAverageSearchTime(),
            .writes_performed = stats.write_count,
            .initializations = stats.initialization_count,
        };

        try response.headers.append("Content-Type", "application/json");
        try response.do();
        try response.writer().print("{{\"vectors_stored\":{d},\"vector_dimension\":{d},\"searches_performed\":{d},\"average_search_time_us\":{d},\"writes_performed\":{d},\"initializations\":{d}}}", .{ db_stats.vectors_stored, db_stats.vector_dimension, db_stats.searches_performed, db_stats.average_search_time_us, db_stats.writes_performed, db_stats.initializations });
    }

    /// Handle add vector endpoint
    fn handleAdd(self: *Self, response: *http.Server.Response) !void {
        if (self.db == null) {
            try self.sendError(response, 503, "Database not connected");
            return;
        }

        // Check authentication for admin operations
        if (self.config.enable_auth) {
            const auth_header = response.request.headers.get("authorization") orelse {
                try self.sendError(response, 401, "Authorization required");
                return;
            };

            if (!std.mem.startsWith(u8, auth_header, "Bearer ")) {
                try self.sendError(response, 401, "Invalid authorization format");
                return;
            }

            // TODO: Implement JWT validation
            const token = auth_header[7..];
            if (!self.validateJWT(token)) {
                try self.sendError(response, 403, "Invalid or expired token");
                return;
            }
        }

        // Parse request body
        const body = try self.readRequestBody(response);
        defer self.allocator.free(body);

        // Parse vector
        const vector = try self.parseVector(body);
        defer self.allocator.free(vector);

        // Add to database
        const db = self.db.?;
        const row_id = try db.addEmbedding(vector);

        const result = .{
            .success = true,
            .row_id = row_id,
            .message = "Vector added successfully",
        };

        try response.headers.append("Content-Type", "application/json");
        try response.do();
        try response.writer().print("{{\"success\":{any},\"row_id\":{d},\"message\":\"{s}\"}}", .{ result.success, result.row_id, result.message });
    }

    /// Handle query endpoint
    fn handleQuery(self: *Self, response: *http.Server.Response) !void {
        if (self.db == null) {
            try self.sendError(response, 503, "Database not connected");
            return;
        }

        // Parse query parameters
        const query = response.request.target;
        const vec_start = std.mem.indexOf(u8, query, "vec=") orelse {
            try self.sendError(response, 400, "Missing 'vec' parameter");
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
            const result = .{
                .success = true,
                .nearest_neighbor = .{
                    .index = results[0].index,
                    .distance = results[0].score,
                },
            };

            try response.headers.append("Content-Type", "application/json");
            try response.do();
            try response.writer().print("{{\"success\":{any},\"nearest_neighbor\":{{\"index\":{d},\"distance\":{d}}}}}", .{ result.success, result.nearest_neighbor.index, result.nearest_neighbor.distance });
        } else {
            try self.sendError(response, 404, "No vectors found in database");
        }
    }

    /// Handle k-nearest neighbors endpoint
    fn handleKnn(self: *Self, response: *http.Server.Response) !void {
        if (self.db == null) {
            try self.sendError(response, 503, "Database not connected");
            return;
        }

        // Parse query parameters
        const query = response.request.target;
        const vec_start = std.mem.indexOf(u8, query, "vec=") orelse {
            try self.sendError(response, 400, "Missing 'vec' parameter");
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
        var neighbors = try std.ArrayList(struct {
            index: u64,
            distance: f32,
        }).initCapacity(self.allocator, results.len);
        defer neighbors.deinit(self.allocator);

        for (results) |result| {
            try neighbors.append(self.allocator, .{
                .index = result.index,
                .distance = result.score,
            });
        }

        const result = .{
            .success = true,
            .k = k,
            .neighbors = neighbors.items,
        };

        try response.headers.append("Content-Type", "application/json");
        try response.do();
        try response.writer().print("{{\"success\":{any},\"k\":{d},\"neighbors\":[{}]}", .{ result.success, result.k, try self.formatNeighbors(result.neighbors) });
    }

    /// Handle monitor endpoint
    fn handleMonitor(self: *Self, response: *http.Server.Response) !void {
        const metrics = .{
            .server_uptime_ms = std.time.milliTimestamp(),
            .rate_limit_enabled = self.config.enable_auth,
            .cors_enabled = self.config.enable_cors,
            .max_request_size = self.config.max_request_size,
            .rate_limit_per_minute = self.config.rate_limit,
        };

        try response.headers.append("Content-Type", "application/json");
        try response.do();
        try response.writer().print("{{\"server_uptime_ms\":{d},\"rate_limit_enabled\":{any},\"cors_enabled\":{any},\"max_request_size\":{d},\"rate_limit_per_minute\":{d}}}", .{ metrics.server_uptime_ms, metrics.rate_limit_enabled, metrics.cors_enabled, metrics.max_request_size, metrics.rate_limit_per_minute });
    }

    /// Send error response
    fn sendError(_: *Self, response: *http.Server.Response, status: u16, message: []const u8) !void {
        response.status = @as(u16, @intCast(status));
        try response.headers.append("Content-Type", "application/json");
        try response.do();
        try response.writer().print("{{\"error\":{any},\"status\":{d},\"message\":\"{s}\"}}", .{ true, status, message });
    }

    /// Add CORS headers
    fn addCorsHeaders(_: *Self, response: *http.Server.Response) !void {
        try response.headers.append("Access-Control-Allow-Origin", "*");
        try response.headers.append("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        try response.headers.append("Access-Control-Allow-Headers", "Content-Type, Authorization");
    }

    /// Get client IP address
    fn getClientIP(_: *Self, _: *http.Server.Response) u32 {
        // Simplified IP extraction - in production, handle X-Forwarded-For, etc.
        return 0x7f000001; // 127.0.0.1 for now
    }

    /// Read request body
    fn readRequestBody(self: *Self, response: *http.Server.Response) ![]u8 {
        const content_length = response.request.headers.get("content-length") orelse "0";
        const length = try std.fmt.parseInt(usize, content_length, 10);

        if (length > self.config.max_request_size) {
            return error.RequestTooLarge;
        }

        const body = try self.allocator.alloc(u8, length);
        errdefer self.allocator.free(body);

        var total_read: usize = 0;
        while (total_read < length) {
            const read = try response.reader().read(body[total_read..]);
            if (read == 0) break;
            total_read += read;
        }

        return body[0..total_read];
    }

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

    /// Validate JWT token
    fn validateJWT(self: *Self, token: []const u8) bool {
        // Basic JWT validation implementation
        // In production, use a proper JWT library

        // Check token format (header.payload.signature)
        var parts_iter = std.mem.split(u8, token, ".");
        var parts_count: usize = 0;
        while (parts_iter.next()) |_| {
            parts_count += 1;
        }

        if (parts_count != 3) {
            return false;
        }

        // Reset iterator
        parts_iter = std.mem.split(u8, token, ".");
        const header = parts_iter.next() orelse return false;
        const payload = parts_iter.next() orelse return false;
        const signature = parts_iter.next() orelse return false;

        // Validate each part has content
        if (header.len == 0 or payload.len == 0 or signature.len == 0) {
            return false;
        }

        // Check if token exists in our valid tokens map
        if (self.auth_tokens.get(token)) |auth_info| {
            // Check expiration
            const current_time = std.time.milliTimestamp();
            if (auth_info.expires_at > 0 and current_time > auth_info.expires_at) {
                return false;
            }
            return true;
        }

        return false;
    }

    /// Format neighbors array for JSON output
    fn formatNeighbors(self: *Self, neighbors: []const struct { index: u64, distance: f32 }) ![]const u8 {
        var buffer = try std.ArrayList(u8).initCapacity(self.allocator, 256);
        defer buffer.deinit(self.allocator);

        for (neighbors, 0..) |neighbor, i| {
            if (i > 0) try buffer.appendSlice(self.allocator, ",");
            try buffer.writer(self.allocator).print("{{\"index\":{d},\"distance\":{d}}}", .{ neighbor.index, neighbor.distance });
        }

        return try buffer.toOwnedSlice(self.allocator);
    }
};
