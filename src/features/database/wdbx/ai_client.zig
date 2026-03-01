//! WDBX Dynamic AI Client
//!
//! Sends texts to an OpenAI-compatible `/v1/embeddings` REST endpoint and returns
//! `[]f32` embedding slices. Uses `std.http.Client` for real HTTP connections.
//!
//! ## Usage
//! ```zig
//! var client = try AIClient.init(allocator, "https://api.openai.com", "sk-...", 30_000);
//! defer client.deinit();
//!
//! const embed = try client.generateEmbedding("hello world");
//! defer allocator.free(embed);
//! ```

const std = @import("std");

pub const AIClientError = error{
    Timeout,
    BadStatus,
    ParseError,
    EmptyResponse,
    BatchTooLarge,
    ConnectionFailed,
    RateLimited,
};

pub const AIClient = struct {
    allocator: std.mem.Allocator,
    base_url: []const u8,
    api_key: ?[]const u8,
    model: []const u8,
    timeout_ms: u32,
    max_batch_size: usize,

    pub fn init(
        allocator: std.mem.Allocator,
        base_url: []const u8,
        api_key: ?[]const u8,
        timeout_ms: u32,
    ) !AIClient {
        const cloned_url = try allocator.dupe(u8, base_url);
        errdefer allocator.free(cloned_url);

        const cloned_key = if (api_key) |k| try allocator.dupe(u8, k) else null;
        errdefer if (cloned_key) |k| allocator.free(k);

        const default_model = try allocator.dupe(u8, "text-embedding-3-small");
        errdefer allocator.free(default_model);

        return .{
            .allocator = allocator,
            .base_url = cloned_url,
            .api_key = cloned_key,
            .model = default_model,
            .timeout_ms = timeout_ms,
            .max_batch_size = 2048,
        };
    }

    pub fn deinit(self: *AIClient) void {
        self.allocator.free(self.base_url);
        if (self.api_key) |k| self.allocator.free(k);
        self.allocator.free(self.model);
    }

    pub fn setModel(self: *AIClient, model: []const u8) !void {
        const cloned = try self.allocator.dupe(u8, model);
        self.allocator.free(self.model);
        self.model = cloned;
    }

    /// Generate an embedding for a single text string.
    /// Caller owns the returned `[]f32`.
    pub fn generateEmbedding(self: *AIClient, text: []const u8) ![]f32 {
        // Build URL: base_url + "/v1/embeddings"
        const endpoint = "/v1/embeddings";
        const url_buf = try self.allocator.alloc(u8, self.base_url.len + endpoint.len);
        defer self.allocator.free(url_buf);
        @memcpy(url_buf[0..self.base_url.len], self.base_url);
        @memcpy(url_buf[self.base_url.len..], endpoint);

        // Build JSON request body.
        const body = try self.buildRequestBody(&[_][]const u8{text});
        defer self.allocator.free(body);

        // Execute HTTP request.
        const response_body = try self.executeHttpPost(url_buf, body);
        defer self.allocator.free(response_body);

        // Parse JSON response.
        return self.parseEmbeddingResponse(response_body);
    }

    /// Generate embeddings for a batch of texts.
    /// Each returned `[]f32` and the outer slice are caller-owned.
    pub fn generateEmbeddingsBatch(
        self: *AIClient,
        texts: []const []const u8,
    ) ![][]f32 {
        if (texts.len > self.max_batch_size) return AIClientError.BatchTooLarge;
        if (texts.len == 0) return try self.allocator.alloc([]f32, 0);

        const endpoint = "/v1/embeddings";
        const url_buf = try self.allocator.alloc(u8, self.base_url.len + endpoint.len);
        defer self.allocator.free(url_buf);
        @memcpy(url_buf[0..self.base_url.len], self.base_url);
        @memcpy(url_buf[self.base_url.len..], endpoint);

        const body = try self.buildRequestBody(texts);
        defer self.allocator.free(body);

        const response_body = try self.executeHttpPost(url_buf, body);
        defer self.allocator.free(response_body);

        return self.parseBatchEmbeddingResponse(response_body, texts.len);
    }

    // ─── HTTP transport ────────────────────────────────────────────────

    fn executeHttpPost(self: *AIClient, url: []const u8, body: []const u8) ![]u8 {
        var io_backend = std.Io.Threaded.init(self.allocator, .{ .environ = std.process.Environ.empty });
        defer io_backend.deinit();
        var client: std.http.Client = .{ .allocator = self.allocator, .io = io_backend.io() };
        defer client.deinit();

        const uri = std.Uri.parse(url) catch return AIClientError.ConnectionFailed;

        var request_options: std.http.Client.RequestOptions = .{};
        request_options.headers.content_type = .{ .override = "application/json" };

        // Track API key string so it outlives the request setup if present
        var auth_val: ?[]u8 = null;
        if (self.api_key) |key| {
            const auth_prefix = "Bearer ";
            auth_val = self.allocator.alloc(u8, auth_prefix.len + key.len) catch return AIClientError.ConnectionFailed;
            @memcpy(auth_val.?[0..auth_prefix.len], auth_prefix);
            @memcpy(auth_val.?[auth_prefix.len..], key);
            request_options.headers.authorization = .{ .override = auth_val.? };
        }
        defer {
            if (auth_val) |val| self.allocator.free(val);
        }

        var req = client.request(.POST, uri, request_options) catch return AIClientError.ConnectionFailed;
        defer req.deinit();

        var send_buffer: [4096]u8 = undefined;
        var body_writer = req.sendBody(&send_buffer) catch return AIClientError.ConnectionFailed;
        body_writer.writer.writeAll(body) catch return AIClientError.ConnectionFailed;
        body_writer.end() catch return AIClientError.ConnectionFailed;

        var redirect_buffer: [4096]u8 = undefined;
        var response = req.receiveHead(&redirect_buffer) catch return AIClientError.Timeout;

        // Check status.
        const status = response.head.status;
        if (status == .too_many_requests) return AIClientError.RateLimited;
        if (status != .ok) return AIClientError.BadStatus;

        // Read response body.
        const max_response = 16 * 1024 * 1024; // 16MB max

        // Try reading all.
        var list = std.ArrayListUnmanaged(u8).empty;
        errdefer list.deinit(self.allocator);

        var transfer_buffer: [4096]u8 = undefined;
        const reader = response.reader(&transfer_buffer);

        var read_buffer: [4096]u8 = undefined;
        while (true) {
            const n = reader.readSliceShort(read_buffer[0..]) catch return AIClientError.ParseError;
            if (n == 0) break;
            if (list.items.len + n > max_response) return AIClientError.ParseError;
            list.appendSlice(self.allocator, read_buffer[0..n]) catch return AIClientError.ParseError;
            if (n < read_buffer.len) break;
        }

        return list.toOwnedSlice(self.allocator) catch return AIClientError.ParseError;
    }

    // ─── JSON request building ─────────────────────────────────────────

    fn buildRequestBody(self: *AIClient, texts: []const []const u8) ![]u8 {
        var buf = std.ArrayListUnmanaged(u8){};
        errdefer buf.deinit(self.allocator);

        try buf.appendSlice(self.allocator, "{\"model\":\"");
        try writeJsonEscaped(&buf, self.allocator, self.model);
        try buf.appendSlice(self.allocator, "\",\"input\":");

        if (texts.len == 1) {
            try buf.append(self.allocator, '"');
            try writeJsonEscaped(&buf, self.allocator, texts[0]);
            try buf.append(self.allocator, '"');
        } else {
            try buf.append(self.allocator, '[');
            for (texts, 0..) |text, i| {
                if (i > 0) try buf.append(self.allocator, ',');
                try buf.append(self.allocator, '"');
                try writeJsonEscaped(&buf, self.allocator, text);
                try buf.append(self.allocator, '"');
            }
            try buf.append(self.allocator, ']');
        }

        try buf.append(self.allocator, '}');
        return buf.toOwnedSlice(self.allocator);
    }

    fn writeJsonEscaped(buf: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, s: []const u8) !void {
        for (s) |c| {
            switch (c) {
                '"' => try buf.appendSlice(allocator, "\\\""),
                '\\' => try buf.appendSlice(allocator, "\\\\"),
                '\n' => try buf.appendSlice(allocator, "\\n"),
                '\r' => try buf.appendSlice(allocator, "\\r"),
                '\t' => try buf.appendSlice(allocator, "\\t"),
                else => try buf.append(allocator, c),
            }
        }
    }

    // ─── JSON response parsing ─────────────────────────────────────────

    fn parseEmbeddingResponse(self: *AIClient, body: []const u8) ![]f32 {
        // Find "embedding": [ ... ] in the response.
        const embed_start = findEmbeddingArray(body) orelse return AIClientError.ParseError;
        return parseFloatArray(self.allocator, embed_start) catch return AIClientError.ParseError;
    }

    fn parseBatchEmbeddingResponse(self: *AIClient, body: []const u8, expected_count: usize) ![][]f32 {
        var results = try self.allocator.alloc([]f32, expected_count);
        var parsed: usize = 0;
        errdefer {
            for (results[0..parsed]) |r| self.allocator.free(r);
            self.allocator.free(results);
        }

        // Find each "embedding": [...] occurrence.
        var search_from: usize = 0;
        for (0..expected_count) |i| {
            const remaining = body[search_from..];
            const embed_start = findEmbeddingArray(remaining) orelse return AIClientError.ParseError;
            const abs_start = search_from + (@intFromPtr(embed_start.ptr) - @intFromPtr(remaining.ptr));
            results[i] = parseFloatArray(self.allocator, embed_start) catch return AIClientError.ParseError;
            parsed += 1;
            // Skip past this embedding array.
            search_from = abs_start + 1;
        }

        return results;
    }

    fn findEmbeddingArray(json: []const u8) ?[]const u8 {
        const key = "\"embedding\"";
        const pos = std.mem.indexOf(u8, json, key) orelse return null;
        const after_key = json[pos + key.len ..];
        // Skip whitespace and colon.
        var i: usize = 0;
        while (i < after_key.len and (after_key[i] == ' ' or after_key[i] == ':' or after_key[i] == '\n' or after_key[i] == '\r' or after_key[i] == '\t')) : (i += 1) {}
        if (i >= after_key.len or after_key[i] != '[') return null;
        return after_key[i..];
    }

    fn parseFloatArray(allocator: std.mem.Allocator, json: []const u8) ![]f32 {
        if (json.len == 0 or json[0] != '[') return error.InvalidCharacter;

        var values = std.ArrayListUnmanaged(f32){};
        errdefer values.deinit(allocator);

        var i: usize = 1; // Skip '['
        while (i < json.len) {
            // Skip whitespace and commas.
            while (i < json.len and (json[i] == ' ' or json[i] == ',' or json[i] == '\n' or json[i] == '\r' or json[i] == '\t')) : (i += 1) {}
            if (i >= json.len or json[i] == ']') break;

            // Find end of number.
            const start = i;
            while (i < json.len and json[i] != ',' and json[i] != ']' and json[i] != ' ' and json[i] != '\n') : (i += 1) {}
            const num_str = json[start..i];

            const val = std.fmt.parseFloat(f32, num_str) catch return error.InvalidCharacter;
            try values.append(allocator, val);
        }

        return values.toOwnedSlice(allocator);
    }
};

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

test "AIClient init and deinit" {
    var client = try AIClient.init(std.testing.allocator, "https://api.example.com", "key-abc", 5_000);
    defer client.deinit();

    try std.testing.expectEqualStrings("https://api.example.com", client.base_url);
    try std.testing.expectEqualStrings("key-abc", client.api_key.?);
    try std.testing.expectEqual(@as(u32, 5_000), client.timeout_ms);
}

test "AIClient init without api_key" {
    var client = try AIClient.init(std.testing.allocator, "http://localhost:8080", null, 3_000);
    defer client.deinit();
    try std.testing.expect(client.api_key == null);
}

test "AIClient setModel replaces model string" {
    var client = try AIClient.init(std.testing.allocator, "https://api.example.com", null, 1_000);
    defer client.deinit();

    try client.setModel("text-embedding-3-large");
    try std.testing.expectEqualStrings("text-embedding-3-large", client.model);
}

test "AIClient buildRequestBody single text" {
    var client = try AIClient.init(std.testing.allocator, "https://api.example.com", null, 1_000);
    defer client.deinit();

    const body = try client.buildRequestBody(&[_][]const u8{"hello world"});
    defer std.testing.allocator.free(body);

    try std.testing.expect(std.mem.indexOf(u8, body, "\"model\":\"text-embedding-3-small\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"input\":\"hello world\"") != null);
}

test "AIClient buildRequestBody batch" {
    var client = try AIClient.init(std.testing.allocator, "https://api.example.com", null, 1_000);
    defer client.deinit();

    const body = try client.buildRequestBody(&[_][]const u8{ "hello", "world" });
    defer std.testing.allocator.free(body);

    try std.testing.expect(std.mem.indexOf(u8, body, "[\"hello\",\"world\"]") != null);
}

test "AIClient buildRequestBody escapes special characters" {
    var client = try AIClient.init(std.testing.allocator, "https://api.example.com", null, 1_000);
    defer client.deinit();

    const body = try client.buildRequestBody(&[_][]const u8{"he said \"hi\"\n"});
    defer std.testing.allocator.free(body);

    try std.testing.expect(std.mem.indexOf(u8, body, "he said \\\"hi\\\"\\n") != null);
}

test "AIClient parseFloatArray" {
    const json = "[0.1, 0.2, 0.3, -0.5]";
    const result = try AIClient.parseFloatArray(std.testing.allocator, json);
    defer std.testing.allocator.free(result);

    try std.testing.expectEqual(@as(usize, 4), result.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), result[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), result[3], 0.001);
}

test "AIClient findEmbeddingArray" {
    const json =
        \\{"data":[{"embedding":[0.1, 0.2]}]}
    ;
    const found = AIClient.findEmbeddingArray(json);
    try std.testing.expect(found != null);
    try std.testing.expect(found.?[0] == '[');
}

test "AIClient batch too large" {
    var client = try AIClient.init(std.testing.allocator, "https://api.example.com", null, 1_000);
    defer client.deinit();
    client.max_batch_size = 2;
    const texts = [_][]const u8{ "a", "b", "c" };
    try std.testing.expectError(AIClientError.BatchTooLarge, client.generateEmbeddingsBatch(&texts));
}
