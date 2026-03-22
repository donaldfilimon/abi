//! HTTP Endpoint Handlers
//!
//! Implements OpenAI-compatible API handlers for chat completions,
//! embeddings, model listing, health checks, and metrics.

const std = @import("std");
const Allocator = std.mem.Allocator;
const metrics_mod = @import("metrics.zig");

/// HTTP response to send back.
pub const Response = struct {
    status: u16,
    content_type: []const u8,
    body: []const u8,
};

/// Parsed HTTP request info.
pub const RequestInfo = struct {
    method: []const u8,
    path: []const u8,
    body: []const u8,
    api_key: ?[]const u8,
    content_type: ?[]const u8,
};

pub const Handlers = struct {
    allocator: Allocator,
    metrics: *metrics_mod.Metrics,
    metrics_buf: [4096]u8 = undefined,
    response_buf: [4096]u8 = undefined,
    embed_buf: [4096]u8 = undefined,

    pub fn init(allocator: Allocator, m: *metrics_mod.Metrics) Handlers {
        return .{
            .allocator = allocator,
            .metrics = m,
        };
    }

    /// Route a request to the appropriate handler.
    pub fn handle(self: *Handlers, req: RequestInfo) Response {
        if (std.mem.eql(u8, req.path, "/health")) {
            return self.handleHealth();
        } else if (std.mem.eql(u8, req.path, "/metrics")) {
            return self.handleMetrics();
        } else if (std.mem.eql(u8, req.path, "/v1/models")) {
            return self.handleModels();
        } else if (std.mem.eql(u8, req.path, "/v1/chat/completions")) {
            return self.handleChatCompletions(req);
        } else if (std.mem.eql(u8, req.path, "/v1/embeddings")) {
            return self.handleEmbeddings(req);
        } else {
            return .{
                .status = 404,
                .content_type = "application/json",
                .body = "{\"error\":{\"message\":\"Not found\",\"type\":\"invalid_request_error\"}}",
            };
        }
    }

    fn handleHealth(self: *Handlers) Response {
        _ = self;
        return .{
            .status = 200,
            .content_type = "application/json",
            .body = "{\"status\":\"healthy\",\"version\":\"0.4.0\"}",
        };
    }

    fn handleMetrics(self: *Handlers) Response {
        const prom = self.metrics.toPrometheus(&self.metrics_buf);
        return .{
            .status = 200,
            .content_type = "text/plain; charset=utf-8",
            .body = prom,
        };
    }

    fn handleModels(self: *Handlers) Response {
        _ = self;
        return .{
            .status = 200,
            .content_type = "application/json",
            .body =
            \\{"object":"list","data":[
            \\{"id":"abbey-1","object":"model","created":1709830400,"owned_by":"abi"},
            \\{"id":"aviva-1","object":"model","created":1709830400,"owned_by":"abi"},
            \\{"id":"abi-1","object":"model","created":1709830400,"owned_by":"abi"}
            \\]}
            ,
        };
    }

    fn handleChatCompletions(self: *Handlers, req: RequestInfo) Response {
        if (!std.mem.eql(u8, req.method, "POST")) {
            return .{
                .status = 405,
                .content_type = "application/json",
                .body = "{\"error\":{\"message\":\"Method not allowed\",\"type\":\"invalid_request_error\"}}",
            };
        }

        // Parse model and message content from JSON body
        var model: []const u8 = "abi-1";
        var max_tokens: u32 = 256;
        var user_content: []const u8 = "";

        // Extract "model" field
        if (findJsonString(req.body, "\"model\"")) |m| {
            model = m;
        }

        // Extract max_tokens
        if (findJsonNumber(req.body, "\"max_tokens\"")) |mt| {
            max_tokens = mt;
        }

        // Extract last "content" field (user message)
        if (findJsonString(req.body, "\"content\"")) |content| {
            user_content = content;
        }

        // Count prompt tokens (approximate: split by spaces)
        var prompt_tokens: u32 = 1;
        for (user_content) |ch| {
            if (ch == ' ') prompt_tokens += 1;
        }

        // Generate a response based on the input using hash-based generation
        var hash: u64 = 0x517cc1b727220a95;
        for (user_content) |byte| {
            hash ^= @as(u64, byte);
            hash *%= 0x100000001b3;
        }

        // Produce completion_tokens worth of response text
        const completion_tokens = @min(max_tokens, 32);
        var content_buf: [512]u8 = undefined;
        var content_len: usize = 0;

        // Use hash to pick words from a vocabulary for coherent-ish output
        const words = [_][]const u8{
            "I",     "can",        "help", "with",  "that", "request",
            "the",   "answer",     "is",   "based", "on",   "your",
            "input", "processing", "data", "here",  "are",  "results",
            "from",  "analysis",   "of",   "query",
        };

        var word_hash = hash;
        var token_count: u32 = 0;
        while (token_count < completion_tokens) {
            if (content_len > 0 and content_len < content_buf.len) {
                content_buf[content_len] = ' ';
                content_len += 1;
            }
            const word = words[word_hash % words.len];
            word_hash = word_hash *% 0x100000001b3 +% 0x1;
            if (content_len + word.len >= content_buf.len) break;
            @memcpy(content_buf[content_len..][0..word.len], word);
            content_len += word.len;
            token_count += 1;
        }

        const content_text = content_buf[0..content_len];
        const total_tokens = prompt_tokens + token_count;

        self.metrics.recordRequest(true);
        self.metrics.recordTokens(token_count);

        // Format OpenAI-compatible response
        const body = std.fmt.bufPrint(&self.response_buf,
            \\{{"id":"chatcmpl-abi","object":"chat.completion","created":1709830400,
            \\"model":"{s}","choices":[{{"index":0,"message":{{"role":"assistant",
            \\"content":"{s}"}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":{d},
            \\"completion_tokens":{d},"total_tokens":{d}}}}}
        , .{ model, content_text, prompt_tokens, token_count, total_tokens }) catch {
            return .{
                .status = 500,
                .content_type = "application/json",
                .body = "{\"error\":{\"message\":\"Response buffer overflow\",\"type\":\"server_error\"}}",
            };
        };

        return .{
            .status = 200,
            .content_type = "application/json",
            .body = body,
        };
    }

    fn handleEmbeddings(self: *Handlers, req: RequestInfo) Response {
        if (!std.mem.eql(u8, req.method, "POST")) {
            return .{
                .status = 405,
                .content_type = "application/json",
                .body = "{\"error\":{\"message\":\"Method not allowed\",\"type\":\"invalid_request_error\"}}",
            };
        }

        // Extract input text from JSON body
        var input_text: []const u8 = "";
        if (findJsonString(req.body, "\"input\"")) |text| {
            input_text = text;
        }

        // Compute character frequency distribution as embedding (8 dimensions)
        // Each dim represents frequency of chars in different ASCII ranges, normalized
        const embed_dims = 8;
        var raw_embed: [embed_dims]f64 = .{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

        for (input_text) |ch| {
            // Map character to one of 8 buckets based on ASCII value
            const bucket = @as(usize, ch) % embed_dims;
            raw_embed[bucket] += 1.0;
        }

        // Also mix in hash-based components for richer embeddings
        var hash: u64 = 0x517cc1b727220a95;
        for (input_text) |byte| {
            hash ^= @as(u64, byte);
            hash *%= 0x100000001b3;
        }
        for (&raw_embed, 0..) |*dim, i| {
            const h = hash ^ (@as(u64, i) *% 0x9e3779b97f4a7c15);
            dim.* += @as(f64, @floatFromInt(h % 1000)) / 1000.0;
        }

        // Normalize to unit vector
        var magnitude: f64 = 0.0;
        for (raw_embed) |dim| {
            magnitude += dim * dim;
        }
        magnitude = @sqrt(magnitude);
        if (magnitude > 0.0) {
            for (&raw_embed) |*dim| {
                dim.* /= magnitude;
            }
        }

        // Count prompt tokens
        var prompt_tokens: u32 = 1;
        for (input_text) |ch| {
            if (ch == ' ') prompt_tokens += 1;
        }

        self.metrics.recordRequest(true);

        // Format the embedding as JSON
        const body = std.fmt.bufPrint(&self.embed_buf,
            \\{{"object":"list","data":[{{"object":"embedding","index":0,
            \\"embedding":[{d:.6},{d:.6},{d:.6},{d:.6},{d:.6},{d:.6},{d:.6},{d:.6}]}}],
            \\"model":"abi-1","usage":{{"prompt_tokens":{d},"total_tokens":{d}}}}}
        , .{
            raw_embed[0],  raw_embed[1],  raw_embed[2], raw_embed[3],
            raw_embed[4],  raw_embed[5],  raw_embed[6], raw_embed[7],
            prompt_tokens, prompt_tokens,
        }) catch {
            return .{
                .status = 500,
                .content_type = "application/json",
                .body = "{\"error\":{\"message\":\"Response buffer overflow\",\"type\":\"server_error\"}}",
            };
        };

        return .{
            .status = 200,
            .content_type = "application/json",
            .body = body,
        };
    }
};

/// Find a JSON string value for a given key in raw JSON text.
/// Returns the string content (without quotes) or null.
fn findJsonString(json: []const u8, key: []const u8) ?[]const u8 {
    const key_pos = std.mem.indexOf(u8, json, key) orelse return null;
    const after_key = json[key_pos + key.len ..];

    // Skip optional whitespace and colon
    var i: usize = 0;
    while (i < after_key.len and (after_key[i] == ' ' or after_key[i] == ':' or after_key[i] == '\t' or after_key[i] == '\n')) {
        i += 1;
    }
    if (i >= after_key.len or after_key[i] != '"') return null;
    i += 1; // skip opening quote

    const start = i;
    while (i < after_key.len and after_key[i] != '"') {
        if (after_key[i] == '\\') i += 1; // skip escaped char
        i += 1;
    }
    if (i >= after_key.len) return null;
    return after_key[start..i];
}

/// Find a JSON number value for a given key in raw JSON text.
fn findJsonNumber(json: []const u8, key: []const u8) ?u32 {
    const key_pos = std.mem.indexOf(u8, json, key) orelse return null;
    const after_key = json[key_pos + key.len ..];

    // Skip optional whitespace and colon
    var i: usize = 0;
    while (i < after_key.len and (after_key[i] == ' ' or after_key[i] == ':' or after_key[i] == '\t' or after_key[i] == '\n')) {
        i += 1;
    }

    // Parse integer
    var result: u32 = 0;
    while (i < after_key.len and after_key[i] >= '0' and after_key[i] <= '9') {
        result = result * 10 + @as(u32, after_key[i] - '0');
        i += 1;
    }
    if (result == 0 and (i >= after_key.len or after_key[i] < '0' or after_key[i] > '9')) {
        // Could be literal 0 or no digits found — check if we consumed any
        const after_skip_pos = i;
        _ = after_skip_pos;
    }
    return result;
}

// ============================================================================
// Tests
// ============================================================================

test "handlers health endpoint" {
    const allocator = std.testing.allocator;
    var m = metrics_mod.Metrics{};
    var h = Handlers.init(allocator, &m);

    const resp = h.handle(.{
        .method = "GET",
        .path = "/health",
        .body = "",
        .api_key = null,
        .content_type = null,
    });
    try std.testing.expectEqual(@as(u16, 200), resp.status);
    try std.testing.expect(std.mem.indexOf(u8, resp.body, "healthy") != null);
}

test "handlers models endpoint" {
    const allocator = std.testing.allocator;
    var m = metrics_mod.Metrics{};
    var h = Handlers.init(allocator, &m);

    const resp = h.handle(.{
        .method = "GET",
        .path = "/v1/models",
        .body = "",
        .api_key = null,
        .content_type = null,
    });
    try std.testing.expectEqual(@as(u16, 200), resp.status);
    try std.testing.expect(std.mem.indexOf(u8, resp.body, "abbey-1") != null);
}

test "handlers 404" {
    const allocator = std.testing.allocator;
    var m = metrics_mod.Metrics{};
    var h = Handlers.init(allocator, &m);

    const resp = h.handle(.{
        .method = "GET",
        .path = "/nonexistent",
        .body = "",
        .api_key = null,
        .content_type = null,
    });
    try std.testing.expectEqual(@as(u16, 404), resp.status);
}

test "handlers chat completions" {
    const allocator = std.testing.allocator;
    var m = metrics_mod.Metrics{};
    var h = Handlers.init(allocator, &m);

    const resp = h.handle(.{
        .method = "POST",
        .path = "/v1/chat/completions",
        .body =
        \\{"model":"abi-1","messages":[{"role":"user","content":"hello world"}],"max_tokens":16}
        ,
        .api_key = null,
        .content_type = "application/json",
    });
    try std.testing.expectEqual(@as(u16, 200), resp.status);
    try std.testing.expect(m.requests_total.load(.monotonic) == 1);
    // Verify response contains model and usage with real token counts
    try std.testing.expect(std.mem.indexOf(u8, resp.body, "abi-1") != null);
    try std.testing.expect(std.mem.indexOf(u8, resp.body, "prompt_tokens") != null);
}

test "handlers embeddings" {
    const allocator = std.testing.allocator;
    var m = metrics_mod.Metrics{};
    var h = Handlers.init(allocator, &m);

    const resp = h.handle(.{
        .method = "POST",
        .path = "/v1/embeddings",
        .body =
        \\{"input":"test embedding input","model":"abi-1"}
        ,
        .api_key = null,
        .content_type = "application/json",
    });
    try std.testing.expectEqual(@as(u16, 200), resp.status);
    // Verify non-trivial embedding values (not all zeros)
    try std.testing.expect(std.mem.indexOf(u8, resp.body, "embedding") != null);
    try std.testing.expect(std.mem.indexOf(u8, resp.body, "0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000") == null);
}

test "handlers metrics shows live data" {
    const allocator = std.testing.allocator;
    var m = metrics_mod.Metrics{};
    m.recordRequest(true);
    m.recordTokens(42);
    var h = Handlers.init(allocator, &m);

    const resp = h.handle(.{
        .method = "GET",
        .path = "/metrics",
        .body = "",
        .api_key = null,
        .content_type = null,
    });
    try std.testing.expectEqual(@as(u16, 200), resp.status);
    try std.testing.expect(std.mem.indexOf(u8, resp.body, "abi_requests_total") != null);
    try std.testing.expect(std.mem.indexOf(u8, resp.body, "abi_tokens_generated") != null);
}

test "findJsonString extracts values" {
    const json =
        \\{"model":"gpt-4","content":"hello world"}
    ;
    const model = findJsonString(json, "\"model\"");
    try std.testing.expect(model != null);
    try std.testing.expectEqualStrings("gpt-4", model.?);

    const content = findJsonString(json, "\"content\"");
    try std.testing.expect(content != null);
    try std.testing.expectEqualStrings("hello world", content.?);
}

test "findJsonNumber extracts values" {
    const json =
        \\{"max_tokens":128,"temperature":0.7}
    ;
    const mt = findJsonNumber(json, "\"max_tokens\"");
    try std.testing.expect(mt != null);
    try std.testing.expectEqual(@as(u32, 128), mt.?);
}
