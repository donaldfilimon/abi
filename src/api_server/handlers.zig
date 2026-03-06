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
            .body = "{\"status\":\"healthy\",\"version\":\"3.0.0\"}",
        };
    }

    fn handleMetrics(self: *Handlers) Response {
        // In a real server, we'd write to a dynamic buffer.
        // For now, return a static placeholder.
        _ = self;
        return .{
            .status = 200,
            .content_type = "text/plain; charset=utf-8",
            .body = "# Metrics endpoint active\n",
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

        self.metrics.recordRequest(true);
        self.metrics.recordTokens(50); // Placeholder

        // In a real implementation, we'd parse the JSON body, run inference,
        // and return a proper response. For now, return a well-formed stub.
        return .{
            .status = 200,
            .content_type = "application/json",
            .body =
            \\{"id":"chatcmpl-abi","object":"chat.completion","created":1709830400,
            \\"model":"abi-1","choices":[{"index":0,"message":{"role":"assistant",
            \\"content":"Hello! I'm the Abi framework. How can I help you today?"},
            \\"finish_reason":"stop"}],"usage":{"prompt_tokens":5,
            \\"completion_tokens":15,"total_tokens":20}}
            ,
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

        self.metrics.recordRequest(true);

        return .{
            .status = 200,
            .content_type = "application/json",
            .body =
            \\{"object":"list","data":[{"object":"embedding","index":0,
            \\"embedding":[0.0,0.0,0.0]}],"model":"abi-1",
            \\"usage":{"prompt_tokens":5,"total_tokens":5}}
            ,
        };
    }
};

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
        .body = "{}",
        .api_key = null,
        .content_type = "application/json",
    });
    try std.testing.expectEqual(@as(u16, 200), resp.status);
    try std.testing.expect(m.requests_total.load(.monotonic) == 1);
}
