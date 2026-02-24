//! OpenAI Embedding Backend
//!
//! Implements the embedding backend interface for OpenAI's embedding models.
//! Supports text-embedding-3-small, text-embedding-3-large, and text-embedding-ada-002.

const std = @import("std");
const backend = @import("../backend.zig");
const connectors = @import("../../../../services/connectors/mod.zig");
const openai_connector = @import("../../../../services/connectors/openai.zig");
const json_utils = @import("../../../../services/shared/utils.zig").json;
const async_http = @import("../../../../services/shared/utils.zig").async_http;

/// OpenAI embedding model configurations.
pub const Model = struct {
    /// text-embedding-3-small: Fast, cost-effective, 1536 dimensions
    pub const text_embedding_3_small = "text-embedding-3-small";
    /// text-embedding-3-large: Higher quality, 3072 dimensions
    pub const text_embedding_3_large = "text-embedding-3-large";
    /// text-embedding-ada-002: Legacy model, 1536 dimensions
    pub const text_embedding_ada_002 = "text-embedding-ada-002";

    pub fn defaultDimensions(model: []const u8) usize {
        if (std.mem.eql(u8, model, text_embedding_3_small)) return 1536;
        if (std.mem.eql(u8, model, text_embedding_3_large)) return 3072;
        if (std.mem.eql(u8, model, text_embedding_ada_002)) return 1536;
        return 1536; // Default
    }
};

/// OpenAI embedding backend implementation.
pub const OpenAIBackend = struct {
    allocator: std.mem.Allocator,
    api_key: []u8,
    base_url: []u8,
    model: []u8,
    http: async_http.AsyncHttpClient,

    /// Initialize OpenAI embedding backend.
    pub fn init(
        allocator: std.mem.Allocator,
        api_key: []const u8,
        model: []const u8,
    ) !*OpenAIBackend {
        return initWithUrl(allocator, api_key, "https://api.openai.com/v1", model);
    }

    /// Initialize with custom base URL (for Azure OpenAI or proxies).
    pub fn initWithUrl(
        allocator: std.mem.Allocator,
        api_key: []const u8,
        base_url: []const u8,
        model: []const u8,
    ) !*OpenAIBackend {
        const self = try allocator.create(OpenAIBackend);
        errdefer allocator.destroy(self);

        self.* = .{
            .allocator = allocator,
            .api_key = try allocator.dupe(u8, api_key),
            .base_url = try allocator.dupe(u8, base_url),
            .model = try allocator.dupe(u8, model),
            .http = try async_http.AsyncHttpClient.init(allocator),
        };

        return self;
    }

    /// Initialize from environment variables.
    pub fn initFromEnv(allocator: std.mem.Allocator) !*OpenAIBackend {
        const api_key = (try connectors.getFirstEnvOwned(allocator, &.{
            "ABI_OPENAI_API_KEY",
            "OPENAI_API_KEY",
        })) orelse return backend.BackendError.MissingCredentials;
        errdefer allocator.free(api_key);

        const base_url = (try connectors.getFirstEnvOwned(allocator, &.{
            "ABI_OPENAI_BASE_URL",
            "OPENAI_BASE_URL",
        })) orelse try allocator.dupe(u8, "https://api.openai.com/v1");
        errdefer allocator.free(base_url);

        const model = (try connectors.getFirstEnvOwned(allocator, &.{
            "ABI_OPENAI_EMBEDDING_MODEL",
            "OPENAI_EMBEDDING_MODEL",
        })) orelse try allocator.dupe(u8, Model.text_embedding_3_small);

        const self = try allocator.create(OpenAIBackend);
        self.* = .{
            .allocator = allocator,
            .api_key = api_key,
            .base_url = base_url,
            .model = model,
            .http = try async_http.AsyncHttpClient.init(allocator),
        };

        return self;
    }

    /// Clean up resources.
    pub fn deinit(self: *OpenAIBackend) void {
        self.http.deinit();
        // Secure cleanup for API key
        @memset(self.api_key, 0);
        self.allocator.free(self.api_key);
        self.allocator.free(self.base_url);
        self.allocator.free(self.model);
        self.allocator.destroy(self);
    }

    /// Convert to generic backend interface.
    pub fn asBackend(self: *OpenAIBackend) backend.EmbeddingBackend {
        return .{
            .ptr = self,
            .embedFn = embedWrapper,
            .embedBatchFn = embedBatchWrapper,
            .deinitFn = deinitWrapper,
            .backend_type = .openai,
            .name = "OpenAI",
            .model = self.model,
            .default_dimensions = Model.defaultDimensions(self.model),
        };
    }

    /// Generate embedding for single text.
    pub fn embed(
        self: *OpenAIBackend,
        allocator: std.mem.Allocator,
        text: []const u8,
        dimensions: usize,
    ) backend.BackendError![]f32 {
        const texts = [_][]const u8{text};
        const results = try self.embedBatch(allocator, &texts, dimensions);
        defer allocator.free(results);

        // Return first result (caller owns it)
        return results[0];
    }

    /// Generate embeddings for multiple texts.
    pub fn embedBatch(
        self: *OpenAIBackend,
        allocator: std.mem.Allocator,
        texts: []const []const u8,
        dimensions: usize,
    ) backend.BackendError![][]f32 {
        if (texts.len == 0) {
            return allocator.alloc([]f32, 0) catch return backend.BackendError.OutOfMemory;
        }

        // Build request JSON
        const json_body = self.buildRequestJson(texts, dimensions) catch
            return backend.BackendError.OutOfMemory;
        defer self.allocator.free(json_body);

        // Make API request
        const url = std.fmt.allocPrint(self.allocator, "{s}/embeddings", .{self.base_url}) catch
            return backend.BackendError.OutOfMemory;
        defer self.allocator.free(url);

        var http_req = async_http.HttpRequest.init(self.allocator, .POST, url) catch
            return backend.BackendError.RequestFailed;
        defer http_req.deinit();

        http_req.setBearerToken(self.api_key) catch return backend.BackendError.RequestFailed;
        http_req.setJsonBody(json_body) catch return backend.BackendError.RequestFailed;

        const http_res = self.http.fetchJson(&http_req) catch
            return backend.BackendError.RequestFailed;
        defer http_res.deinit();

        if (!http_res.isSuccess()) {
            if (http_res.status_code == 429) {
                return backend.BackendError.RateLimitExceeded;
            }
            return backend.BackendError.RequestFailed;
        }

        // Parse response
        return self.parseResponse(allocator, http_res.body, texts.len, dimensions) catch
            return backend.BackendError.InvalidResponse;
    }

    fn buildRequestJson(
        self: *OpenAIBackend,
        texts: []const []const u8,
        dimensions: usize,
    ) ![]u8 {
        var json_str = std.ArrayListUnmanaged(u8).empty;
        errdefer json_str.deinit(self.allocator);

        try json_str.appendSlice(self.allocator, "{\"model\":\"");
        try json_str.appendSlice(self.allocator, self.model);
        try json_str.appendSlice(self.allocator, "\",\"input\":[");

        for (texts, 0..) |text, i| {
            if (i > 0) try json_str.append(self.allocator, ',');
            try json_str.append(self.allocator, '"');
            // Escape JSON string
            for (text) |c| {
                switch (c) {
                    '"' => try json_str.appendSlice(self.allocator, "\\\""),
                    '\\' => try json_str.appendSlice(self.allocator, "\\\\"),
                    '\n' => try json_str.appendSlice(self.allocator, "\\n"),
                    '\r' => try json_str.appendSlice(self.allocator, "\\r"),
                    '\t' => try json_str.appendSlice(self.allocator, "\\t"),
                    else => try json_str.append(self.allocator, c),
                }
            }
            try json_str.append(self.allocator, '"');
        }

        try json_str.appendSlice(self.allocator, "]");

        // Add dimensions parameter for text-embedding-3 models
        if (std.mem.startsWith(u8, self.model, "text-embedding-3")) {
            try json_str.writer(self.allocator).print(",\"dimensions\":{d}", .{dimensions});
        }

        try json_str.append(self.allocator, '}');

        return json_str.toOwnedSlice(self.allocator);
    }

    fn parseResponse(
        self: *OpenAIBackend,
        allocator: std.mem.Allocator,
        json_body: []const u8,
        expected_count: usize,
        expected_dimensions: usize,
    ) ![][]f32 {
        _ = self;
        _ = expected_dimensions;

        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            allocator,
            json_body,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const object = try json_utils.getRequiredObject(parsed.value);
        const data_array = try json_utils.parseArrayField(object, "data");

        if (data_array.items.len != expected_count) {
            return error.InvalidResponse;
        }

        var results = try allocator.alloc([]f32, expected_count);
        var completed: usize = 0;
        errdefer {
            for (results[0..completed]) |embedding| {
                allocator.free(embedding);
            }
            allocator.free(results);
        }

        for (data_array.items) |item| {
            const item_obj = try json_utils.getRequiredObject(item);
            const embedding_array = try json_utils.parseArrayField(item_obj, "embedding");
            const index: usize = @intCast(try json_utils.parseIntField(item_obj, "index"));

            var embedding = try allocator.alloc(f32, embedding_array.items.len);
            errdefer allocator.free(embedding);

            for (embedding_array.items, 0..) |val, i| {
                embedding[i] = @floatCast(switch (val) {
                    .float => |f| f,
                    .integer => |int| @as(f64, @floatFromInt(int)),
                    else => return error.InvalidResponse,
                });
            }

            results[index] = embedding;
            completed += 1;
        }

        return results;
    }

    // Wrapper functions for vtable
    fn embedWrapper(
        ctx: *anyopaque,
        allocator: std.mem.Allocator,
        text: []const u8,
        dimensions: usize,
    ) backend.BackendError![]f32 {
        const self: *OpenAIBackend = @ptrCast(@alignCast(ctx));
        return self.embed(allocator, text, dimensions);
    }

    fn embedBatchWrapper(
        ctx: *anyopaque,
        allocator: std.mem.Allocator,
        texts: []const []const u8,
        dimensions: usize,
    ) backend.BackendError![][]f32 {
        const self: *OpenAIBackend = @ptrCast(@alignCast(ctx));
        return self.embedBatch(allocator, texts, dimensions);
    }

    fn deinitWrapper(ctx: *anyopaque) void {
        const self: *OpenAIBackend = @ptrCast(@alignCast(ctx));
        self.deinit();
    }
};

// ============================================================================
// Tests
// ============================================================================

test "model default dimensions" {
    try std.testing.expectEqual(@as(usize, 1536), Model.defaultDimensions(Model.text_embedding_3_small));
    try std.testing.expectEqual(@as(usize, 3072), Model.defaultDimensions(Model.text_embedding_3_large));
    try std.testing.expectEqual(@as(usize, 1536), Model.defaultDimensions(Model.text_embedding_ada_002));
    try std.testing.expectEqual(@as(usize, 1536), Model.defaultDimensions("unknown-model"));
}

test {
    std.testing.refAllDecls(@This());
}
