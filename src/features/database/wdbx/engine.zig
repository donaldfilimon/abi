//! WDBX High-Level Engine
//!
//! Orchestrates the semantic configurations, caching layer optimizations,
//! embedded dynamic AI client HTTP inferences, and search query executions.

const std = @import("std");
const config = @import("config.zig");
const metrics = @import("distance.zig").Distance;
const Cache = @import("cache.zig").Cache;
const HNSW = @import("hnsw.zig").HNSW;
const AIClient = @import("ai_client.zig").AIClient;

pub const Metadata = struct {
    text: []const u8,
    category: ?[]const u8 = null,
    tags: []const []const u8 = &.{},
    score: f32 = 1.0,
    extra: ?[]const u8 = null,
};

pub const SearchOptions = struct {
    k: usize = 10,
    ef: u16 = 64,
    filter_category: ?[]const u8 = null,
    filter_tags: ?[]const []const u8 = null,
    min_score: f32 = 0.0,
};

pub const SearchResult = struct {
    id: []const u8,
    similarity: f32,
    distance: f32,
    metadata: Metadata,
    vector: []const f32,
};

// Internal tracking layout (pub for persistence access).
pub const EngineVector = struct {
    id: []const u8,
    vec: []const f32,
    metadata: Metadata,
};

pub const Engine = struct {
    allocator: std.mem.Allocator,
    config: config.Config,
    cache: Cache,
    ai_client: ?AIClient = null,
    hnsw_index: HNSW,
    vectors_array: std.ArrayListUnmanaged(EngineVector) = .empty,

    pub fn init(allocator: std.mem.Allocator, cfg: config.Config) !Engine {
        try cfg.validateRuntime();

        const cache = try Cache.init(allocator, cfg.cache.capacity, cfg.cache.segments);
        errdefer cache.deinit();

        const idx = try HNSW.init(allocator, cfg, cfg.metric);
        errdefer idx.deinit();

        return .{
            .allocator = allocator,
            .config = cfg,
            .cache = cache,
            .hnsw_index = idx,
        };
    }

    pub fn deinit(self: *Engine) void {
        self.cache.deinit();
        self.hnsw_index.deinit();
        if (self.ai_client) |*c| c.deinit();

        for (self.vectors_array.items) |item| {
            self.allocator.free(item.id);
            self.allocator.free(item.vec);
            self.deinitOwnedMetadata(item.metadata);
        }
        self.vectors_array.deinit(self.allocator);
    }

    pub fn connectAI(self: *Engine, base_url: []const u8, api_key: ?[]const u8) !void {
        self.ai_client = try AIClient.init(self.allocator, base_url, api_key, self.config.network.request_timeout_ms);
    }

    pub fn index(
        self: *Engine,
        id: []const u8,
        text: []const u8,
        metadata: Metadata,
    ) !void {
        var embedding: []const f32 = undefined;
        var from_cache = false;

        if (self.cache.get(text)) |cached| {
            embedding = cached;
            from_cache = true;
        } else if (self.ai_client) |*client| {
            embedding = try client.generateEmbedding(text);
            try self.cache.put(text, embedding);
        } else {
            return error.NoAIClient;
        }
        defer if (!from_cache) self.allocator.free(embedding);

        const cloned_id = try self.allocator.dupe(u8, id);
        errdefer self.allocator.free(cloned_id);
        const cloned_vec = try self.allocator.dupe(f32, embedding);
        errdefer self.allocator.free(cloned_vec);
        const owned_metadata = try self.cloneMetadata(metadata);
        errdefer self.deinitOwnedMetadata(owned_metadata);

        // Track internally
        const mapped = EngineVector{
            .id = cloned_id,
            .vec = cloned_vec,
            .metadata = owned_metadata,
        };

        try self.vectors_array.append(self.allocator, mapped);

        // Placed in HW index bounds mapping array offsets directly inline
        _ = try self.hnsw_index.insert(cloned_vec);
    }

    /// Index a document with a pre-computed embedding (bypasses AI client).
    pub fn indexByVector(
        self: *Engine,
        id: []const u8,
        vector: []const f32,
        metadata: Metadata,
    ) !void {
        const cloned_id = try self.allocator.dupe(u8, id);
        errdefer self.allocator.free(cloned_id);
        const cloned_vec = try self.allocator.dupe(f32, vector);
        errdefer self.allocator.free(cloned_vec);
        const owned_metadata = try self.cloneMetadata(metadata);
        errdefer self.deinitOwnedMetadata(owned_metadata);

        try self.vectors_array.append(self.allocator, EngineVector{
            .id = cloned_id,
            .vec = cloned_vec,
            .metadata = owned_metadata,
        });
        _ = try self.hnsw_index.insert(cloned_vec);
    }

    /// Delete a vector by ID. Returns true if found and removed.
    pub fn delete(self: *Engine, id: []const u8) bool {
        for (self.vectors_array.items, 0..) |item, i| {
            if (std.mem.eql(u8, item.id, id)) {
                self.allocator.free(item.id);
                self.allocator.free(item.vec);
                self.deinitOwnedMetadata(item.metadata);
                _ = self.vectors_array.swapRemove(i);
                return true;
            }
        }
        return false;
    }

    /// Return the current number of indexed vectors.
    pub fn count(self: *const Engine) usize {
        return self.vectors_array.items.len;
    }

    pub fn search(
        self: *Engine,
        query: []const u8,
        options: SearchOptions,
    ) ![]SearchResult {
        const query_embedding = if (self.ai_client) |*client|
            try client.generateEmbedding(query)
        else
            return error.NoAIClient;

        defer self.allocator.free(query_embedding);
        return try self.searchByVector(query_embedding, options);
    }

    pub fn searchByVector(
        self: *Engine,
        query_vector: []const f32,
        options: SearchOptions,
    ) ![]SearchResult {

        // HW boundary execution.
        // This natively returns absolute index numeric layout mappings matched natively against
        // flat slice offsets stored implicitly.
        const matched_ids = try self.hnsw_index.search(query_vector, options.k * 2, options.ef);
        defer self.allocator.free(matched_ids);

        var final_results: std.ArrayListUnmanaged(SearchResult) = .empty;
        errdefer final_results.deinit(self.allocator);

        for (matched_ids) |idx| {
            if (idx >= self.vectors_array.items.len) continue;

            const item = self.vectors_array.items[idx];

            // Meta Filtering bounds
            if (options.filter_category) |cat| {
                if (item.metadata.category) |icat| {
                    if (!std.mem.eql(u8, cat, icat)) continue;
                } else {
                    continue;
                }
            }

            const sim = switch (self.config.metric) {
                .cosine => metrics.cosineSimilarity(query_vector, item.vec),
                .euclidean => -metrics.euclideanDistance(query_vector, item.vec),
                .manhattan => -metrics.manhattanDistance(query_vector, item.vec),
                .dot_product => metrics.dotProduct(query_vector, item.vec),
            };

            if ((self.config.metric == .cosine or self.config.metric == .dot_product) and sim < options.min_score) {
                continue;
            }

            const dist = switch (self.config.metric) {
                .euclidean => metrics.euclideanDistance(query_vector, item.vec),
                .manhattan => metrics.manhattanDistance(query_vector, item.vec),
                else => 0,
            };

            try final_results.append(self.allocator, .{
                .id = item.id,
                .similarity = sim,
                .distance = dist,
                .metadata = item.metadata,
                .vector = item.vec,
            });

            if (final_results.items.len >= options.k) break;
        }

        return final_results.toOwnedSlice(self.allocator);
    }

    fn cloneMetadata(self: *Engine, metadata: Metadata) !Metadata {
        const cloned_text = try self.allocator.dupe(u8, metadata.text);
        errdefer self.allocator.free(cloned_text);

        const cloned_category = if (metadata.category) |category|
            try self.allocator.dupe(u8, category)
        else
            null;
        errdefer if (cloned_category) |category| self.allocator.free(category);

        var cloned_tags: []const []const u8 = &.{};
        if (metadata.tags.len > 0) {
            const tags = try self.allocator.alloc([]const u8, metadata.tags.len);
            var initialized: usize = 0;
            errdefer {
                for (tags[0..initialized]) |tag| {
                    self.allocator.free(tag);
                }
                self.allocator.free(tags);
            }

            for (metadata.tags, 0..) |tag, i| {
                tags[i] = try self.allocator.dupe(u8, tag);
                initialized += 1;
            }
            cloned_tags = tags;
        }
        errdefer if (cloned_tags.len > 0) {
            for (cloned_tags) |tag| {
                self.allocator.free(tag);
            }
            self.allocator.free(cloned_tags);
        };

        const cloned_extra = if (metadata.extra) |extra|
            try self.allocator.dupe(u8, extra)
        else
            null;
        errdefer if (cloned_extra) |extra| self.allocator.free(extra);

        return .{
            .text = cloned_text,
            .category = cloned_category,
            .tags = cloned_tags,
            .score = metadata.score,
            .extra = cloned_extra,
        };
    }

    fn deinitOwnedMetadata(self: *Engine, metadata: Metadata) void {
        self.allocator.free(metadata.text);
        if (metadata.category) |category| {
            self.allocator.free(category);
        }
        if (metadata.tags.len > 0) {
            for (metadata.tags) |tag| {
                self.allocator.free(tag);
            }
            self.allocator.free(metadata.tags);
        }
        if (metadata.extra) |extra| {
            self.allocator.free(extra);
        }
    }
};

test "Engine API routing basic" {
    var engine = try Engine.init(std.testing.allocator, .{});
    defer engine.deinit();

    // Cannot mock HTTP inference dynamically in simple unit test
    const dummy_vec = [_]f32{ 0.5, 0.4, 0.3 };
    _ = try engine.searchByVector(&dummy_vec, .{ .k = 5 });
}

test "Engine deep-copies metadata ownership and survives source mutation" {
    var engine = try Engine.init(std.testing.allocator, .{});
    defer engine.deinit();

    const embedding = [_]f32{ 1.0, 0.0, 0.0 };
    try engine.cache.put("doc-cache-key", &embedding);

    var text_buf = [_]u8{ 'h', 'e', 'l', 'l', 'o' };
    var category_buf = [_]u8{ 'n', 'e', 'w', 's' };
    var tag_buf = [_]u8{ 'a', 'l', 'p', 'h', 'a' };
    var extra_buf = [_]u8{ 'x', '1' };
    const tags = [_][]const u8{tag_buf[0..]};

    try engine.index("doc-1", "doc-cache-key", .{
        .text = text_buf[0..],
        .category = category_buf[0..],
        .tags = tags[0..],
        .score = 0.7,
        .extra = extra_buf[0..],
    });

    text_buf[0] = 'X';
    category_buf[0] = 'Y';
    tag_buf[0] = 'Z';
    extra_buf[0] = 'Q';

    const results = try engine.searchByVector(&embedding, .{ .k = 1, .ef = 16 });
    defer std.testing.allocator.free(results);

    try std.testing.expectEqual(@as(usize, 1), results.len);
    try std.testing.expectEqualStrings("hello", results[0].metadata.text);
    try std.testing.expectEqualStrings("news", results[0].metadata.category.?);
    try std.testing.expectEqualStrings("alpha", results[0].metadata.tags[0]);
    try std.testing.expectEqualStrings("x1", results[0].metadata.extra.?);
}

test "Engine manhattan metric path sets similarity and distance consistently" {
    var engine = try Engine.init(std.testing.allocator, .{ .metric = .manhattan });
    defer engine.deinit();

    const embedding = [_]f32{ 1.0, 1.0, 1.0 };
    try engine.cache.put("doc-cache-key", &embedding);
    try engine.index("doc-1", "doc-cache-key", .{
        .text = "doc",
        .tags = &.{},
    });

    const query = [_]f32{ 1.0, 2.0, 1.0 };
    const results = try engine.searchByVector(&query, .{ .k = 1, .ef = 16 });
    defer std.testing.allocator.free(results);

    try std.testing.expectEqual(@as(usize, 1), results.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), results[0].distance, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), results[0].similarity, 0.0001);
}
