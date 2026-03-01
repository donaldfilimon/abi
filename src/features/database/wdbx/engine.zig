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

// Simplified internal tracking layout
const EngineVector = struct {
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
    vectors_map: std.StringHashMapUnmanaged(EngineVector) = .{},
    vectors_array: std.ArrayListUnmanaged(EngineVector) = .empty,

    pub fn init(allocator: std.mem.Allocator, cfg: config.Config) !Engine {
        cfg.validate();

        const cache = try Cache.init(allocator, cfg.cache.capacity, cfg.cache.segments);
        errdefer cache.deinit();

        const index = try HNSW.init(allocator, cfg, cfg.metric);
        errdefer index.deinit();

        return .{
            .allocator = allocator,
            .config = cfg,
            .cache = cache,
            .hnsw_index = index,
        };
    }

    pub fn deinit(self: *Engine) void {
        self.cache.deinit();
        self.hnsw_index.deinit();
        if (self.ai_client) |*c| c.deinit();

        for (self.vectors_array.items) |item| {
            self.allocator.free(item.id);
            self.allocator.free(item.vec);
            // Deep free for Meta omitted for brevity
        }
        self.vectors_array.deinit(self.allocator);
        self.vectors_map.deinit(self.allocator);
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
        var embedding: []f32 = undefined;
        var from_cache = false;

        if (self.cache.get(text)) |cached| {
            embedding = try self.allocator.dupe(f32, cached);
            from_cache = true;
        } else if (self.ai_client) |*client| {
            embedding = try client.generateEmbedding(text);
            try self.cache.put(text, embedding);
        } else {
            return error.NoAIClient;
        }
        defer if (!from_cache) self.allocator.free(embedding);

        const cloned_id = try self.allocator.dupe(u8, id);
        const cloned_vec = try self.allocator.dupe(f32, embedding);

        // Track internally
        const mapped = EngineVector{
            .id = cloned_id,
            .vec = cloned_vec,
            .metadata = metadata,
        };

        try self.vectors_array.append(self.allocator, mapped);
        try self.vectors_map.put(self.allocator, cloned_id, mapped);

        // Placed in HW index bounds mapping array offsets directly inline
        _ = try self.hnsw_index.insert(cloned_vec);
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

        var final_results = std.ArrayList(SearchResult).init(self.allocator);
        errdefer final_results.deinit();

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

            const sim = if (self.config.metric == .cosine)
                metrics.cosineSimilarity(query_vector, item.vec)
            else if (self.config.metric == .euclidean)
                -metrics.euclideanDistance(query_vector, item.vec) // negative standard metric fallback routing
            else
                metrics.dotProduct(query_vector, item.vec);

            if (sim < options.min_score) continue;

            const dist = if (self.config.metric == .euclidean)
                metrics.euclideanDistance(query_vector, item.vec)
            else
                0;

            try final_results.append(.{
                .id = item.id,
                .similarity = sim,
                .distance = dist,
                .metadata = item.metadata,
                .vector = item.vec,
            });

            if (final_results.items.len >= options.k) break;
        }

        return final_results.toOwnedSlice();
    }
};

test "Engine API routing basic" {
    var engine = try Engine.init(std.testing.allocator, .{});
    defer engine.deinit();

    // Cannot mock HTTP inference dynamically in simple unit test
    const dummy_vec = [_]f32{ 0.5, 0.4, 0.3 };
    _ = try engine.searchByVector(&dummy_vec, .{ .k = 5 });
}
