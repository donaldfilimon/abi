//! Aviva Knowledge Retriever
//!
//! Manages knowledge retrieval for Aviva's factual responses.
//! Integrates with WDBX for vector-based semantic search.
//!
//! Features:
//! - Domain-specific knowledge retrieval
//! - Source attribution for transparency
//! - Confidence scoring for retrieved facts
//! - Caching for frequently accessed knowledge

const std = @import("std");
const classifier = @import("classifier.zig");

/// A retrieved knowledge fragment.
pub const KnowledgeFragment = struct {
    /// The actual content.
    content: []const u8,
    /// Source attribution.
    source: Source,
    /// Relevance score to the query (0.0 - 1.0).
    relevance: f32,
    /// Confidence in the accuracy of this knowledge (0.0 - 1.0).
    confidence: f32,
    /// Domain this knowledge belongs to.
    domain: classifier.Domain,
    /// When this knowledge was last verified.
    last_verified: i64,
};

/// Source attribution for a piece of knowledge.
pub const Source = struct {
    /// Name or title of the source.
    name: []const u8,
    /// Type of source.
    source_type: SourceType,
    /// URL if available.
    url: ?[]const u8 = null,
    /// Specific section/page reference.
    reference: ?[]const u8 = null,
    /// Reliability score (0.0 - 1.0).
    reliability: f32 = 0.8,
};

/// Types of knowledge sources.
pub const SourceType = enum {
    /// Official documentation.
    documentation,
    /// Academic paper or research.
    research,
    /// Code repository or example.
    code_example,
    /// Community knowledge (StackOverflow, forums).
    community,
    /// Internal knowledge base.
    internal,
    /// Training data knowledge.
    training_data,
    /// Real-time search result.
    web_search,
    /// User-provided information.
    user_provided,

    pub fn getReliabilityDefault(self: SourceType) f32 {
        return switch (self) {
            .documentation => 0.95,
            .research => 0.9,
            .code_example => 0.85,
            .community => 0.7,
            .internal => 0.9,
            .training_data => 0.75,
            .web_search => 0.6,
            .user_provided => 0.5,
        };
    }
};

/// Result of a knowledge retrieval operation.
pub const RetrievalResult = struct {
    /// Retrieved knowledge fragments.
    fragments: std.ArrayList(KnowledgeFragment),
    /// Query that was used.
    query: []const u8,
    /// Total fragments found before filtering.
    total_found: usize,
    /// Time taken for retrieval (ms).
    retrieval_time_ms: u64,
    /// Whether cache was used.
    from_cache: bool,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, query: []const u8) Self {
        return .{
            .fragments = std.ArrayList(KnowledgeFragment).init(allocator),
            .query = query,
            .total_found = 0,
            .retrieval_time_ms = 0,
            .from_cache = false,
        };
    }

    pub fn deinit(self: *Self) void {
        self.fragments.deinit();
    }

    /// Get the most relevant fragment.
    pub fn getBest(self: *const Self) ?KnowledgeFragment {
        if (self.fragments.items.len == 0) return null;
        return self.fragments.items[0];
    }

    /// Get fragments above a relevance threshold.
    pub fn getAboveThreshold(self: *const Self, threshold: f32) []const KnowledgeFragment {
        var count: usize = 0;
        for (self.fragments.items) |frag| {
            if (frag.relevance >= threshold) count += 1 else break;
        }
        return self.fragments.items[0..count];
    }
};

/// Configuration for the knowledge retriever.
pub const RetrieverConfig = struct {
    /// Maximum fragments to retrieve.
    max_fragments: usize = 5,
    /// Minimum relevance score.
    min_relevance: f32 = 0.5,
    /// Minimum confidence score.
    min_confidence: f32 = 0.6,
    /// Whether to use caching.
    enable_cache: bool = true,
    /// Cache TTL in seconds.
    cache_ttl_seconds: u64 = 3600,
    /// Whether to include source attribution.
    include_sources: bool = true,
};

/// Cached retrieval entry.
const CacheEntry = struct {
    result: RetrievalResult,
    cached_at: i64,
};

/// Knowledge retriever for Aviva.
pub const KnowledgeRetriever = struct {
    allocator: std.mem.Allocator,
    config: RetrieverConfig,
    /// Cache of recent retrievals.
    cache: std.StringHashMapUnmanaged(CacheEntry),
    /// Knowledge base (in production, would connect to WDBX).
    knowledge_base: KnowledgeBase,

    const Self = @This();

    /// Initialize the retriever.
    pub fn init(allocator: std.mem.Allocator) Self {
        return initWithConfig(allocator, .{});
    }

    /// Initialize with custom configuration.
    pub fn initWithConfig(allocator: std.mem.Allocator, config: RetrieverConfig) Self {
        return .{
            .allocator = allocator,
            .config = config,
            .cache = .{},
            .knowledge_base = KnowledgeBase.init(allocator),
        };
    }

    /// Shutdown and free resources.
    pub fn deinit(self: *Self) void {
        // Free cache entries
        var it = self.cache.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.result.deinit();
        }
        self.cache.deinit(self.allocator);
        self.knowledge_base.deinit();
    }

    /// Retrieve knowledge for a query.
    pub fn retrieve(
        self: *Self,
        query: []const u8,
        domain: ?classifier.Domain,
    ) !RetrievalResult {
        var timer = std.time.Timer.start() catch {
            return error.TimerFailed;
        };

        // Check cache
        if (self.config.enable_cache) {
            if (self.cache.get(query)) |cached| {
                const now = std.time.timestamp();
                if (now - cached.cached_at < @as(i64, @intCast(self.config.cache_ttl_seconds))) {
                    var result = cached.result;
                    result.from_cache = true;
                    return result;
                }
            }
        }

        // Create result
        var result = RetrievalResult.init(self.allocator, query);
        errdefer result.deinit();

        // Retrieve from knowledge base
        const fragments = try self.knowledge_base.search(
            self.allocator,
            query,
            domain,
            self.config.max_fragments * 2, // Over-fetch for filtering
        );
        defer self.allocator.free(fragments);

        result.total_found = fragments.len;

        // Filter and add to result
        for (fragments) |frag| {
            if (frag.relevance >= self.config.min_relevance and
                frag.confidence >= self.config.min_confidence)
            {
                try result.fragments.append(frag);
                if (result.fragments.items.len >= self.config.max_fragments) break;
            }
        }

        // Sort by relevance
        std.mem.sort(KnowledgeFragment, result.fragments.items, {}, struct {
            fn lessThan(_: void, a: KnowledgeFragment, b: KnowledgeFragment) bool {
                return a.relevance > b.relevance; // Descending
            }
        }.lessThan);

        result.retrieval_time_ms = timer.read() / std.time.ns_per_ms;

        // Update cache
        if (self.config.enable_cache) {
            const key = try self.allocator.dupe(u8, query);
            errdefer self.allocator.free(key);
            try self.cache.put(self.allocator, key, .{
                .result = result,
                .cached_at = std.time.timestamp(),
            });
        }

        return result;
    }

    /// Retrieve knowledge for a specific domain.
    pub fn retrieveForDomain(
        self: *Self,
        domain: classifier.Domain,
        limit: usize,
    ) ![]KnowledgeFragment {
        return self.knowledge_base.getByDomain(self.allocator, domain, limit);
    }

    /// Add knowledge to the base.
    pub fn addKnowledge(self: *Self, fragment: KnowledgeFragment) !void {
        try self.knowledge_base.add(fragment);
    }

    /// Clear the cache.
    pub fn clearCache(self: *Self) void {
        var it = self.cache.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.result.deinit();
        }
        self.cache.clearRetainingCapacity();
    }
};

/// In-memory knowledge base (would connect to WDBX in production).
pub const KnowledgeBase = struct {
    allocator: std.mem.Allocator,
    /// Stored knowledge fragments.
    fragments: std.ArrayList(KnowledgeFragment),

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .fragments = std.ArrayList(KnowledgeFragment).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        self.fragments.deinit();
    }

    /// Search for relevant fragments.
    pub fn search(
        self: *Self,
        allocator: std.mem.Allocator,
        query: []const u8,
        domain: ?classifier.Domain,
        limit: usize,
    ) ![]KnowledgeFragment {
        var results = std.ArrayList(KnowledgeFragment).init(allocator);
        errdefer results.deinit();

        // In production, this would use vector similarity search via WDBX
        // For now, we do simple keyword matching
        for (self.fragments.items) |frag| {
            // Filter by domain if specified
            if (domain) |d| {
                if (frag.domain != d and frag.domain != .general) continue;
            }

            // Simple relevance scoring based on keyword overlap
            const relevance = self.calculateRelevance(query, frag.content);
            if (relevance > 0.1) {
                var scored_frag = frag;
                scored_frag.relevance = relevance;
                try results.append(scored_frag);
            }

            if (results.items.len >= limit) break;
        }

        return results.toOwnedSlice();
    }

    /// Get fragments by domain.
    pub fn getByDomain(
        self: *Self,
        allocator: std.mem.Allocator,
        domain: classifier.Domain,
        limit: usize,
    ) ![]KnowledgeFragment {
        var results = std.ArrayList(KnowledgeFragment).init(allocator);
        errdefer results.deinit();

        for (self.fragments.items) |frag| {
            if (frag.domain == domain) {
                try results.append(frag);
                if (results.items.len >= limit) break;
            }
        }

        return results.toOwnedSlice();
    }

    /// Add a fragment to the knowledge base.
    pub fn add(self: *Self, fragment: KnowledgeFragment) !void {
        try self.fragments.append(fragment);
    }

    /// Calculate simple relevance score.
    fn calculateRelevance(self: *const Self, query: []const u8, content: []const u8) f32 {
        _ = self;
        // Simple word overlap scoring
        var matches: usize = 0;
        var total: usize = 0;

        var query_iter = std.mem.splitScalar(u8, query, ' ');
        while (query_iter.next()) |word| {
            if (word.len < 3) continue; // Skip short words
            total += 1;
            if (std.mem.indexOf(u8, content, word) != null) {
                matches += 1;
            }
        }

        if (total == 0) return 0.0;
        return @as(f32, @floatFromInt(matches)) / @as(f32, @floatFromInt(total));
    }
};

/// Format knowledge for inclusion in response.
pub fn formatKnowledgeForResponse(
    allocator: std.mem.Allocator,
    fragments: []const KnowledgeFragment,
    include_sources: bool,
) ![]const u8 {
    var result = std.ArrayList(u8).init(allocator);
    errdefer result.deinit();

    for (fragments, 0..) |frag, i| {
        // Add content
        try result.appendSlice(frag.content);

        // Add source attribution if requested
        if (include_sources) {
            try result.appendSlice(" [");
            try result.appendSlice(frag.source.name);
            try result.append(']');
        }

        // Add separator if not last
        if (i < fragments.len - 1) {
            try result.appendSlice("\n\n");
        }
    }

    return result.toOwnedSlice();
}

// Tests

test "knowledge retriever initialization" {
    var retriever = KnowledgeRetriever.init(std.testing.allocator);
    defer retriever.deinit();

    try std.testing.expectEqual(@as(usize, 5), retriever.config.max_fragments);
}

test "source type reliability" {
    try std.testing.expect(SourceType.documentation.getReliabilityDefault() > SourceType.community.getReliabilityDefault());
}

test "retrieval result initialization" {
    var result = RetrievalResult.init(std.testing.allocator, "test query");
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 0), result.fragments.items.len);
    try std.testing.expect(!result.from_cache);
}

test "get best fragment" {
    var result = RetrievalResult.init(std.testing.allocator, "test");
    defer result.deinit();

    try result.fragments.append(.{
        .content = "Test content",
        .source = .{
            .name = "Test Source",
            .source_type = .documentation,
        },
        .relevance = 0.9,
        .confidence = 0.85,
        .domain = .general,
        .last_verified = 0,
    });

    const best = result.getBest();
    try std.testing.expect(best != null);
    try std.testing.expectEqual(@as(f32, 0.9), best.?.relevance);
}

test "knowledge base search" {
    var kb = KnowledgeBase.init(std.testing.allocator);
    defer kb.deinit();

    try kb.add(.{
        .content = "Zig is a systems programming language",
        .source = .{ .name = "Zig Docs", .source_type = .documentation },
        .relevance = 1.0,
        .confidence = 0.95,
        .domain = .systems_programming,
        .last_verified = 0,
    });

    const results = try kb.search(std.testing.allocator, "Zig programming", null, 10);
    defer std.testing.allocator.free(results);

    try std.testing.expect(results.len > 0);
}

test "format knowledge for response" {
    const fragments = [_]KnowledgeFragment{
        .{
            .content = "First fact",
            .source = .{ .name = "Source A", .source_type = .documentation },
            .relevance = 0.9,
            .confidence = 0.9,
            .domain = .general,
            .last_verified = 0,
        },
        .{
            .content = "Second fact",
            .source = .{ .name = "Source B", .source_type = .research },
            .relevance = 0.8,
            .confidence = 0.85,
            .domain = .general,
            .last_verified = 0,
        },
    };

    const formatted = try formatKnowledgeForResponse(std.testing.allocator, &fragments, true);
    defer std.testing.allocator.free(formatted);

    try std.testing.expect(std.mem.indexOf(u8, formatted, "Source A") != null);
    try std.testing.expect(std.mem.indexOf(u8, formatted, "Source B") != null);
}
