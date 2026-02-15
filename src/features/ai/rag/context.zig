//! Context building for RAG pipeline.
//!
//! Provides utilities for building prompts with retrieved context
//! for language model input.

const std = @import("std");
const retriever = @import("retriever.zig");
const RetrievalResult = retriever.RetrievalResult;

/// Context building configuration.
pub const ContextConfig = struct {
    /// Template for context injection.
    context_template: []const u8 = default_context_template,
    /// Template for individual chunks.
    chunk_template: []const u8 = default_chunk_template,
    /// Separator between chunks.
    chunk_separator: []const u8 = "\n\n",
    /// Include source references.
    include_sources: bool = true,
    /// Include relevance scores.
    include_scores: bool = false,
    /// Maximum context length in characters.
    max_context_length: usize = 10000,
    /// Truncation strategy.
    truncation: TruncationStrategy = .drop_last,
};

const default_context_template =
    \\Use the following context to answer the question:
    \\
    \\{context}
    \\
    \\Question: {query}
    \\Answer:
;

const default_chunk_template = "[{index}] {content}";

/// Truncation strategy when context exceeds limits.
pub const TruncationStrategy = enum {
    /// Drop the last (least relevant) chunks.
    drop_last,
    /// Truncate each chunk proportionally.
    truncate_all,
    /// Truncate only the last chunk.
    truncate_last,
};

/// Built RAG context.
pub const RagContext = struct {
    /// The complete prompt with context.
    prompt: []const u8,
    /// Just the context portion.
    context_only: []const u8,
    /// Number of chunks included.
    chunks_used: usize,
    /// Total tokens (estimated).
    total_tokens: usize,
    /// Whether context was truncated.
    truncated: bool,
    /// Source references.
    sources: []const SourceRef,

    pub const SourceRef = struct {
        doc_id: []const u8,
        chunk_index: usize,
        score: f32,
    };

    pub fn deinit(self: *RagContext, allocator: std.mem.Allocator) void {
        allocator.free(self.prompt);
        allocator.free(self.context_only);
        allocator.free(self.sources);
        self.* = undefined;
    }

    /// Get estimated token count.
    pub fn estimateTokens(self: *const RagContext) usize {
        return self.total_tokens;
    }
};

/// Context builder for RAG.
pub const ContextBuilder = struct {
    allocator: std.mem.Allocator,
    config: ContextConfig,

    /// Initialize context builder.
    pub fn init(allocator: std.mem.Allocator, config: ContextConfig) ContextBuilder {
        return .{
            .allocator = allocator,
            .config = config,
        };
    }

    /// Deinitialize context builder.
    pub fn deinit(self: *ContextBuilder) void {
        self.* = undefined;
    }

    /// Build context from retrieval results.
    pub fn build(
        self: *ContextBuilder,
        query: []const u8,
        results: []const RetrievalResult,
        max_tokens: usize,
    ) !RagContext {
        var context_parts = std.ArrayListUnmanaged(u8).empty;
        defer context_parts.deinit(self.allocator);

        var sources = std.ArrayListUnmanaged(RagContext.SourceRef).empty;
        errdefer sources.deinit(self.allocator);

        var chunks_used: usize = 0;
        var truncated = false;

        // Build context from chunks
        for (results, 0..) |result, i| {
            // Format chunk
            const chunk_text = try self.formatChunk(result, i);
            defer self.allocator.free(chunk_text);

            // Check token limit
            const current_tokens = (context_parts.items.len + 3) / 4;
            const chunk_tokens = (chunk_text.len + 3) / 4;

            if (current_tokens + chunk_tokens > max_tokens) {
                if (self.config.truncation == .drop_last) {
                    truncated = true;
                    break;
                } else if (self.config.truncation == .truncate_last) {
                    const remaining = max_tokens - current_tokens;
                    const max_chars = remaining * 4;
                    if (max_chars > 0) {
                        try context_parts.appendSlice(
                            self.allocator,
                            chunk_text[0..@min(max_chars, chunk_text.len)],
                        );
                    }
                    truncated = true;
                    break;
                }
            }

            if (chunks_used > 0) {
                try context_parts.appendSlice(self.allocator, self.config.chunk_separator);
            }

            try context_parts.appendSlice(self.allocator, chunk_text);
            chunks_used += 1;

            // Add source reference
            try sources.append(self.allocator, .{
                .doc_id = result.doc_id,
                .chunk_index = result.chunk.index,
                .score = result.score,
            });
        }

        // Build final prompt
        const context_only = try self.allocator.dupe(u8, context_parts.items);
        errdefer self.allocator.free(context_only);

        const prompt = try self.formatPrompt(query, context_only);
        errdefer self.allocator.free(prompt);

        const total_tokens = (prompt.len + 3) / 4;

        return .{
            .prompt = prompt,
            .context_only = context_only,
            .chunks_used = chunks_used,
            .total_tokens = total_tokens,
            .truncated = truncated,
            .sources = try sources.toOwnedSlice(self.allocator),
        };
    }

    /// Format a single chunk using the template.
    fn formatChunk(self: *ContextBuilder, result: RetrievalResult, index: usize) ![]u8 {
        var output = std.ArrayListUnmanaged(u8).empty;
        errdefer output.deinit(self.allocator);

        var i: usize = 0;
        while (i < self.config.chunk_template.len) {
            if (i + 1 < self.config.chunk_template.len and
                self.config.chunk_template[i] == '{')
            {
                // Find closing brace
                var end = i + 1;
                while (end < self.config.chunk_template.len and
                    self.config.chunk_template[end] != '}')
                {
                    end += 1;
                }

                if (end < self.config.chunk_template.len) {
                    const placeholder = self.config.chunk_template[i + 1 .. end];

                    if (std.mem.eql(u8, placeholder, "index")) {
                        var buf: [20]u8 = undefined;
                        const idx_str = std.fmt.bufPrint(&buf, "{d}", .{index + 1}) catch "?";
                        try output.appendSlice(self.allocator, idx_str);
                    } else if (std.mem.eql(u8, placeholder, "content")) {
                        try output.appendSlice(self.allocator, result.chunk.content);
                    } else if (std.mem.eql(u8, placeholder, "score")) {
                        var buf: [20]u8 = undefined;
                        const score_str = std.fmt.bufPrint(&buf, "{d:.2}", .{result.score}) catch "?";
                        try output.appendSlice(self.allocator, score_str);
                    } else if (std.mem.eql(u8, placeholder, "doc_id")) {
                        try output.appendSlice(self.allocator, result.doc_id);
                    }

                    i = end + 1;
                    continue;
                }
            }

            try output.append(self.allocator, self.config.chunk_template[i]);
            i += 1;
        }

        return output.toOwnedSlice(self.allocator);
    }

    /// Format the final prompt with context and query.
    fn formatPrompt(self: *ContextBuilder, query: []const u8, context_text: []const u8) ![]u8 {
        var output = std.ArrayListUnmanaged(u8).empty;
        errdefer output.deinit(self.allocator);

        var i: usize = 0;
        while (i < self.config.context_template.len) {
            if (i + 1 < self.config.context_template.len and
                self.config.context_template[i] == '{')
            {
                var end = i + 1;
                while (end < self.config.context_template.len and
                    self.config.context_template[end] != '}')
                {
                    end += 1;
                }

                if (end < self.config.context_template.len) {
                    const placeholder = self.config.context_template[i + 1 .. end];

                    if (std.mem.eql(u8, placeholder, "context")) {
                        try output.appendSlice(self.allocator, context_text);
                    } else if (std.mem.eql(u8, placeholder, "query")) {
                        try output.appendSlice(self.allocator, query);
                    }

                    i = end + 1;
                    continue;
                }
            }

            try output.append(self.allocator, self.config.context_template[i]);
            i += 1;
        }

        return output.toOwnedSlice(self.allocator);
    }

    /// Build a simple context string without template.
    pub fn buildSimple(
        self: *ContextBuilder,
        results: []const RetrievalResult,
    ) ![]u8 {
        var output = std.ArrayListUnmanaged(u8).empty;
        errdefer output.deinit(self.allocator);

        for (results, 0..) |result, i| {
            if (i > 0) {
                try output.appendSlice(self.allocator, self.config.chunk_separator);
            }
            try output.appendSlice(self.allocator, result.chunk.content);
        }

        return output.toOwnedSlice(self.allocator);
    }
};

/// Create a basic RAG prompt.
pub fn createRagPrompt(
    allocator: std.mem.Allocator,
    query: []const u8,
    context_text: []const u8,
) ![]u8 {
    var output = std.ArrayListUnmanaged(u8).empty;
    errdefer output.deinit(allocator);

    try output.appendSlice(allocator, "Context:\n");
    try output.appendSlice(allocator, context_text);
    try output.appendSlice(allocator, "\n\nQuestion: ");
    try output.appendSlice(allocator, query);
    try output.appendSlice(allocator, "\nAnswer:");

    return output.toOwnedSlice(allocator);
}

test "context builder basic" {
    const allocator = std.testing.allocator;
    var builder = ContextBuilder.init(allocator, .{});
    defer builder.deinit();

    const chunk = @import("chunker.zig").Chunk{
        .content = "This is test content.",
        .start_offset = 0,
        .end_offset = 21,
        .index = 0,
        .metadata = null,
    };

    const results = [_]RetrievalResult{
        .{ .chunk = chunk, .doc_id = "doc1", .score = 0.9, .rank = 1 },
    };

    var ctx = try builder.build("What is the content?", &results, 1000);
    defer ctx.deinit(allocator);

    try std.testing.expect(ctx.prompt.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, ctx.prompt, "test content") != null);
    try std.testing.expectEqual(@as(usize, 1), ctx.chunks_used);
}

test "context builder truncation" {
    const allocator = std.testing.allocator;
    var builder = ContextBuilder.init(allocator, .{
        .truncation = .drop_last,
    });
    defer builder.deinit();

    const chunk = @import("chunker.zig").Chunk{
        .content = "A" ** 100,
        .start_offset = 0,
        .end_offset = 100,
        .index = 0,
        .metadata = null,
    };

    const results = [_]RetrievalResult{
        .{ .chunk = chunk, .doc_id = "doc1", .score = 0.9, .rank = 1 },
        .{ .chunk = chunk, .doc_id = "doc2", .score = 0.8, .rank = 2 },
    };

    // Very small token limit
    var ctx = try builder.build("query", &results, 10);
    defer ctx.deinit(allocator);

    try std.testing.expect(ctx.truncated);
}

test "create rag prompt" {
    const allocator = std.testing.allocator;

    const prompt = try createRagPrompt(
        allocator,
        "What is AI?",
        "AI stands for Artificial Intelligence.",
    );
    defer allocator.free(prompt);

    try std.testing.expect(std.mem.indexOf(u8, prompt, "What is AI?") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "Artificial Intelligence") != null);
}
