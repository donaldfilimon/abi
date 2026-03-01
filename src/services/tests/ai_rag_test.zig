//! AI RAG Tests — Chunker, Retriever, Context Builder
//!
//! Tests document chunking strategies, similarity metrics, and context assembly.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const rag = if (build_options.enable_ai) abi.features.ai.rag else struct {};
const Chunker = if (build_options.enable_ai) rag.Chunker else struct {};
const Retriever = if (build_options.enable_ai) rag.Retriever else struct {};
const ContextBuilder = if (build_options.enable_ai) rag.ContextBuilder else struct {};

// ============================================================================
// Chunker Strategy Tests
// ============================================================================

test "chunker: fixed strategy splits at chunk_size" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var chunker = Chunker.init(allocator, .{
        .strategy = .fixed,
        .chunk_size = 20,
        .chunk_overlap = 0,
        .min_chunk_size = 1,
    });
    defer chunker.deinit();

    const text = "Hello world, this is a test of the chunking system.";
    const chunks = try chunker.chunk(text);
    defer {
        for (chunks) |*c| c.deinit(allocator);
        allocator.free(chunks);
    }

    // Should produce multiple chunks
    try std.testing.expect(chunks.len >= 2);
    // First chunk should be ~20 chars
    try std.testing.expect(chunks[0].content.len <= 25);
}

test "chunker: sentence strategy splits on boundaries" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var chunker = Chunker.init(allocator, .{
        .strategy = .sentence,
        .chunk_size = 500,
        .min_chunk_size = 1,
    });
    defer chunker.deinit();

    const text = "First sentence. Second sentence. Third sentence.";
    const chunks = try chunker.chunk(text);
    defer {
        for (chunks) |*c| c.deinit(allocator);
        allocator.free(chunks);
    }

    // Should produce at least 1 chunk (sentences may be grouped)
    try std.testing.expect(chunks.len >= 1);
}

test "chunker: paragraph strategy splits on double newlines" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var chunker = Chunker.init(allocator, .{
        .strategy = .paragraph,
        .chunk_size = 500,
        .min_chunk_size = 1,
    });
    defer chunker.deinit();

    const text = "Paragraph one content here.\n\nParagraph two is different.\n\nParagraph three ends the document.";
    const chunks = try chunker.chunk(text);
    defer {
        for (chunks) |*c| c.deinit(allocator);
        allocator.free(chunks);
    }

    // Should split on \n\n boundaries
    try std.testing.expect(chunks.len >= 2);
}

test "chunker: empty content returns empty" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var chunker = Chunker.init(allocator, .{
        .strategy = .fixed,
        .chunk_size = 100,
        .min_chunk_size = 1,
    });
    defer chunker.deinit();

    var chunks = try chunker.chunk("");
    defer {
        for (chunks) |*c| c.deinit(allocator);
        allocator.free(chunks);
    }

    try std.testing.expectEqual(@as(usize, 0), chunks.len);
}

test "chunker: chunk offsets map to original content" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var chunker = Chunker.init(allocator, .{
        .strategy = .fixed,
        .chunk_size = 10,
        .chunk_overlap = 0,
        .min_chunk_size = 1,
    });
    defer chunker.deinit();

    const text = "0123456789ABCDEFGHIJ";
    const chunks = try chunker.chunk(text);
    defer {
        for (chunks) |*c| c.deinit(allocator);
        allocator.free(chunks);
    }

    if (chunks.len >= 1) {
        // First chunk starts at 0
        try std.testing.expectEqual(@as(usize, 0), chunks[0].start_offset);
        try std.testing.expect(chunks[0].end_offset > 0);
    }
}

test "chunker: chunk indices are sequential" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var chunker = Chunker.init(allocator, .{
        .strategy = .fixed,
        .chunk_size = 10,
        .chunk_overlap = 0,
        .min_chunk_size = 1,
    });
    defer chunker.deinit();

    const text = "The quick brown fox jumps over the lazy dog and runs away fast.";
    const chunks = try chunker.chunk(text);
    defer {
        for (chunks) |*c| c.deinit(allocator);
        allocator.free(chunks);
    }

    for (chunks, 0..) |c, i| {
        try std.testing.expectEqual(i, c.index);
    }
}

test "chunker: estimateTokens approximation" {
    if (!build_options.enable_ai) return error.SkipZigTest;

    const chunk = rag.Chunk{
        .content = "Hello world test content",
        .start_offset = 0,
        .end_offset = 24,
        .index = 0,
        .metadata = null,
    };

    // 24 chars → ~6 tokens at 4 chars/token
    const tokens = chunk.estimateTokens();
    try std.testing.expect(tokens >= 5);
    try std.testing.expect(tokens <= 8);
}

// ============================================================================
// Retriever Tests
// ============================================================================

test "retriever: default embedding is deterministic" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var ret = Retriever.init(allocator, .{});
    defer ret.deinit();

    const emb1 = try ret.computeEmbedding("hello world");
    defer if (emb1) |e| allocator.free(e);

    const emb2 = try ret.computeEmbedding("hello world");
    defer if (emb2) |e| allocator.free(e);

    if (emb1 != null and emb2 != null) {
        for (emb1.?, emb2.?) |a, b| {
            try std.testing.expectApproxEqAbs(a, b, 1e-6);
        }
    }
}

test "retriever: different texts produce different embeddings" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var ret = Retriever.init(allocator, .{});
    defer ret.deinit();

    const emb1 = try ret.computeEmbedding("hello world");
    defer if (emb1) |e| allocator.free(e);

    const emb2 = try ret.computeEmbedding("completely different text");
    defer if (emb2) |e| allocator.free(e);

    if (emb1 != null and emb2 != null) {
        var all_equal = true;
        for (emb1.?, emb2.?) |a, b| {
            if (@abs(a - b) > 1e-6) {
                all_equal = false;
                break;
            }
        }
        try std.testing.expect(!all_equal);
    }
}

test "retriever: text similarity self-match is high" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var ret = Retriever.init(allocator, .{});
    defer ret.deinit();

    const sim = ret.textSimilarity("the quick brown fox", "the quick brown fox");
    try std.testing.expect(sim > 0.9);
}

test "retriever: text similarity different texts is lower" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var ret = Retriever.init(allocator, .{});
    defer ret.deinit();

    const sim_same = ret.textSimilarity("the quick brown fox", "the quick brown fox");
    const sim_diff = ret.textSimilarity("the quick brown fox", "a lazy sleeping cat");

    try std.testing.expect(sim_same > sim_diff);
}

// ============================================================================
// Context Builder Tests
// ============================================================================

test "context builder: builds prompt with template" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var builder = ContextBuilder.init(allocator, .{});
    defer builder.deinit();

    const results = [_]rag.RetrievalResult{
        .{
            .chunk = .{
                .content = "Paris is the capital of France.",
                .start_offset = 0,
                .end_offset = 30,
                .index = 0,
                .metadata = null,
            },
            .score = 0.9,
            .doc_id = "doc1",
            .rank = 0,
        },
    };

    var ctx = try builder.build(
        "What is the capital of France?",
        &results,
        500,
    );
    defer ctx.deinit(allocator);

    // Prompt should contain both the context and the query
    try std.testing.expect(ctx.prompt.len > 0);
    try std.testing.expect(ctx.chunks_used >= 1);
    try std.testing.expect(ctx.total_tokens > 0);
}

test "context builder: empty results produce valid prompt" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var builder = ContextBuilder.init(allocator, .{});
    defer builder.deinit();

    var ctx = try builder.build(
        "What is the meaning of life?",
        &[_]rag.RetrievalResult{},
        500,
    );
    defer ctx.deinit(allocator);

    try std.testing.expect(ctx.prompt.len > 0);
    try std.testing.expectEqual(@as(usize, 0), ctx.chunks_used);
}
