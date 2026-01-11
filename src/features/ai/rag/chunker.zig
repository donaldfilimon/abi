//! Document chunking for RAG pipeline.
//!
//! Provides multiple strategies for splitting documents into
//! chunks suitable for embedding and retrieval.

const std = @import("std");

/// Chunking strategy.
pub const ChunkingStrategy = enum {
    /// Fixed-size chunks with optional overlap.
    fixed,
    /// Split on sentence boundaries.
    sentence,
    /// Split on paragraph boundaries.
    paragraph,
    /// Recursive splitting (try larger units first).
    recursive,
    /// Semantic chunking (content-aware).
    semantic,
};

/// Chunker configuration.
pub const ChunkerConfig = struct {
    /// Chunking strategy.
    strategy: ChunkingStrategy = .recursive,
    /// Target chunk size in characters.
    chunk_size: usize = 500,
    /// Overlap between chunks (for fixed strategy).
    chunk_overlap: usize = 50,
    /// Minimum chunk size.
    min_chunk_size: usize = 50,
    /// Maximum chunk size.
    max_chunk_size: usize = 2000,
    /// Separators for recursive chunking.
    separators: []const []const u8 = &[_][]const u8{
        "\n\n", // Paragraph
        "\n", // Line
        ". ", // Sentence
        ", ", // Clause
        " ", // Word
    },
};

/// A chunk of document content.
pub const Chunk = struct {
    /// Chunk content.
    content: []const u8,
    /// Start offset in original document.
    start_offset: usize,
    /// End offset in original document.
    end_offset: usize,
    /// Chunk index within document.
    index: usize,
    /// Metadata (e.g., section header).
    metadata: ?[]const u8,

    /// Estimate token count.
    pub fn estimateTokens(self: *const Chunk) usize {
        return (self.content.len + 3) / 4;
    }

    /// Clone the chunk.
    pub fn clone(self: Chunk, allocator: std.mem.Allocator) !Chunk {
        return .{
            .content = try allocator.dupe(u8, self.content),
            .start_offset = self.start_offset,
            .end_offset = self.end_offset,
            .index = self.index,
            .metadata = if (self.metadata) |m| try allocator.dupe(u8, m) else null,
        };
    }

    /// Free chunk resources.
    pub fn deinit(self: *Chunk, allocator: std.mem.Allocator) void {
        allocator.free(self.content);
        if (self.metadata) |m| allocator.free(m);
        self.* = undefined;
    }
};

/// Document chunker.
pub const Chunker = struct {
    allocator: std.mem.Allocator,
    config: ChunkerConfig,

    /// Initialize chunker.
    pub fn init(allocator: std.mem.Allocator, config: ChunkerConfig) Chunker {
        return .{
            .allocator = allocator,
            .config = config,
        };
    }

    /// Deinitialize chunker.
    pub fn deinit(self: *Chunker) void {
        self.* = undefined;
    }

    /// Chunk a document.
    pub fn chunk(self: *Chunker, content: []const u8) ![]Chunk {
        return switch (self.config.strategy) {
            .fixed => self.chunkFixed(content),
            .sentence => self.chunkSentence(content),
            .paragraph => self.chunkParagraph(content),
            .recursive => self.chunkRecursive(content, 0),
            .semantic => self.chunkSemantic(content),
        };
    }

    /// Fixed-size chunking with overlap.
    fn chunkFixed(self: *Chunker, content: []const u8) ![]Chunk {
        if (content.len == 0) return &[_]Chunk{};

        var chunks = std.ArrayListUnmanaged(Chunk){};
        errdefer {
            for (chunks.items) |*c| c.deinit(self.allocator);
            chunks.deinit(self.allocator);
        }

        var pos: usize = 0;
        var idx: usize = 0;

        while (pos < content.len) {
            const end = @min(pos + self.config.chunk_size, content.len);
            const chunk_content = try self.allocator.dupe(u8, content[pos..end]);

            try chunks.append(self.allocator, .{
                .content = chunk_content,
                .start_offset = pos,
                .end_offset = end,
                .index = idx,
                .metadata = null,
            });

            if (end >= content.len) break;

            // Move position with overlap
            pos = if (self.config.chunk_overlap < self.config.chunk_size)
                pos + self.config.chunk_size - self.config.chunk_overlap
            else
                end;

            idx += 1;
        }

        return chunks.toOwnedSlice(self.allocator);
    }

    /// Sentence-based chunking.
    fn chunkSentence(self: *Chunker, content: []const u8) ![]Chunk {
        var chunks = std.ArrayListUnmanaged(Chunk){};
        errdefer {
            for (chunks.items) |*c| c.deinit(self.allocator);
            chunks.deinit(self.allocator);
        }

        var current_chunk = std.ArrayListUnmanaged(u8){};
        defer current_chunk.deinit(self.allocator);

        var chunk_start: usize = 0;
        var pos: usize = 0;
        var idx: usize = 0;

        while (pos < content.len) {
            const c = content[pos];
            try current_chunk.append(self.allocator, c);

            // Check for sentence end
            const is_sentence_end = (c == '.' or c == '!' or c == '?') and
                (pos + 1 >= content.len or content[pos + 1] == ' ' or content[pos + 1] == '\n');

            if (is_sentence_end or pos == content.len - 1) {
                // Check if chunk is large enough
                if (current_chunk.items.len >= self.config.min_chunk_size or
                    pos == content.len - 1)
                {
                    // Check if chunk exceeds max size
                    if (current_chunk.items.len >= self.config.chunk_size) {
                        const chunk_content = try current_chunk.toOwnedSlice(self.allocator);

                        try chunks.append(self.allocator, .{
                            .content = chunk_content,
                            .start_offset = chunk_start,
                            .end_offset = pos + 1,
                            .index = idx,
                            .metadata = null,
                        });

                        chunk_start = pos + 1;
                        idx += 1;
                    }
                }
            }

            pos += 1;
        }

        // Handle remaining content
        if (current_chunk.items.len > 0) {
            const chunk_content = try current_chunk.toOwnedSlice(self.allocator);

            try chunks.append(self.allocator, .{
                .content = chunk_content,
                .start_offset = chunk_start,
                .end_offset = content.len,
                .index = idx,
                .metadata = null,
            });
        }

        return chunks.toOwnedSlice(self.allocator);
    }

    /// Paragraph-based chunking.
    fn chunkParagraph(self: *Chunker, content: []const u8) ![]Chunk {
        var chunks = std.ArrayListUnmanaged(Chunk){};
        errdefer {
            for (chunks.items) |*c| c.deinit(self.allocator);
            chunks.deinit(self.allocator);
        }

        var pos: usize = 0;
        var idx: usize = 0;

        while (pos < content.len) {
            // Skip leading whitespace
            while (pos < content.len and std.ascii.isWhitespace(content[pos])) {
                pos += 1;
            }
            if (pos >= content.len) break;

            const para_start = pos;

            // Find paragraph end (double newline or end of content)
            var para_end = pos;
            while (para_end < content.len) {
                if (para_end + 1 < content.len and
                    content[para_end] == '\n' and content[para_end + 1] == '\n')
                {
                    break;
                }
                para_end += 1;
            }

            // Extract paragraph
            const para_content = content[para_start..para_end];
            if (para_content.len >= self.config.min_chunk_size) {
                const chunk_content = try self.allocator.dupe(u8, para_content);

                try chunks.append(self.allocator, .{
                    .content = chunk_content,
                    .start_offset = para_start,
                    .end_offset = para_end,
                    .index = idx,
                    .metadata = null,
                });

                idx += 1;
            }

            pos = para_end + 1;
        }

        // If no paragraphs found, fall back to fixed chunking
        if (chunks.items.len == 0 and content.len > 0) {
            chunks.deinit(self.allocator);
            return self.chunkFixed(content);
        }

        return chunks.toOwnedSlice(self.allocator);
    }

    /// Recursive chunking (tries larger separators first).
    fn chunkRecursive(self: *Chunker, content: []const u8, sep_idx: usize) ![]Chunk {
        if (content.len <= self.config.chunk_size) {
            // Content fits in one chunk
            var chunks = try self.allocator.alloc(Chunk, 1);
            chunks[0] = .{
                .content = try self.allocator.dupe(u8, content),
                .start_offset = 0,
                .end_offset = content.len,
                .index = 0,
                .metadata = null,
            };
            return chunks;
        }

        if (sep_idx >= self.config.separators.len) {
            // No more separators, use fixed chunking
            return self.chunkFixed(content);
        }

        const separator = self.config.separators[sep_idx];

        // Split by separator
        var parts = std.ArrayListUnmanaged([]const u8){};
        defer parts.deinit(self.allocator);

        var start: usize = 0;
        var pos: usize = 0;

        while (pos + separator.len <= content.len) {
            if (std.mem.eql(u8, content[pos .. pos + separator.len], separator)) {
                if (pos > start) {
                    try parts.append(self.allocator, content[start..pos]);
                }
                start = pos + separator.len;
                pos = start;
            } else {
                pos += 1;
            }
        }

        if (start < content.len) {
            try parts.append(self.allocator, content[start..]);
        }

        if (parts.items.len <= 1) {
            // Separator not found, try next
            return self.chunkRecursive(content, sep_idx + 1);
        }

        // Merge small parts and chunk large ones
        var chunks = std.ArrayListUnmanaged(Chunk){};
        errdefer {
            for (chunks.items) |*c| c.deinit(self.allocator);
            chunks.deinit(self.allocator);
        }

        var current = std.ArrayListUnmanaged(u8){};
        defer current.deinit(self.allocator);

        var offset: usize = 0;
        var idx: usize = 0;

        for (parts.items) |part| {
            if (current.items.len + part.len > self.config.chunk_size and
                current.items.len >= self.config.min_chunk_size)
            {
                // Save current chunk
                const chunk_content = try current.toOwnedSlice(self.allocator);

                try chunks.append(self.allocator, .{
                    .content = chunk_content,
                    .start_offset = offset,
                    .end_offset = offset + chunk_content.len,
                    .index = idx,
                    .metadata = null,
                });

                offset += chunk_content.len;
                idx += 1;
            }

            try current.appendSlice(self.allocator, part);
            try current.appendSlice(self.allocator, separator);
        }

        // Handle remaining
        if (current.items.len > 0) {
            const chunk_content = try current.toOwnedSlice(self.allocator);

            try chunks.append(self.allocator, .{
                .content = chunk_content,
                .start_offset = offset,
                .end_offset = offset + chunk_content.len,
                .index = idx,
                .metadata = null,
            });
        }

        return chunks.toOwnedSlice(self.allocator);
    }

    /// Semantic chunking (placeholder for more advanced implementation).
    fn chunkSemantic(self: *Chunker, content: []const u8) ![]Chunk {
        // For now, falls back to recursive chunking
        // A full implementation would use embeddings to find semantic boundaries
        return self.chunkRecursive(content, 0);
    }
};

test "fixed chunking" {
    const allocator = std.testing.allocator;
    var chunker_inst = Chunker.init(allocator, .{
        .strategy = .fixed,
        .chunk_size = 10,
        .chunk_overlap = 2,
    });
    defer chunker_inst.deinit();

    const chunks = try chunker_inst.chunk("Hello world, this is a test.");
    defer {
        for (chunks) |*c| {
            var chunk = c.*;
            chunk.deinit(allocator);
        }
        allocator.free(chunks);
    }

    try std.testing.expect(chunks.len > 1);
}

test "sentence chunking" {
    const allocator = std.testing.allocator;
    var chunker_inst = Chunker.init(allocator, .{
        .strategy = .sentence,
        .min_chunk_size = 5,
        .chunk_size = 50,
    });
    defer chunker_inst.deinit();

    const chunks = try chunker_inst.chunk("First sentence. Second sentence. Third sentence.");
    defer {
        for (chunks) |*c| {
            var chunk = c.*;
            chunk.deinit(allocator);
        }
        allocator.free(chunks);
    }

    try std.testing.expect(chunks.len >= 1);
}

test "paragraph chunking" {
    const allocator = std.testing.allocator;
    var chunker_inst = Chunker.init(allocator, .{
        .strategy = .paragraph,
        .min_chunk_size = 5,
    });
    defer chunker_inst.deinit();

    const text = "First paragraph content.\n\nSecond paragraph content.\n\nThird paragraph.";
    const chunks = try chunker_inst.chunk(text);
    defer {
        for (chunks) |*c| {
            var chunk = c.*;
            chunk.deinit(allocator);
        }
        allocator.free(chunks);
    }

    try std.testing.expect(chunks.len >= 1);
}

test "recursive chunking" {
    const allocator = std.testing.allocator;
    var chunker_inst = Chunker.init(allocator, .{
        .strategy = .recursive,
        .chunk_size = 50,
        .min_chunk_size = 10,
    });
    defer chunker_inst.deinit();

    const chunks = try chunker_inst.chunk("Short text that fits in one chunk.");
    defer {
        for (chunks) |*c| {
            var chunk = c.*;
            chunk.deinit(allocator);
        }
        allocator.free(chunks);
    }

    try std.testing.expect(chunks.len >= 1);
}
