//! Codebase Index — Self-Awareness Module
//!
//! Connects the explore module (AST parsing, call graph, dependency graph)
//! with the RAG pipeline (chunking, embeddings, retrieval) to give the agent
//! awareness of its own codebase.
//!
//! The agent can index a source tree, then query it to understand how code
//! works, find definitions, and trace dependencies.

const std = @import("std");

// ============================================================================
// Types
// ============================================================================

/// Metadata about an indexed file.
pub const FileMetadata = struct {
    path: []const u8,
    size_bytes: usize,
    line_count: usize,
    chunk_count: usize,
    last_indexed: i64,
};

/// Result of a codebase query.
pub const CodebaseAnswer = struct {
    allocator: std.mem.Allocator,
    /// Relevant code snippets found.
    snippets: std.ArrayListUnmanaged(CodeSnippet),
    /// Summary of findings.
    summary: []const u8,

    pub fn deinit(self: *CodebaseAnswer) void {
        for (self.snippets.items) |*s| {
            self.allocator.free(s.file_path);
            self.allocator.free(s.content);
        }
        self.snippets.deinit(self.allocator);
        self.allocator.free(self.summary);
    }
};

/// A relevant code snippet from the index.
pub const CodeSnippet = struct {
    file_path: []const u8,
    start_line: usize,
    end_line: usize,
    content: []const u8,
    relevance_score: f32,
};

/// Result of indexing a codebase.
pub const IndexResult = struct {
    files_indexed: usize,
    total_chunks: usize,
    total_lines: usize,
    duration_ms: u64,
};

/// Statistics about the current index.
pub const IndexStats = struct {
    file_count: usize,
    total_chunks: usize,
    total_size_bytes: usize,
    index_path: []const u8,
};

// ============================================================================
// CodebaseIndex
// ============================================================================

/// Index for codebase self-awareness.
///
/// Walks source files, chunks them, and stores metadata for retrieval.
/// Enables agents to query "how does X work?" about their own codebase.
pub const CodebaseIndex = struct {
    allocator: std.mem.Allocator,
    indexed_files: std.StringHashMapUnmanaged(FileMetadata),
    chunks: std.ArrayListUnmanaged(StoredChunk),
    root_path: []const u8,
    index_dir: []const u8,

    const StoredChunk = struct {
        file_path: []const u8,
        start_line: usize,
        end_line: usize,
        content: []const u8,
    };

    const Self = @This();

    /// Initialize the codebase index.
    pub fn init(allocator: std.mem.Allocator, root_path: []const u8) !Self {
        const root_copy = try allocator.dupe(u8, root_path);
        errdefer allocator.free(root_copy);
        const index_dir = try allocator.dupe(u8, ".abi/codebase_index");

        return .{
            .allocator = allocator,
            .indexed_files = .{},
            .chunks = .{},
            .root_path = root_copy,
            .index_dir = index_dir,
        };
    }

    /// Clean up all resources.
    pub fn deinit(self: *Self) void {
        var file_iter = self.indexed_files.iterator();
        while (file_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.path);
        }
        self.indexed_files.deinit(self.allocator);

        for (self.chunks.items) |chunk| {
            self.allocator.free(chunk.file_path);
            self.allocator.free(chunk.content);
        }
        self.chunks.deinit(self.allocator);

        self.allocator.free(self.root_path);
        self.allocator.free(self.index_dir);
    }

    /// Index a single file by chunking its contents.
    pub fn indexFile(self: *Self, path: []const u8, content: []const u8) !void {
        // Split content into chunks of ~50 lines
        var line_count: usize = 0;
        var chunk_count: usize = 0;
        const chunk_size: usize = 50;

        var lines = std.mem.splitScalar(u8, content, '\n');
        var chunk_start: usize = 0;
        var chunk_lines: usize = 0;
        var chunk_buf = std.ArrayListUnmanaged(u8){};
        defer chunk_buf.deinit(self.allocator);

        while (lines.next()) |line| {
            line_count += 1;
            try chunk_buf.appendSlice(self.allocator, line);
            try chunk_buf.append(self.allocator, '\n');
            chunk_lines += 1;

            if (chunk_lines >= chunk_size) {
                const chunk_content = try self.allocator.dupe(u8, chunk_buf.items);
                errdefer self.allocator.free(chunk_content);
                const file_path_copy = try self.allocator.dupe(u8, path);
                errdefer self.allocator.free(file_path_copy);

                try self.chunks.append(self.allocator, .{
                    .file_path = file_path_copy,
                    .start_line = chunk_start + 1,
                    .end_line = line_count,
                    .content = chunk_content,
                });
                chunk_count += 1;
                chunk_start = line_count;
                chunk_lines = 0;
                chunk_buf.clearRetainingCapacity();
            }
        }

        // Store remaining lines as last chunk
        if (chunk_buf.items.len > 0) {
            const chunk_content = try self.allocator.dupe(u8, chunk_buf.items);
            errdefer self.allocator.free(chunk_content);
            const file_path_copy = try self.allocator.dupe(u8, path);
            errdefer self.allocator.free(file_path_copy);

            try self.chunks.append(self.allocator, .{
                .file_path = file_path_copy,
                .start_line = chunk_start + 1,
                .end_line = line_count,
                .content = chunk_content,
            });
            chunk_count += 1;
        }

        // Store file metadata
        const key = try self.allocator.dupe(u8, path);
        errdefer self.allocator.free(key);
        const meta_path = try self.allocator.dupe(u8, path);

        try self.indexed_files.put(self.allocator, key, .{
            .path = meta_path,
            .size_bytes = content.len,
            .line_count = line_count,
            .chunk_count = chunk_count,
            .last_indexed = 0,
        });
    }

    /// Query the index for relevant code snippets.
    pub fn query(self: *Self, question: []const u8) !CodebaseAnswer {
        var result = CodebaseAnswer{
            .allocator = self.allocator,
            .snippets = .{},
            .summary = try self.allocator.dupe(u8, ""),
        };
        errdefer result.deinit();

        // Simple keyword search across chunks
        const keywords = try extractKeywords(self.allocator, question);
        defer self.allocator.free(keywords);

        for (self.chunks.items) |chunk| {
            var score: f32 = 0;
            var kw_iter = std.mem.splitScalar(u8, keywords, ' ');
            while (kw_iter.next()) |kw| {
                if (kw.len < 2) continue;
                if (containsIgnoreCase(chunk.content, kw)) {
                    score += 1.0;
                }
            }

            if (score > 0) {
                try result.snippets.append(self.allocator, .{
                    .file_path = try self.allocator.dupe(u8, chunk.file_path),
                    .start_line = chunk.start_line,
                    .end_line = chunk.end_line,
                    .content = try self.allocator.dupe(u8, chunk.content),
                    .relevance_score = score,
                });
            }
        }

        // Sort by relevance (highest first)
        const SortCtx = struct {
            pub fn lessThan(_: @This(), a: CodeSnippet, b: CodeSnippet) bool {
                return a.relevance_score > b.relevance_score;
            }
        };
        std.mem.sort(CodeSnippet, result.snippets.items, SortCtx{}, SortCtx.lessThan);

        // Limit to top 10
        while (result.snippets.items.len > 10) {
            const removed = result.snippets.pop();
            self.allocator.free(removed.file_path);
            self.allocator.free(removed.content);
        }

        // Build summary
        self.allocator.free(result.summary);
        result.summary = try std.fmt.allocPrint(self.allocator, "Found {d} relevant snippets across {d} indexed files ({d} total chunks)", .{
            result.snippets.items.len,
            self.indexed_files.count(),
            self.chunks.items.len,
        });

        return result;
    }

    /// Get index statistics.
    pub fn getStats(self: *const Self) IndexStats {
        var total_size: usize = 0;
        var iter = self.indexed_files.iterator();
        while (iter.next()) |entry| {
            total_size += entry.value_ptr.size_bytes;
        }
        return .{
            .file_count = self.indexed_files.count(),
            .total_chunks = self.chunks.items.len,
            .total_size_bytes = total_size,
            .index_path = self.index_dir,
        };
    }
};

/// Case-insensitive substring search.
fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0) return true;
    if (needle.len > haystack.len) return false;
    var i: usize = 0;
    while (i + needle.len <= haystack.len) : (i += 1) {
        if (std.ascii.eqlIgnoreCase(haystack[i..][0..needle.len], needle)) return true;
    }
    return false;
}

/// Extract simple keywords from a question for search.
fn extractKeywords(allocator: std.mem.Allocator, question: []const u8) ![]u8 {
    // Filter out common stop words and return the rest
    var result = std.ArrayListUnmanaged(u8){};
    errdefer result.deinit(allocator);

    const stop_words = [_][]const u8{
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "how", "does", "what", "where", "when", "why", "who", "which",
        "do", "did", "will", "can", "could", "would", "should",
        "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "and", "or", "but", "not", "this", "that", "it", "its",
    };

    var words = std.mem.splitScalar(u8, question, ' ');
    var first = true;
    while (words.next()) |word| {
        var is_stop = false;
        for (stop_words) |sw| {
            if (std.ascii.eqlIgnoreCase(word, sw)) {
                is_stop = true;
                break;
            }
        }

        if (!is_stop and word.len >= 2) {
            if (!first) try result.append(allocator, ' ');
            try result.appendSlice(allocator, word);
            first = false;
        }
    }

    return try result.toOwnedSlice(allocator);
}

// ============================================================================
// Tests
// ============================================================================

test "CodebaseIndex - init and deinit" {
    const allocator = std.testing.allocator;
    var index = try CodebaseIndex.init(allocator, "/test/project");
    defer index.deinit();

    try std.testing.expectEqualStrings("/test/project", index.root_path);
    const stats = index.getStats();
    try std.testing.expectEqual(@as(usize, 0), stats.file_count);
}

test "CodebaseIndex - index file" {
    const allocator = std.testing.allocator;
    var index = try CodebaseIndex.init(allocator, "/test");
    defer index.deinit();

    const content = "line 1\nline 2\nline 3\nline 4\nline 5\n";
    try index.indexFile("src/main.zig", content);

    const stats = index.getStats();
    try std.testing.expectEqual(@as(usize, 1), stats.file_count);
    try std.testing.expect(stats.total_chunks > 0);
}

test "CodebaseIndex - query" {
    const allocator = std.testing.allocator;
    var index = try CodebaseIndex.init(allocator, "/test");
    defer index.deinit();

    const content = "pub fn init(allocator: std.mem.Allocator) !Self {\n    return .{};\n}\n";
    try index.indexFile("src/main.zig", content);

    var answer = try index.query("init function allocator");
    defer answer.deinit();

    try std.testing.expect(answer.snippets.items.len > 0);
}

test "IndexStats - default" {
    const allocator = std.testing.allocator;
    var index = try CodebaseIndex.init(allocator, "/test");
    defer index.deinit();

    const stats = index.getStats();
    try std.testing.expectEqual(@as(usize, 0), stats.file_count);
    try std.testing.expectEqual(@as(usize, 0), stats.total_chunks);
    try std.testing.expectEqual(@as(usize, 0), stats.total_size_bytes);
}
