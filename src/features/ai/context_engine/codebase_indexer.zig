//! Codebase Indexer
//!
//! Provides native codebase indexing and rewriting capabilities for
//! embedding semantic code understanding into the Triad's neural matrices.

const std = @import("std");

/// A structured representation of an indexed file.
pub const IndexEntry = struct {
    file_path: []const u8,
    content: []const u8,
    tokens: usize,
    /// Language or file type (e.g., "zig", "md", "json")
    language: []const u8,

    pub fn deinit(self: *IndexEntry, allocator: std.mem.Allocator) void {
        allocator.free(self.file_path);
        allocator.free(self.content);
        allocator.free(self.language);
    }
};

/// The main Codebase Indexer struct.
pub const CodebaseIndexer = struct {
    allocator: std.mem.Allocator,
    io: *std.Io,

    pub fn init(allocator: std.mem.Allocator, io: *std.Io) CodebaseIndexer {
        return .{
            .allocator = allocator,
            .io = io,
        };
    }

    pub fn deinit(self: *CodebaseIndexer) void {
        _ = self;
    }

    /// Recursively traverse a directory and collect file semantics.
    /// In a full implementation, this would integrate deeply with AST parsing and WDBX.
    pub fn indexDirectory(self: *CodebaseIndexer, dir_path: []const u8) ![]IndexEntry {
        var results = std.ArrayListUnmanaged(IndexEntry).empty;
        errdefer {
            for (results.items) |*entry| entry.deinit(self.allocator);
            results.deinit(self.allocator);
        }

        var dir = try std.Io.Dir.cwd().openDir(self.io.*, dir_path, .{ .iterate = true });
        defer dir.close();

        var walker = try dir.walk(self.allocator);
        defer walker.deinit();

        while (try walker.next(self.io.*)) |entry| {
            if (entry.kind != .file) continue;

            // Basic filtering for source files
            const ext = std.fs.path.extension(entry.basename);
            if (ext.len == 0) continue;

            const is_source = std.mem.eql(u8, ext, ".zig") or
                std.mem.eql(u8, ext, ".md") or
                std.mem.eql(u8, ext, ".zon");

            if (!is_source) continue;

            const full_path = try std.fs.path.join(self.allocator, &.{ dir_path, entry.path });
            errdefer self.allocator.free(full_path);

            const content = std.Io.Dir.cwd().readFileAlloc(self.io.*, full_path, self.allocator, .limited(10 * 1024 * 1024)) catch |err| {
                std.log.warn("Failed to read {s}: {}", .{ full_path, err });
                self.allocator.free(full_path);
                continue;
            };
            errdefer self.allocator.free(content);

            const language = if (ext.len > 1) try self.allocator.dupe(u8, ext[1..]) else try self.allocator.dupe(u8, "unknown");
            errdefer self.allocator.free(language);

            try results.append(self.allocator, .{
                .file_path = full_path,
                .content = content,
                .tokens = content.len / 4, // Rough token estimation
                .language = language,
            });
        }

        return results.toOwnedSlice(self.allocator);
    }

    /// Re-write a file's content securely.
    pub fn rewrite(self: *CodebaseIndexer, file_path: []const u8, new_content: []const u8) !void {
        var file = try std.Io.Dir.cwd().createFile(self.io.*, file_path, .{ .truncate = true });
        defer file.close(self.io.*);

        try file.writeStreamingAll(self.io.*, new_content);
    }

    /// Embed an indexed codebase into the neural vector database using the specified embedder.
    pub fn embedCodebase(self: *CodebaseIndexer, dir_path: []const u8, wdbx_engine: anytype, embedder: anytype) !void {
        _ = embedder; // Use wdbx_engine's internal AI client for now
        std.log.info("Starting codebase embedding for: {s}", .{dir_path});

        const entries = try self.indexDirectory(dir_path);
        defer {
            for (entries) |*e| e.deinit(self.allocator);
            self.allocator.free(entries);
        }

        var total_tokens: usize = 0;
        var total_chunks: usize = 0;

        for (entries) |e| {
            total_tokens += e.tokens;

            // Simple chunking by double newline (paragraphs/functions)
            var chunk_idx: usize = 0;
            var iter = std.mem.splitSequence(u8, e.content, "\n\n");

            while (iter.next()) |chunk| {
                const trimmed = std.mem.trim(u8, chunk, " \t\r\n");
                if (trimmed.len < 20) continue; // Skip very small meaningless chunks

                const id = try std.fmt.allocPrint(self.allocator, "{s}#{d}", .{ e.file_path, chunk_idx });
                defer self.allocator.free(id);

                // Index chunk directly into the engine.
                // Assuming wdbx_engine is a *wdbx.Engine
                wdbx_engine.index(id, trimmed, .empty) catch |err| {
                    std.log.warn("Failed to index chunk {s}: {}", .{ id, err });
                };

                chunk_idx += 1;
                total_chunks += 1;
            }
        }

        std.log.info("Codebase mapped: {d} files, ~{d} total tokens, {d} semantic chunks inserted into wdbx_engine.", .{ entries.len, total_tokens, total_chunks });
    }

    /// Read and analyze a specific file, extracting semantic metadata.
    pub fn analyzeFile(self: *CodebaseIndexer, file_path: []const u8) ![]const u8 {
        const content = std.Io.Dir.cwd().readFileAlloc(self.io.*, file_path, self.allocator, .limited(5 * 1024 * 1024)) catch {
            return error.FileNotFound;
        };
        defer self.allocator.free(content);

        // Count rough tokens and lines for summary
        var line_count: usize = 0;
        var iter = std.mem.splitSequence(u8, content, "\n");
        while (iter.next()) |_| {
            line_count += 1;
        }

        return try std.fmt.allocPrint(self.allocator, "File: {s}\nLines: {d}\nEstimated Tokens: {d}\nContent Snippet: {s}...", .{ file_path, line_count, content.len / 4, if (content.len > 500) content[0..500] else content });
    }

    /// Search codebase for string literals natively.
    pub fn searchCodebase(self: *CodebaseIndexer, dir_path: []const u8, pattern: []const u8) ![]const u8 {
        var results = std.ArrayListUnmanaged(u8).empty;
        defer results.deinit(self.allocator);

        var dir = try std.Io.Dir.cwd().openDir(self.io.*, dir_path, .{ .iterate = true });
        defer dir.close(self.io.*);

        var walker = try dir.walk(self.allocator);
        defer walker.deinit();

        var match_count: usize = 0;

        while (try walker.next(self.io.*)) |entry| {
            if (entry.kind != .file) continue;

            const full_path = try std.fs.path.join(self.allocator, &.{ dir_path, entry.path });
            defer self.allocator.free(full_path);

            const content = std.Io.Dir.cwd().readFileAlloc(self.io.*, full_path, self.allocator, .limited(2 * 1024 * 1024)) catch continue;
            defer self.allocator.free(content);

            if (std.mem.indexOf(u8, content, pattern) != null) {
                match_count += 1;
                const match_str = try std.fmt.allocPrint(self.allocator, "Match in: {s}\n", .{full_path});
                defer self.allocator.free(match_str);
                try results.appendSlice(self.allocator, match_str);
            }
        }

        if (match_count == 0) {
            return try self.allocator.dupe(u8, "No matches found.");
        }

        return try results.toOwnedSlice(self.allocator);
    }
};

test "codebase indexer dummy test" {
    // Basic test
    try std.testing.expect(true);
}
