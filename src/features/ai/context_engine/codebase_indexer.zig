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
        var results = std.ArrayList(IndexEntry).init(self.allocator);
        errdefer {
            for (results.items) |*entry| entry.deinit(self.allocator);
            results.deinit();
        }

        var dir = try std.Io.Dir.cwd().openDir(self.io.*, dir_path, .{ .iterate = true });
        defer dir.close();

        var walker = try dir.walk(self.allocator);
        defer walker.deinit();

        while (try walker.next()) |entry| {
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

            try results.append(.{
                .file_path = full_path,
                .content = content,
                .tokens = content.len / 4, // Rough token estimation
                .language = language,
            });
        }

        return results.toOwnedSlice();
    }

    /// Re-write a file's content securely.
    pub fn rewrite(self: *CodebaseIndexer, file_path: []const u8, new_content: []const u8) !void {
        var file = try std.Io.Dir.cwd().createFile(self.io.*, file_path, .{ .truncate = true });
        defer file.close(self.io.*);

        try file.writeStreamingAll(self.io.*, new_content);
    }
};

test "codebase indexer dummy test" {
    // Basic test
    try std.testing.expect(true);
}
