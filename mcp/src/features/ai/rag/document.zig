//! Document types for RAG pipeline.
//!
//! Provides document representation and metadata structures
//! for indexing and retrieval.

const std = @import("std");

/// Document type classification.
pub const DocumentType = enum {
    /// Plain text document.
    text,
    /// Markdown document.
    markdown,
    /// HTML document.
    html,
    /// Code file.
    code,
    /// PDF document.
    pdf,
    /// JSON data.
    json,
    /// Unknown/other format.
    other,

    /// Infer document type from file extension.
    pub fn fromExtension(ext: []const u8) DocumentType {
        if (std.mem.eql(u8, ext, ".md") or std.mem.eql(u8, ext, ".markdown")) {
            return .markdown;
        } else if (std.mem.eql(u8, ext, ".html") or std.mem.eql(u8, ext, ".htm")) {
            return .html;
        } else if (std.mem.eql(u8, ext, ".pdf")) {
            return .pdf;
        } else if (std.mem.eql(u8, ext, ".json")) {
            return .json;
        } else if (std.mem.eql(u8, ext, ".py") or
            std.mem.eql(u8, ext, ".js") or
            std.mem.eql(u8, ext, ".ts") or
            std.mem.eql(u8, ext, ".zig") or
            std.mem.eql(u8, ext, ".rs") or
            std.mem.eql(u8, ext, ".go") or
            std.mem.eql(u8, ext, ".c") or
            std.mem.eql(u8, ext, ".cpp") or
            std.mem.eql(u8, ext, ".java"))
        {
            return .code;
        } else if (std.mem.eql(u8, ext, ".txt")) {
            return .text;
        }
        return .other;
    }
};

/// Document metadata.
pub const DocumentMetadata = struct {
    /// Document source URL or path.
    source: ?[]const u8 = null,
    /// Author name.
    author: ?[]const u8 = null,
    /// Creation timestamp.
    created_at: ?i64 = null,
    /// Last modified timestamp.
    modified_at: ?i64 = null,
    /// Document language.
    language: ?[]const u8 = null,
    /// Custom tags.
    tags: ?[]const []const u8 = null,
    /// Additional custom fields (JSON).
    custom: ?[]const u8 = null,

    pub fn clone(self: DocumentMetadata, allocator: std.mem.Allocator) !DocumentMetadata {
        return .{
            .source = if (self.source) |s| try allocator.dupe(u8, s) else null,
            .author = if (self.author) |a| try allocator.dupe(u8, a) else null,
            .created_at = self.created_at,
            .modified_at = self.modified_at,
            .language = if (self.language) |l| try allocator.dupe(u8, l) else null,
            .tags = if (self.tags) |tags| blk: {
                var cloned = try allocator.alloc([]const u8, tags.len);
                for (tags, 0..) |tag, i| {
                    cloned[i] = try allocator.dupe(u8, tag);
                }
                break :blk cloned;
            } else null,
            .custom = if (self.custom) |c| try allocator.dupe(u8, c) else null,
        };
    }

    pub fn deinit(self: *DocumentMetadata, allocator: std.mem.Allocator) void {
        if (self.source) |s| allocator.free(s);
        if (self.author) |a| allocator.free(a);
        if (self.language) |l| allocator.free(l);
        if (self.tags) |tags| {
            for (tags) |tag| {
                allocator.free(tag);
            }
            allocator.free(tags);
        }
        if (self.custom) |c| allocator.free(c);
        self.* = undefined;
    }
};

/// A document in the RAG corpus.
pub const Document = struct {
    /// Unique document identifier.
    id: []const u8,
    /// Document title.
    title: ?[]const u8,
    /// Document content.
    content: []const u8,
    /// Document type.
    doc_type: DocumentType,
    /// Document metadata.
    metadata: ?DocumentMetadata,

    /// Create a text document.
    pub fn text(id: []const u8, content: []const u8) Document {
        return .{
            .id = id,
            .title = null,
            .content = content,
            .doc_type = .text,
            .metadata = null,
        };
    }

    /// Create a document with title.
    pub fn withTitle(id: []const u8, title: []const u8, content: []const u8) Document {
        return .{
            .id = id,
            .title = title,
            .content = content,
            .doc_type = .text,
            .metadata = null,
        };
    }

    /// Create a code document.
    pub fn code(id: []const u8, content: []const u8, language: ?[]const u8) Document {
        return .{
            .id = id,
            .title = null,
            .content = content,
            .doc_type = .code,
            .metadata = if (language) |l| DocumentMetadata{ .language = l } else null,
        };
    }

    /// Clone the document.
    pub fn clone(self: Document, allocator: std.mem.Allocator) !Document {
        return .{
            .id = try allocator.dupe(u8, self.id),
            .title = if (self.title) |t| try allocator.dupe(u8, t) else null,
            .content = try allocator.dupe(u8, self.content),
            .doc_type = self.doc_type,
            .metadata = if (self.metadata) |m| try m.clone(allocator) else null,
        };
    }

    /// Free document resources.
    pub fn deinit(self: *Document, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        if (self.title) |t| allocator.free(t);
        allocator.free(self.content);
        if (self.metadata) |*m| {
            var meta = m.*;
            meta.deinit(allocator);
        }
        self.* = undefined;
    }

    /// Get document length in characters.
    pub fn length(self: *const Document) usize {
        return self.content.len;
    }

    /// Estimate token count (roughly 4 chars per token).
    pub fn estimateTokens(self: *const Document) usize {
        return (self.content.len + 3) / 4;
    }

    /// Get a preview of the document content.
    pub fn preview(self: *const Document, max_len: usize) []const u8 {
        return self.content[0..@min(max_len, self.content.len)];
    }
};

/// Document collection statistics.
pub const CollectionStats = struct {
    /// Total number of documents.
    document_count: usize,
    /// Total content length.
    total_length: usize,
    /// Average document length.
    avg_length: f64,
    /// Document type distribution.
    type_counts: TypeCounts,

    const TypeCounts = struct {
        text: usize = 0,
        markdown: usize = 0,
        html: usize = 0,
        code: usize = 0,
        pdf: usize = 0,
        json: usize = 0,
        other: usize = 0,
    };
};

/// Compute statistics for a document collection.
pub fn computeStats(documents: []const Document) CollectionStats {
    var stats = CollectionStats{
        .document_count = documents.len,
        .total_length = 0,
        .avg_length = 0,
        .type_counts = .{},
    };

    for (documents) |doc| {
        stats.total_length += doc.content.len;

        switch (doc.doc_type) {
            .text => stats.type_counts.text += 1,
            .markdown => stats.type_counts.markdown += 1,
            .html => stats.type_counts.html += 1,
            .code => stats.type_counts.code += 1,
            .pdf => stats.type_counts.pdf += 1,
            .json => stats.type_counts.json += 1,
            .other => stats.type_counts.other += 1,
        }
    }

    if (documents.len > 0) {
        stats.avg_length = @as(f64, @floatFromInt(stats.total_length)) /
            @as(f64, @floatFromInt(documents.len));
    }

    return stats;
}

test "document creation" {
    const doc = Document.text("doc1", "Hello world");
    try std.testing.expectEqualStrings("doc1", doc.id);
    try std.testing.expectEqualStrings("Hello world", doc.content);
    try std.testing.expectEqual(DocumentType.text, doc.doc_type);
}

test "document with title" {
    const doc = Document.withTitle("doc2", "My Title", "Content here");
    try std.testing.expectEqualStrings("My Title", doc.title.?);
}

test "document clone and deinit" {
    const allocator = std.testing.allocator;

    const original = Document{
        .id = "test_id",
        .title = "Test Title",
        .content = "Test content",
        .doc_type = .markdown,
        .metadata = DocumentMetadata{ .author = "Author" },
    };

    var cloned = try original.clone(allocator);
    defer cloned.deinit(allocator);

    try std.testing.expectEqualStrings(original.id, cloned.id);
    try std.testing.expectEqualStrings(original.content, cloned.content);
}

test "document type from extension" {
    try std.testing.expectEqual(DocumentType.markdown, DocumentType.fromExtension(".md"));
    try std.testing.expectEqual(DocumentType.code, DocumentType.fromExtension(".py"));
    try std.testing.expectEqual(DocumentType.html, DocumentType.fromExtension(".html"));
    try std.testing.expectEqual(DocumentType.other, DocumentType.fromExtension(".xyz"));
}

test {
    std.testing.refAllDecls(@This());
}
