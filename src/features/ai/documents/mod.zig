//! Document Understanding Module
//!
//! Provides comprehensive document processing capabilities including:
//! - Text extraction and segmentation
//! - Layout analysis (tables, lists, headings)
//! - Document structure understanding
//! - Named entity recognition
//! - Document summarization
//! - Question answering over documents
//!
//! ## Features
//!
//! - **DocumentParser**: Parse various document formats
//! - **LayoutAnalyzer**: Detect document structure (tables, lists, sections)
//! - **TextSegmenter**: Segment text into semantic chunks
//! - **EntityExtractor**: Extract named entities
//! - **DocumentEmbedder**: Generate embeddings for retrieval
//! - **DocumentQA**: Question answering over documents
//!
//! ## Example
//!
//! ```zig
//! const docs = @import("abi").ai.documents;
//!
//! var pipeline = try docs.DocumentPipeline.init(allocator, .{});
//! defer pipeline.deinit();
//!
//! const doc = try pipeline.parse("document.txt", content);
//! const summary = try pipeline.summarize(&doc, .{ .max_length = 200 });
//! const answer = try pipeline.answer(&doc, "What is the main topic?");
//! ```

const std = @import("std");
const build_options = @import("build_options");

// Sub-modules
pub const parser = @import("parser.zig");
pub const pipeline = @import("pipeline.zig");

// ============================================================================
// Document Types
// ============================================================================

/// Document format types
pub const DocumentFormat = enum {
    plain_text,
    markdown,
    html,
    json,
    xml,
    csv,
    code,
    unknown,

    /// Detect format from file extension
    pub fn fromExtension(ext: []const u8) DocumentFormat {
        if (std.mem.eql(u8, ext, ".txt")) return .plain_text;
        if (std.mem.eql(u8, ext, ".md")) return .markdown;
        if (std.mem.eql(u8, ext, ".html") or std.mem.eql(u8, ext, ".htm")) return .html;
        if (std.mem.eql(u8, ext, ".json")) return .json;
        if (std.mem.eql(u8, ext, ".xml")) return .xml;
        if (std.mem.eql(u8, ext, ".csv")) return .csv;
        // Code files
        if (std.mem.eql(u8, ext, ".zig") or
            std.mem.eql(u8, ext, ".rs") or
            std.mem.eql(u8, ext, ".py") or
            std.mem.eql(u8, ext, ".js") or
            std.mem.eql(u8, ext, ".ts") or
            std.mem.eql(u8, ext, ".go") or
            std.mem.eql(u8, ext, ".c") or
            std.mem.eql(u8, ext, ".cpp") or
            std.mem.eql(u8, ext, ".h"))
        {
            return .code;
        }
        return .unknown;
    }
};

/// Document element types found during layout analysis
pub const ElementType = enum {
    heading1,
    heading2,
    heading3,
    heading4,
    heading5,
    heading6,
    paragraph,
    list_item,
    ordered_list,
    unordered_list,
    table,
    table_row,
    table_cell,
    code_block,
    inline_code,
    blockquote,
    link,
    image,
    horizontal_rule,
    emphasis,
    strong,
    strikethrough,
    unknown,
};

/// A structural element within a document
pub const DocumentElement = struct {
    element_type: ElementType,
    content: []const u8,
    start_offset: usize,
    end_offset: usize,
    level: u8 = 0, // For nested elements
    children: []DocumentElement = &.{},
    metadata: ElementMetadata = .{},

    pub const ElementMetadata = struct {
        language: ?[]const u8 = null, // For code blocks
        link_url: ?[]const u8 = null, // For links
        alt_text: ?[]const u8 = null, // For images
    };
};

/// A parsed document with structure
pub const Document = struct {
    allocator: std.mem.Allocator,
    /// Original raw content
    raw_content: []const u8,
    /// Detected format
    format: DocumentFormat,
    /// Structural elements
    elements: []DocumentElement,
    /// Document metadata
    metadata: DocumentMetadata,
    /// Extracted text segments
    segments: []TextSegment,
    /// Extracted entities
    entities: []NamedEntity,

    pub const DocumentMetadata = struct {
        title: ?[]const u8 = null,
        author: ?[]const u8 = null,
        date: ?[]const u8 = null,
        word_count: usize = 0,
        char_count: usize = 0,
        line_count: usize = 0,
        language: ?[]const u8 = null,
    };

    pub fn deinit(self: *Document) void {
        self.allocator.free(self.raw_content);
        self.allocator.free(self.elements);
        self.allocator.free(self.segments);
        self.allocator.free(self.entities);
    }
};

/// A semantic text segment
pub const TextSegment = struct {
    content: []const u8,
    start_offset: usize,
    end_offset: usize,
    segment_type: SegmentType,
    importance_score: f32 = 0.5,

    pub const SegmentType = enum {
        sentence,
        paragraph,
        section,
        chunk, // Fixed-size chunk for embeddings
    };
};

/// Named entity types
pub const EntityType = enum {
    person,
    organization,
    location,
    date,
    time,
    money,
    percentage,
    email,
    url,
    phone,
    code_identifier,
    file_path,
    version,
    custom,
};

/// A named entity extracted from text
pub const NamedEntity = struct {
    text: []const u8,
    entity_type: EntityType,
    start_offset: usize,
    end_offset: usize,
    confidence: f32 = 1.0,
};

// ============================================================================
// Re-exports from sub-modules
// ============================================================================

// Parser re-exports
pub const SegmentationConfig = parser.SegmentationConfig;
pub const TextSegmenter = parser.TextSegmenter;
pub const EntityExtractor = parser.EntityExtractor;
pub const LayoutAnalyzer = parser.LayoutAnalyzer;

// Pipeline re-exports
pub const PipelineConfig = pipeline.PipelineConfig;
pub const DocumentPipeline = pipeline.DocumentPipeline;

// ============================================================================
// Public Exports
// ============================================================================

pub const Error = DocumentPipeline.Error || error{OutOfMemory};

/// Check if documents feature is enabled
pub fn isEnabled() bool {
    return build_options.feat_ai;
}

// ============================================================================
// Tests
// ============================================================================

test "DocumentFormat detection" {
    try std.testing.expectEqual(DocumentFormat.markdown, DocumentFormat.fromExtension(".md"));
    try std.testing.expectEqual(DocumentFormat.plain_text, DocumentFormat.fromExtension(".txt"));
    try std.testing.expectEqual(DocumentFormat.code, DocumentFormat.fromExtension(".zig"));
    try std.testing.expectEqual(DocumentFormat.unknown, DocumentFormat.fromExtension(".xyz"));
}

test "TextSegmenter sentences" {
    const allocator = std.testing.allocator;
    var segmenter = TextSegmenter.init(allocator, .{ .min_segment_size = 5 });

    const text = "Hello world. This is a test. Another sentence here!";
    const segments = try segmenter.segmentSentences(text);
    defer allocator.free(segments);

    try std.testing.expect(segments.len >= 2);
}

test "TextSegmenter chunks" {
    const allocator = std.testing.allocator;
    var segmenter = TextSegmenter.init(allocator, .{
        .chunk_size = 50,
        .chunk_overlap = 10,
        .min_segment_size = 5,
    });

    const text = "This is a longer text that should be split into multiple chunks for processing. " ++
        "Each chunk will have some overlap with the previous one to maintain context.";
    const segments = try segmenter.segmentChunks(text);
    defer allocator.free(segments);

    try std.testing.expect(segments.len >= 2);
}

test "EntityExtractor URLs" {
    const allocator = std.testing.allocator;
    var extractor = EntityExtractor.init(allocator);

    const text = "Check out https://example.com and http://test.org for more info.";
    const entities = try extractor.extractAll(text);
    defer allocator.free(entities);

    var url_count: usize = 0;
    for (entities) |e| {
        if (e.entity_type == .url) url_count += 1;
    }
    try std.testing.expect(url_count >= 2);
}

test "LayoutAnalyzer markdown" {
    const allocator = std.testing.allocator;
    var analyzer = LayoutAnalyzer.init(allocator);

    const markdown =
        \\# Heading 1
        \\
        \\Some paragraph text here.
        \\
        \\## Heading 2
        \\
        \\- List item 1
        \\- List item 2
        \\
        \\```
        \\code block
        \\```
    ;

    const elements = try analyzer.analyzeMarkdown(markdown);
    defer allocator.free(elements);

    var heading_count: usize = 0;
    var list_count: usize = 0;
    for (elements) |e| {
        if (e.element_type == .heading1 or e.element_type == .heading2) heading_count += 1;
        if (e.element_type == .list_item) list_count += 1;
    }

    try std.testing.expect(heading_count >= 2);
    try std.testing.expect(list_count >= 2);
}

test "DocumentPipeline parse" {
    const allocator = std.testing.allocator;
    var doc_pipeline = DocumentPipeline.init(allocator, .{});
    defer doc_pipeline.deinit();

    const content = "# Test Document\n\nThis is test content with https://example.com link.";
    var doc = try doc_pipeline.parse("test.md", content);
    defer doc.deinit();

    try std.testing.expectEqual(DocumentFormat.markdown, doc.format);
    try std.testing.expect(doc.metadata.word_count > 0);
    try std.testing.expect(doc.segments.len > 0);
}

test "LayoutAnalyzer stats" {
    const allocator = std.testing.allocator;
    var analyzer = LayoutAnalyzer.init(allocator);

    const text = "Hello world.\nThis is a test.\nThree lines here.";
    const stats = analyzer.computeStats(text);

    try std.testing.expect(stats.word_count >= 9);
    try std.testing.expectEqual(@as(usize, 3), stats.line_count);
}

test {
    _ = parser;
    _ = pipeline;
    std.testing.refAllDecls(@This());
}
