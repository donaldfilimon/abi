//! Document Understanding Module Stub
//!
//! Stub implementation when AI features are disabled.

const std = @import("std");

pub const DocumentFormat = enum {
    plain_text,
    markdown,
    html,
    json,
    xml,
    csv,
    code,
    unknown,

    pub fn fromExtension(ext: []const u8) DocumentFormat {
        _ = ext;
        return .unknown;
    }
};

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

pub const DocumentElement = struct {
    element_type: ElementType,
    content: []const u8,
    start_offset: usize,
    end_offset: usize,
    level: u8 = 0,
    children: []DocumentElement = &.{},
    metadata: ElementMetadata = .{},

    pub const ElementMetadata = struct {
        language: ?[]const u8 = null,
        link_url: ?[]const u8 = null,
        alt_text: ?[]const u8 = null,
    };
};

pub const Document = struct {
    allocator: std.mem.Allocator,
    raw_content: []const u8,
    format: DocumentFormat,
    elements: []DocumentElement,
    metadata: DocumentMetadata,
    segments: []TextSegment,
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
        _ = self;
    }
};

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
        chunk,
    };
};

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

pub const NamedEntity = struct {
    text: []const u8,
    entity_type: EntityType,
    start_offset: usize,
    end_offset: usize,
    confidence: f32 = 1.0,
};

pub const SegmentationConfig = struct {
    chunk_size: usize = 512,
    chunk_overlap: usize = 64,
    respect_sentences: bool = true,
    respect_paragraphs: bool = true,
    min_segment_size: usize = 50,
};

pub const TextSegmenter = struct {
    allocator: std.mem.Allocator,
    config: SegmentationConfig,

    pub fn init(allocator: std.mem.Allocator, config: SegmentationConfig) TextSegmenter {
        return .{ .allocator = allocator, .config = config };
    }

    pub fn segmentSentences(self: *TextSegmenter, text: []const u8) ![]TextSegment {
        _ = self;
        _ = text;
        return error.DocumentsDisabled;
    }

    pub fn segmentChunks(self: *TextSegmenter, text: []const u8) ![]TextSegment {
        _ = self;
        _ = text;
        return error.DocumentsDisabled;
    }

    pub fn segmentParagraphs(self: *TextSegmenter, text: []const u8) ![]TextSegment {
        _ = self;
        _ = text;
        return error.DocumentsDisabled;
    }
};

pub const EntityExtractor = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) EntityExtractor {
        return .{ .allocator = allocator };
    }

    pub fn extractAll(self: *EntityExtractor, text: []const u8) ![]NamedEntity {
        _ = self;
        _ = text;
        return error.DocumentsDisabled;
    }
};

pub const LayoutAnalyzer = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) LayoutAnalyzer {
        return .{ .allocator = allocator };
    }

    pub fn analyzeMarkdown(self: *LayoutAnalyzer, text: []const u8) ![]DocumentElement {
        _ = self;
        _ = text;
        return error.DocumentsDisabled;
    }

    pub fn computeStats(self: *LayoutAnalyzer, text: []const u8) Document.DocumentMetadata {
        _ = self;
        _ = text;
        return .{};
    }
};

pub const PipelineConfig = struct {
    segmentation: SegmentationConfig = .{},
    extract_entities: bool = true,
    analyze_layout: bool = true,
    max_document_size: usize = 10 * 1024 * 1024,
};

pub const DocumentPipeline = struct {
    allocator: std.mem.Allocator,
    config: PipelineConfig,
    segmenter: TextSegmenter,
    entity_extractor: EntityExtractor,
    layout_analyzer: LayoutAnalyzer,

    pub fn init(allocator: std.mem.Allocator, config: PipelineConfig) DocumentPipeline {
        return .{
            .allocator = allocator,
            .config = config,
            .segmenter = TextSegmenter.init(allocator, config.segmentation),
            .entity_extractor = EntityExtractor.init(allocator),
            .layout_analyzer = LayoutAnalyzer.init(allocator),
        };
    }

    pub fn deinit(self: *DocumentPipeline) void {
        _ = self;
    }

    pub fn parse(self: *DocumentPipeline, filename: ?[]const u8, content: []const u8) !Document {
        _ = self;
        _ = filename;
        _ = content;
        return error.DocumentsDisabled;
    }

    pub fn getSummarySegments(self: *DocumentPipeline, doc: *const Document, max_segments: usize) []const TextSegment {
        _ = self;
        _ = doc;
        _ = max_segments;
        return &.{};
    }

    pub fn searchSegments(self: *DocumentPipeline, doc: *const Document, query: []const u8) ![]const TextSegment {
        _ = self;
        _ = doc;
        _ = query;
        return error.DocumentsDisabled;
    }

    pub fn getHeadings(self: *DocumentPipeline, doc: *const Document) ![]const DocumentElement {
        _ = self;
        _ = doc;
        return error.DocumentsDisabled;
    }

    pub fn getTableOfContents(self: *DocumentPipeline, doc: *const Document) ![]DocumentElement {
        _ = self;
        _ = doc;
        return error.DocumentsDisabled;
    }

    pub const Error = error{
        DocumentTooLarge,
        InvalidFormat,
        ParseError,
        OutOfMemory,
        DocumentsDisabled,
    };
};

pub const Error = DocumentPipeline.Error;

pub fn isEnabled() bool {
    return false;
}
