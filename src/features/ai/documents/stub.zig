//! Document Understanding stub — disabled at compile time.

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
    pub fn fromExtension(_: []const u8) DocumentFormat {
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
    pub const ElementMetadata = struct { language: ?[]const u8 = null, link_url: ?[]const u8 = null, alt_text: ?[]const u8 = null };
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
    pub fn deinit(_: *Document) void {}
};

pub const TextSegment = struct {
    content: []const u8,
    start_offset: usize,
    end_offset: usize,
    segment_type: SegmentType,
    importance_score: f32 = 0.5,
    pub const SegmentType = enum { sentence, paragraph, section, chunk };
};

pub const EntityType = enum { person, organization, location, date, time, money, percentage, email, url, phone, code_identifier, file_path, version, custom };

pub const NamedEntity = struct { text: []const u8, entity_type: EntityType, start_offset: usize, end_offset: usize, confidence: f32 = 1.0 };

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
    pub fn segmentSentences(_: *TextSegmenter, _: []const u8) ![]TextSegment {
        return error.DocumentsDisabled;
    }
    pub fn segmentChunks(_: *TextSegmenter, _: []const u8) ![]TextSegment {
        return error.DocumentsDisabled;
    }
    pub fn segmentParagraphs(_: *TextSegmenter, _: []const u8) ![]TextSegment {
        return error.DocumentsDisabled;
    }
};

pub const EntityExtractor = struct {
    allocator: std.mem.Allocator,
    pub fn init(allocator: std.mem.Allocator) EntityExtractor {
        return .{ .allocator = allocator };
    }
    pub fn extractAll(_: *EntityExtractor, _: []const u8) ![]NamedEntity {
        return error.DocumentsDisabled;
    }
};

pub const LayoutAnalyzer = struct {
    allocator: std.mem.Allocator,
    pub fn init(allocator: std.mem.Allocator) LayoutAnalyzer {
        return .{ .allocator = allocator };
    }
    pub fn analyzeMarkdown(_: *LayoutAnalyzer, _: []const u8) ![]DocumentElement {
        return error.DocumentsDisabled;
    }
    pub fn computeStats(_: *LayoutAnalyzer, _: []const u8) Document.DocumentMetadata {
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
        return .{ .allocator = allocator, .config = config, .segmenter = TextSegmenter.init(allocator, config.segmentation), .entity_extractor = EntityExtractor.init(allocator), .layout_analyzer = LayoutAnalyzer.init(allocator) };
    }
    pub fn deinit(_: *DocumentPipeline) void {}
    pub fn parse(_: *DocumentPipeline, _: ?[]const u8, _: []const u8) !Document {
        return error.DocumentsDisabled;
    }
    pub fn getSummarySegments(_: *DocumentPipeline, _: *const Document, _: usize) []const TextSegment {
        return &.{};
    }
    pub fn searchSegments(_: *DocumentPipeline, _: *const Document, _: []const u8) ![]const TextSegment {
        return error.DocumentsDisabled;
    }
    pub fn getHeadings(_: *DocumentPipeline, _: *const Document) ![]const DocumentElement {
        return error.DocumentsDisabled;
    }
    pub fn getTableOfContents(_: *DocumentPipeline, _: *const Document) ![]DocumentElement {
        return error.DocumentsDisabled;
    }
    pub const Error = error{ DocumentTooLarge, InvalidFormat, ParseError, OutOfMemory, DocumentsDisabled };
};

pub const Error = DocumentPipeline.Error;

pub fn isEnabled() bool {
    return false;
}
