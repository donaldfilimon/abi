//! Document Understanding stub types — extracted from stub.zig.

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

pub const DocumentMetadata = struct {
    title: ?[]const u8 = null,
    author: ?[]const u8 = null,
    date: ?[]const u8 = null,
    word_count: usize = 0,
    char_count: usize = 0,
    line_count: usize = 0,
    language: ?[]const u8 = null,
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

pub const PipelineConfig = struct {
    segmentation: SegmentationConfig = .{},
    extract_entities: bool = true,
    analyze_layout: bool = true,
    max_document_size: usize = 10 * 1024 * 1024,
};

pub const Error = error{ DocumentTooLarge, InvalidFormat, ParseError, OutOfMemory, FeatureDisabled };
