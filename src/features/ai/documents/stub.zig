//! Document Understanding stub — disabled at compile time.

const std = @import("std");
const types = @import("types.zig");

// Re-export types
pub const DocumentFormat = types.DocumentFormat;
pub const ElementType = types.ElementType;
pub const DocumentElement = types.DocumentElement;
pub const TextSegment = types.TextSegment;
pub const EntityType = types.EntityType;
pub const NamedEntity = types.NamedEntity;
pub const SegmentationConfig = types.SegmentationConfig;
pub const PipelineConfig = types.PipelineConfig;
pub const Error = types.Error;

pub const Document = struct {
    allocator: std.mem.Allocator,
    raw_content: []const u8,
    format: DocumentFormat,
    elements: []DocumentElement,
    metadata: types.DocumentMetadata,
    segments: []TextSegment,
    entities: []NamedEntity,
    pub const DocumentMetadata = types.DocumentMetadata;
    pub fn deinit(_: *Document) void {}
};

pub const TextSegmenter = struct {
    allocator: std.mem.Allocator,
    config: SegmentationConfig,
    pub fn init(allocator: std.mem.Allocator, config: SegmentationConfig) TextSegmenter {
        return .{ .allocator = allocator, .config = config };
    }
    pub fn segmentSentences(_: *TextSegmenter, _: []const u8) ![]TextSegment {
        return error.FeatureDisabled;
    }
    pub fn segmentChunks(_: *TextSegmenter, _: []const u8) ![]TextSegment {
        return error.FeatureDisabled;
    }
    pub fn segmentParagraphs(_: *TextSegmenter, _: []const u8) ![]TextSegment {
        return error.FeatureDisabled;
    }
};

pub const EntityExtractor = struct {
    allocator: std.mem.Allocator,
    pub fn init(allocator: std.mem.Allocator) EntityExtractor {
        return .{ .allocator = allocator };
    }
    pub fn extractAll(_: *EntityExtractor, _: []const u8) ![]NamedEntity {
        return error.FeatureDisabled;
    }
};

pub const LayoutAnalyzer = struct {
    allocator: std.mem.Allocator,
    pub fn init(allocator: std.mem.Allocator) LayoutAnalyzer {
        return .{ .allocator = allocator };
    }
    pub fn analyzeMarkdown(_: *LayoutAnalyzer, _: []const u8) ![]DocumentElement {
        return error.FeatureDisabled;
    }
    pub fn computeStats(_: *LayoutAnalyzer, _: []const u8) Document.DocumentMetadata {
        return .{};
    }
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
        return error.FeatureDisabled;
    }
    pub fn getSummarySegments(_: *DocumentPipeline, _: *const Document, _: usize) []const TextSegment {
        return &.{};
    }
    pub fn searchSegments(_: *DocumentPipeline, _: *const Document, _: []const u8) ![]const TextSegment {
        return error.FeatureDisabled;
    }
    pub fn getHeadings(_: *DocumentPipeline, _: *const Document) ![]const DocumentElement {
        return error.FeatureDisabled;
    }
    pub fn getTableOfContents(_: *DocumentPipeline, _: *const Document) ![]DocumentElement {
        return error.FeatureDisabled;
    }
    pub const PipelineError = Error;
};

pub fn isEnabled() bool {
    return false;
}

test {
    std.testing.refAllDecls(@This());
}
