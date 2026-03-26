//! Document Processing Pipeline
//!
//! Orchestrates parsing, segmentation, entity extraction, and layout analysis.

const std = @import("std");
const types = @import("mod.zig");
const parser = @import("parser.zig");

pub const DocumentFormat = types.DocumentFormat;
pub const DocumentElement = types.DocumentElement;
pub const Document = types.Document;
pub const TextSegment = types.TextSegment;
pub const NamedEntity = types.NamedEntity;

pub const SegmentationConfig = parser.SegmentationConfig;
pub const TextSegmenter = parser.TextSegmenter;
pub const EntityExtractor = parser.EntityExtractor;
pub const LayoutAnalyzer = parser.LayoutAnalyzer;

// ============================================================================
// Document Pipeline
// ============================================================================

/// Configuration for document pipeline
pub const PipelineConfig = struct {
    /// Segmentation settings
    segmentation: SegmentationConfig = .{},
    /// Extract entities
    extract_entities: bool = true,
    /// Analyze layout
    analyze_layout: bool = true,
    /// Maximum document size
    max_document_size: usize = 10 * 1024 * 1024, // 10MB
};

/// Complete document processing pipeline
pub const DocumentPipeline = struct {
    allocator: std.mem.Allocator,
    config: PipelineConfig,
    segmenter: TextSegmenter,
    entity_extractor: EntityExtractor,
    layout_analyzer: LayoutAnalyzer,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, config: PipelineConfig) Self {
        return .{
            .allocator = allocator,
            .config = config,
            .segmenter = TextSegmenter.init(allocator, config.segmentation),
            .entity_extractor = EntityExtractor.init(allocator),
            .layout_analyzer = LayoutAnalyzer.init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        _ = self;
    }

    /// Parse a document from content with optional filename for format detection
    pub fn parse(self: *Self, filename: ?[]const u8, content: []const u8) !Document {
        if (content.len > self.config.max_document_size) {
            return error.DocumentTooLarge;
        }

        // Detect format
        const format = if (filename) |f|
            DocumentFormat.fromExtension(std.fs.path.extension(f))
        else
            .unknown;

        // Copy content
        const raw_content = try self.allocator.dupe(u8, content);
        errdefer self.allocator.free(raw_content);

        // Analyze layout
        const elements = if (self.config.analyze_layout and format == .markdown)
            try self.layout_analyzer.analyzeMarkdown(content)
        else
            try self.allocator.alloc(DocumentElement, 0);
        errdefer self.allocator.free(elements);

        // Segment text
        const segments = try self.segmenter.segmentChunks(content);
        errdefer self.allocator.free(segments);

        // Extract entities
        const entities = if (self.config.extract_entities)
            try self.entity_extractor.extractAll(content)
        else
            try self.allocator.alloc(NamedEntity, 0);
        errdefer self.allocator.free(entities);

        // Compute stats
        const metadata = self.layout_analyzer.computeStats(content);

        return Document{
            .allocator = self.allocator,
            .raw_content = raw_content,
            .format = format,
            .elements = elements,
            .metadata = metadata,
            .segments = segments,
            .entities = entities,
        };
    }

    /// Get document summary (first N segments)
    pub fn getSummarySegments(self: *Self, doc: *const Document, max_segments: usize) []const TextSegment {
        _ = self;
        return doc.segments[0..@min(max_segments, doc.segments.len)];
    }

    /// Find segments containing a query string
    pub fn searchSegments(
        self: *Self,
        doc: *const Document,
        query: []const u8,
    ) ![]const TextSegment {
        var results = std.ArrayListUnmanaged(*const TextSegment).empty;
        defer results.deinit(self.allocator);

        for (doc.segments) |*segment| {
            if (std.mem.indexOf(u8, segment.content, query) != null) {
                try results.append(self.allocator, segment);
            }
        }

        // Return as slice of segments
        var segment_slice = try self.allocator.alloc(TextSegment, results.items.len);
        for (results.items, 0..) |seg_ptr, i| {
            segment_slice[i] = seg_ptr.*;
        }
        return segment_slice;
    }

    /// Get all headings from document
    pub fn getHeadings(self: *Self, doc: *const Document) ![]const DocumentElement {
        var headings = std.ArrayListUnmanaged(DocumentElement).empty;
        defer headings.deinit(self.allocator);

        for (doc.elements) |elem| {
            switch (elem.element_type) {
                .heading1, .heading2, .heading3, .heading4, .heading5, .heading6 => {
                    try headings.append(self.allocator, elem);
                },
                else => {},
            }
        }

        return headings.toOwnedSlice(self.allocator);
    }

    /// Get table of contents (headings with hierarchy)
    pub fn getTableOfContents(self: *Self, doc: *const Document) ![]DocumentElement {
        return self.getHeadings(doc);
    }

    pub const Error = error{
        DocumentTooLarge,
        InvalidFormat,
        ParseError,
        OutOfMemory,
    };
};


test {
    std.testing.refAllDecls(@This());
}
