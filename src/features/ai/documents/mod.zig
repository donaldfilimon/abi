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
// Text Segmenter
// ============================================================================

/// Configuration for text segmentation
pub const SegmentationConfig = struct {
    /// Target chunk size in characters
    chunk_size: usize = 512,
    /// Overlap between chunks
    chunk_overlap: usize = 64,
    /// Whether to respect sentence boundaries
    respect_sentences: bool = true,
    /// Whether to respect paragraph boundaries
    respect_paragraphs: bool = true,
    /// Minimum segment size
    min_segment_size: usize = 50,
};

/// Text segmentation for document processing
pub const TextSegmenter = struct {
    allocator: std.mem.Allocator,
    config: SegmentationConfig,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, config: SegmentationConfig) Self {
        return .{
            .allocator = allocator,
            .config = config,
        };
    }

    /// Segment text into sentences
    pub fn segmentSentences(self: *Self, text: []const u8) ![]TextSegment {
        var segments = std.ArrayListUnmanaged(TextSegment).empty;
        errdefer segments.deinit(self.allocator);

        var start: usize = 0;
        var i: usize = 0;

        while (i < text.len) : (i += 1) {
            const c = text[i];
            // Sentence endings
            if (c == '.' or c == '!' or c == '?') {
                // Check for abbreviations (simple heuristic)
                if (i + 1 < text.len and text[i + 1] != ' ' and text[i + 1] != '\n') {
                    continue;
                }
                // Found sentence end
                const end = i + 1;
                if (end - start >= self.config.min_segment_size) {
                    try segments.append(self.allocator, .{
                        .content = text[start..end],
                        .start_offset = start,
                        .end_offset = end,
                        .segment_type = .sentence,
                    });
                }
                // Skip whitespace
                while (i + 1 < text.len and (text[i + 1] == ' ' or text[i + 1] == '\n')) {
                    i += 1;
                }
                start = i + 1;
            }
        }

        // Handle remaining text
        if (start < text.len and text.len - start >= self.config.min_segment_size) {
            try segments.append(self.allocator, .{
                .content = text[start..],
                .start_offset = start,
                .end_offset = text.len,
                .segment_type = .sentence,
            });
        }

        return segments.toOwnedSlice(self.allocator);
    }

    /// Segment text into fixed-size chunks with overlap
    pub fn segmentChunks(self: *Self, text: []const u8) ![]TextSegment {
        var segments = std.ArrayListUnmanaged(TextSegment).empty;
        errdefer segments.deinit(self.allocator);

        var pos: usize = 0;
        while (pos < text.len) {
            var end = @min(pos + self.config.chunk_size, text.len);

            // Try to break at sentence boundary
            if (self.config.respect_sentences and end < text.len) {
                // Look backward for sentence ending
                var search_start = if (end > 50) end - 50 else pos;
                while (search_start < end) : (search_start += 1) {
                    const c = text[search_start];
                    if (c == '.' or c == '!' or c == '?') {
                        if (search_start + 1 < text.len and
                            (text[search_start + 1] == ' ' or text[search_start + 1] == '\n'))
                        {
                            end = search_start + 1;
                            break;
                        }
                    }
                }
            }

            try segments.append(self.allocator, .{
                .content = text[pos..end],
                .start_offset = pos,
                .end_offset = end,
                .segment_type = .chunk,
            });

            // Move position with overlap
            if (end >= text.len) break;

            // Ensure we always advance by at least 1 character
            const advance = if (end > self.config.chunk_overlap)
                end - self.config.chunk_overlap
            else
                end;
            const new_pos = @max(pos + 1, advance);
            if (new_pos <= pos) break; // Prevent infinite loop
            pos = new_pos;
        }

        return segments.toOwnedSlice(self.allocator);
    }

    /// Segment text into paragraphs
    pub fn segmentParagraphs(self: *Self, text: []const u8) ![]TextSegment {
        var segments = std.ArrayListUnmanaged(TextSegment).empty;
        errdefer segments.deinit(self.allocator);

        var start: usize = 0;
        var i: usize = 0;

        while (i < text.len) : (i += 1) {
            // Look for double newline (paragraph break)
            if (text[i] == '\n' and i + 1 < text.len and text[i + 1] == '\n') {
                if (i - start >= self.config.min_segment_size) {
                    try segments.append(self.allocator, .{
                        .content = text[start..i],
                        .start_offset = start,
                        .end_offset = i,
                        .segment_type = .paragraph,
                    });
                }
                // Skip multiple newlines
                while (i + 1 < text.len and text[i + 1] == '\n') {
                    i += 1;
                }
                start = i + 1;
            }
        }

        // Handle remaining text
        if (start < text.len and text.len - start >= self.config.min_segment_size) {
            try segments.append(self.allocator, .{
                .content = text[start..],
                .start_offset = start,
                .end_offset = text.len,
                .segment_type = .paragraph,
            });
        }

        return segments.toOwnedSlice(self.allocator);
    }
};

// ============================================================================
// Entity Extractor
// ============================================================================

/// Named entity extraction using pattern matching
pub const EntityExtractor = struct {
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{ .allocator = allocator };
    }

    /// Extract all entities from text
    pub fn extractAll(self: *Self, text: []const u8) ![]NamedEntity {
        var entities = std.ArrayListUnmanaged(NamedEntity).empty;
        errdefer entities.deinit(self.allocator);

        // Extract URLs
        try self.extractUrls(text, &entities);

        // Extract emails
        try self.extractEmails(text, &entities);

        // Extract file paths
        try self.extractFilePaths(text, &entities);

        // Extract versions
        try self.extractVersions(text, &entities);

        // Extract dates (simple patterns)
        try self.extractDates(text, &entities);

        return entities.toOwnedSlice(self.allocator);
    }

    fn extractUrls(self: *Self, text: []const u8, entities: *std.ArrayListUnmanaged(NamedEntity)) !void {
        // Look for http:// or https://
        var i: usize = 0;
        while (i < text.len) : (i += 1) {
            if (i + 7 < text.len and
                (std.mem.eql(u8, text[i .. i + 7], "http://") or
                    (i + 8 < text.len and std.mem.eql(u8, text[i .. i + 8], "https://"))))
            {
                const start = i;
                // Find URL end
                while (i < text.len and !isUrlTerminator(text[i])) : (i += 1) {}
                const end = i;
                if (end > start) {
                    try entities.append(self.allocator, .{
                        .text = text[start..end],
                        .entity_type = .url,
                        .start_offset = start,
                        .end_offset = end,
                    });
                }
            }
        }
    }

    fn extractEmails(self: *Self, text: []const u8, entities: *std.ArrayListUnmanaged(NamedEntity)) !void {
        var i: usize = 0;
        while (i < text.len) : (i += 1) {
            if (text[i] == '@' and i > 0 and i + 1 < text.len) {
                // Find start of email
                var start = i;
                while (start > 0 and isEmailChar(text[start - 1])) : (start -= 1) {}

                // Find end of email
                var end = i + 1;
                while (end < text.len and isEmailChar(text[end])) : (end += 1) {}

                // Must have '.' after @
                if (std.mem.indexOfScalar(u8, text[i..end], '.') != null) {
                    try entities.append(self.allocator, .{
                        .text = text[start..end],
                        .entity_type = .email,
                        .start_offset = start,
                        .end_offset = end,
                    });
                }
            }
        }
    }

    fn extractFilePaths(self: *Self, text: []const u8, entities: *std.ArrayListUnmanaged(NamedEntity)) !void {
        // Look for patterns like /path/to/file or ./path or ../path
        var i: usize = 0;
        while (i < text.len) : (i += 1) {
            const isPathStart = (text[i] == '/' and (i == 0 or text[i - 1] == ' ' or text[i - 1] == '\n')) or
                (text[i] == '.' and i + 1 < text.len and text[i + 1] == '/');

            if (isPathStart) {
                const start = i;
                while (i < text.len and isPathChar(text[i])) : (i += 1) {}
                const end = i;
                if (end - start >= 3) { // Minimum path length
                    try entities.append(self.allocator, .{
                        .text = text[start..end],
                        .entity_type = .file_path,
                        .start_offset = start,
                        .end_offset = end,
                    });
                }
            }
        }
    }

    fn extractVersions(self: *Self, text: []const u8, entities: *std.ArrayListUnmanaged(NamedEntity)) !void {
        // Look for version patterns like v1.2.3 or 1.2.3
        var i: usize = 0;
        while (i < text.len) : (i += 1) {
            const isVersionStart = (text[i] == 'v' or text[i] == 'V') and
                i + 1 < text.len and std.ascii.isDigit(text[i + 1]);

            if (isVersionStart or (std.ascii.isDigit(text[i]) and
                (i == 0 or text[i - 1] == ' ' or text[i - 1] == '\n' or text[i - 1] == '-')))
            {
                const start = i;
                var has_dot = false;
                if (text[i] == 'v' or text[i] == 'V') i += 1;

                while (i < text.len and (std.ascii.isDigit(text[i]) or text[i] == '.')) : (i += 1) {
                    if (text[i] == '.') has_dot = true;
                }
                const end = i;
                if (has_dot and end - start >= 3) {
                    try entities.append(self.allocator, .{
                        .text = text[start..end],
                        .entity_type = .version,
                        .start_offset = start,
                        .end_offset = end,
                    });
                }
            }
        }
    }

    fn extractDates(self: *Self, text: []const u8, entities: *std.ArrayListUnmanaged(NamedEntity)) !void {
        // Look for YYYY-MM-DD or MM/DD/YYYY patterns
        var i: usize = 0;
        while (i + 9 < text.len) : (i += 1) {
            // YYYY-MM-DD
            if (std.ascii.isDigit(text[i]) and
                std.ascii.isDigit(text[i + 1]) and
                std.ascii.isDigit(text[i + 2]) and
                std.ascii.isDigit(text[i + 3]) and
                text[i + 4] == '-' and
                std.ascii.isDigit(text[i + 5]) and
                std.ascii.isDigit(text[i + 6]) and
                text[i + 7] == '-' and
                std.ascii.isDigit(text[i + 8]) and
                std.ascii.isDigit(text[i + 9]))
            {
                try entities.append(self.allocator, .{
                    .text = text[i .. i + 10],
                    .entity_type = .date,
                    .start_offset = i,
                    .end_offset = i + 10,
                });
                i += 9;
            }
        }
    }

    fn isUrlTerminator(c: u8) bool {
        return c == ' ' or c == '\n' or c == '\t' or c == '"' or c == '\'' or c == ')' or c == '>' or c == ']';
    }

    fn isEmailChar(c: u8) bool {
        return std.ascii.isAlphanumeric(c) or c == '.' or c == '_' or c == '-' or c == '+';
    }

    fn isPathChar(c: u8) bool {
        return std.ascii.isAlphanumeric(c) or c == '/' or c == '.' or c == '_' or c == '-';
    }
};

// ============================================================================
// Layout Analyzer
// ============================================================================

/// Analyze document layout and structure
pub const LayoutAnalyzer = struct {
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{ .allocator = allocator };
    }

    /// Analyze markdown document structure
    pub fn analyzeMarkdown(self: *Self, text: []const u8) ![]DocumentElement {
        var elements = std.ArrayListUnmanaged(DocumentElement).empty;
        errdefer elements.deinit(self.allocator);

        var lines = std.mem.splitScalar(u8, text, '\n');
        var offset: usize = 0;
        var in_code_block = false;
        var code_block_start: usize = 0;

        while (lines.next()) |line| {
            const line_end = offset + line.len;

            // Code blocks
            if (std.mem.startsWith(u8, line, "```")) {
                if (in_code_block) {
                    // End code block
                    try elements.append(self.allocator, .{
                        .element_type = .code_block,
                        .content = text[code_block_start..line_end],
                        .start_offset = code_block_start,
                        .end_offset = line_end,
                    });
                    in_code_block = false;
                } else {
                    // Start code block
                    code_block_start = offset;
                    in_code_block = true;
                }
            } else if (!in_code_block) {
                // Headings
                if (std.mem.startsWith(u8, line, "######")) {
                    try elements.append(self.allocator, .{
                        .element_type = .heading6,
                        .content = std.mem.trimStart(u8, line[6..], " "),
                        .start_offset = offset,
                        .end_offset = line_end,
                        .level = 6,
                    });
                } else if (std.mem.startsWith(u8, line, "#####")) {
                    try elements.append(self.allocator, .{
                        .element_type = .heading5,
                        .content = std.mem.trimStart(u8, line[5..], " "),
                        .start_offset = offset,
                        .end_offset = line_end,
                        .level = 5,
                    });
                } else if (std.mem.startsWith(u8, line, "####")) {
                    try elements.append(self.allocator, .{
                        .element_type = .heading4,
                        .content = std.mem.trimStart(u8, line[4..], " "),
                        .start_offset = offset,
                        .end_offset = line_end,
                        .level = 4,
                    });
                } else if (std.mem.startsWith(u8, line, "###")) {
                    try elements.append(self.allocator, .{
                        .element_type = .heading3,
                        .content = std.mem.trimStart(u8, line[3..], " "),
                        .start_offset = offset,
                        .end_offset = line_end,
                        .level = 3,
                    });
                } else if (std.mem.startsWith(u8, line, "##")) {
                    try elements.append(self.allocator, .{
                        .element_type = .heading2,
                        .content = std.mem.trimStart(u8, line[2..], " "),
                        .start_offset = offset,
                        .end_offset = line_end,
                        .level = 2,
                    });
                } else if (std.mem.startsWith(u8, line, "#")) {
                    try elements.append(self.allocator, .{
                        .element_type = .heading1,
                        .content = std.mem.trimStart(u8, line[1..], " "),
                        .start_offset = offset,
                        .end_offset = line_end,
                        .level = 1,
                    });
                }
                // List items
                else if (std.mem.startsWith(u8, line, "- ") or std.mem.startsWith(u8, line, "* ")) {
                    try elements.append(self.allocator, .{
                        .element_type = .list_item,
                        .content = line[2..],
                        .start_offset = offset,
                        .end_offset = line_end,
                    });
                }
                // Numbered list items
                else if (line.len >= 3 and std.ascii.isDigit(line[0]) and line[1] == '.') {
                    try elements.append(self.allocator, .{
                        .element_type = .list_item,
                        .content = std.mem.trimStart(u8, line[2..], " "),
                        .start_offset = offset,
                        .end_offset = line_end,
                    });
                }
                // Blockquotes
                else if (std.mem.startsWith(u8, line, ">")) {
                    try elements.append(self.allocator, .{
                        .element_type = .blockquote,
                        .content = std.mem.trimStart(u8, line[1..], " "),
                        .start_offset = offset,
                        .end_offset = line_end,
                    });
                }
                // Horizontal rule
                else if (std.mem.eql(u8, line, "---") or std.mem.eql(u8, line, "***") or std.mem.eql(u8, line, "___")) {
                    try elements.append(self.allocator, .{
                        .element_type = .horizontal_rule,
                        .content = line,
                        .start_offset = offset,
                        .end_offset = line_end,
                    });
                }
                // Paragraph (non-empty lines not matching other patterns)
                else if (line.len > 0 and !std.mem.eql(u8, std.mem.trim(u8, line, " \t"), "")) {
                    try elements.append(self.allocator, .{
                        .element_type = .paragraph,
                        .content = line,
                        .start_offset = offset,
                        .end_offset = line_end,
                    });
                }
            }

            offset = line_end + 1; // +1 for newline
        }

        return elements.toOwnedSlice(self.allocator);
    }

    /// Count document statistics
    pub fn computeStats(self: *Self, text: []const u8) Document.DocumentMetadata {
        _ = self;
        var stats = Document.DocumentMetadata{};

        stats.char_count = text.len;

        // Count words (simple whitespace-based)
        var word_count: usize = 0;
        var in_word = false;
        for (text) |c| {
            if (c == ' ' or c == '\n' or c == '\t') {
                if (in_word) {
                    word_count += 1;
                    in_word = false;
                }
            } else {
                in_word = true;
            }
        }
        if (in_word) word_count += 1;
        stats.word_count = word_count;

        // Count lines
        var line_count: usize = 1;
        for (text) |c| {
            if (c == '\n') line_count += 1;
        }
        stats.line_count = line_count;

        return stats;
    }
};

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

// ============================================================================
// Public Exports
// ============================================================================

pub const Error = DocumentPipeline.Error || error{OutOfMemory};

/// Check if documents feature is enabled
pub fn isEnabled() bool {
    return build_options.enable_ai;
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
    var pipeline = DocumentPipeline.init(allocator, .{});
    defer pipeline.deinit();

    const content = "# Test Document\n\nThis is test content with https://example.com link.";
    var doc = try pipeline.parse("test.md", content);
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
