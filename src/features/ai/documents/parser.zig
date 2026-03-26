//! Document Parsing Components
//!
//! Text segmentation, entity extraction, and layout analysis for documents.

const std = @import("std");
const types = @import("mod.zig");

pub const DocumentFormat = types.DocumentFormat;
pub const ElementType = types.ElementType;
pub const DocumentElement = types.DocumentElement;
pub const Document = types.Document;
pub const TextSegment = types.TextSegment;
pub const EntityType = types.EntityType;
pub const NamedEntity = types.NamedEntity;

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

    pub fn isUrlTerminator(c: u8) bool {
        return c == ' ' or c == '\n' or c == '\t' or c == '"' or c == '\'' or c == ')' or c == '>' or c == ']';
    }

    pub fn isEmailChar(c: u8) bool {
        return std.ascii.isAlphanumeric(c) or c == '.' or c == '_' or c == '-' or c == '+';
    }

    pub fn isPathChar(c: u8) bool {
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


test {
    std.testing.refAllDecls(@This());
}
