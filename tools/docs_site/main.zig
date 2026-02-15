//! Documentation Site Generator
//!
//! Static site generator for the ABI framework documentation.
//! Processes markdown content files into a styled HTML website with
//! navigation, table of contents, theme toggle, and full-text search.
//!
//! Run with: `zig build docs-site`

const std = @import("std");

// =============================================================================
// Configuration Types
// =============================================================================

const BuildConfig = struct {
    source_dir: []const u8,
    out_dir: []const u8,
    manifest_path: []const u8,
};

const SiteConfig = struct {
    title: []const u8,
    tagline: []const u8,
    base_url: []const u8,
};

const FooterConfig = struct {
    text: []const u8,
};

const PageConfig = struct {
    title: []const u8,
    slug: []const u8,
    source: []const u8,
    section: []const u8,
};

const Manifest = struct {
    site: SiteConfig,
    pages: []PageConfig,
    footer: ?FooterConfig = null,
};

const ArgsError = error{
    InvalidArgs,
    MissingValue,
    ShowHelp,
};

// =============================================================================
// Front Matter
// =============================================================================

const FrontMatter = struct {
    title: []const u8 = "",
    description: []const u8 = "",
    section: []const u8 = "",
    order: u32 = 999,
};

fn parseFrontMatter(content: []const u8) struct { front_matter: FrontMatter, body: []const u8 } {
    if (!std.mem.startsWith(u8, content, "---\n") and !std.mem.startsWith(u8, content, "---\r\n")) {
        return .{ .front_matter = .{}, .body = content };
    }

    const start = if (std.mem.startsWith(u8, content, "---\r\n")) @as(usize, 5) else @as(usize, 4);
    const end_marker = std.mem.indexOf(u8, content[start..], "\n---") orelse return .{ .front_matter = .{}, .body = content };
    const yaml = content[start .. start + end_marker];
    const body_start = start + end_marker + 4; // skip "\n---"
    const body = if (body_start < content.len and content[body_start] == '\n')
        content[body_start + 1 ..]
    else if (body_start < content.len)
        content[body_start..]
    else
        "";

    var front_matter = FrontMatter{};
    var lines = std.mem.splitScalar(u8, yaml, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (trimmed.len == 0) continue;

        if (std.mem.indexOfScalar(u8, trimmed, ':')) |colon_pos| {
            const key = std.mem.trim(u8, trimmed[0..colon_pos], " \t\"");
            const value = std.mem.trim(u8, trimmed[colon_pos + 1 ..], " \t\"");

            if (std.mem.eql(u8, key, "title")) front_matter.title = value;
            if (std.mem.eql(u8, key, "description")) front_matter.description = value;
            if (std.mem.eql(u8, key, "section")) front_matter.section = value;
            if (std.mem.eql(u8, key, "order")) front_matter.order = std.fmt.parseInt(u32, value, 10) catch 999;
        }
    }

    return .{ .front_matter = front_matter, .body = body };
}

// =============================================================================
// TOC Entry
// =============================================================================

const TocEntry = struct {
    level: u8,
    text: []const u8,
    slug: []const u8,
};

// =============================================================================
// Search Entry
// =============================================================================

const SearchEntry = struct {
    title: []const u8,
    slug: []const u8,
    section: []const u8,
    content: []const u8,
};

// =============================================================================
// Markdown Parser
// =============================================================================

const MarkdownParser = struct {
    allocator: std.mem.Allocator,
    source: []const u8,
    pos: usize,
    output: std.ArrayListUnmanaged(u8),
    toc: std.ArrayListUnmanaged(TocEntry),
    text_content: std.ArrayListUnmanaged(u8),
    in_code_block: bool,
    code_lang: []const u8,
    in_list: bool,
    in_table: bool,
    in_blockquote: bool,

    fn init(allocator: std.mem.Allocator, source: []const u8) MarkdownParser {
        return .{
            .allocator = allocator,
            .source = source,
            .pos = 0,
            .output = .empty,
            .toc = .empty,
            .text_content = .empty,
            .in_code_block = false,
            .code_lang = "",
            .in_list = false,
            .in_table = false,
            .in_blockquote = false,
        };
    }

    fn deinit(self: *MarkdownParser) void {
        self.output.deinit(self.allocator);
        for (self.toc.items) |entry| self.allocator.free(entry.slug);
        self.toc.deinit(self.allocator);
        self.text_content.deinit(self.allocator);
    }

    const ParseResult = struct {
        html: []const u8,
        toc: []const TocEntry,
        text: []const u8,
    };

    fn parse(self: *MarkdownParser) !ParseResult {
        while (self.pos < self.source.len) {
            try self.parseLine();
        }

        // Close any open blocks
        if (self.in_code_block) try self.emit("</code></pre>\n");
        if (self.in_list) try self.emit("</ul>\n");
        if (self.in_table) try self.emit("</tbody></table>\n");
        if (self.in_blockquote) try self.emit("</blockquote>\n");

        return .{
            .html = try self.output.toOwnedSlice(self.allocator),
            .toc = self.toc.items,
            .text = try self.text_content.toOwnedSlice(self.allocator),
        };
    }

    fn parseLine(self: *MarkdownParser) !void {
        const line_end = std.mem.indexOfScalar(u8, self.source[self.pos..], '\n') orelse self.source.len - self.pos;
        const raw_line = self.source[self.pos .. self.pos + line_end];
        const line = std.mem.trimEnd(u8, raw_line, "\r");
        self.pos += line_end + 1;

        // Collect text for search index (skip code blocks)
        if (!self.in_code_block and !std.mem.startsWith(u8, line, "```")) {
            const stripped = std.mem.trim(u8, line, "#>|-* \t");
            if (stripped.len > 0) {
                try self.text_content.appendSlice(self.allocator, stripped);
                try self.text_content.append(self.allocator, ' ');
            }
        }

        // Code block toggle
        if (std.mem.startsWith(u8, line, "```")) {
            if (self.in_code_block) {
                try self.emit("</code></pre>\n");
                self.in_code_block = false;
            } else {
                self.in_code_block = true;
                self.code_lang = if (line.len > 3) std.mem.trim(u8, line[3..], " \t") else "";
                try self.emit("<pre class=\"code-block");
                if (self.code_lang.len > 0) {
                    try self.emit(" language-");
                    try self.emit(self.code_lang);
                }
                try self.emit("\"><code>");
            }
            return;
        }

        if (self.in_code_block) {
            try self.emitEscaped(line);
            try self.emit("\n");
            return;
        }

        // Close open list if not a list item
        if (self.in_list and !(line.len > 1 and (line[0] == '-' or line[0] == '*' or line[0] == '+') and line[1] == ' ') and !isOrderedListItem(line)) {
            try self.emit("</ul>\n");
            self.in_list = false;
        }

        // Close table if not a table row
        if (self.in_table and (line.len == 0 or line[0] != '|')) {
            try self.emit("</tbody></table>\n");
            self.in_table = false;
        }

        // Close blockquote if not a blockquote line
        if (self.in_blockquote and (line.len == 0 or line[0] != '>')) {
            try self.emit("</blockquote>\n");
            self.in_blockquote = false;
        }

        // Empty line
        if (line.len == 0) return;

        // Headers
        if (line[0] == '#') {
            try self.parseHeader(line);
            return;
        }

        // Horizontal rule
        if (line.len >= 3 and (std.mem.eql(u8, line, "---") or std.mem.eql(u8, line, "***") or std.mem.eql(u8, line, "___"))) {
            try self.emit("<hr>\n");
            return;
        }

        // Blockquote
        if (line[0] == '>') {
            if (!self.in_blockquote) {
                try self.emit("<blockquote>\n");
                self.in_blockquote = true;
            }
            const content = if (line.len > 2 and line[1] == ' ') line[2..] else if (line.len > 1) line[1..] else "";
            try self.emit("<p>");
            try self.parseInline(content);
            try self.emit("</p>\n");
            return;
        }

        // Unordered list items
        if (line.len > 1 and (line[0] == '-' or line[0] == '*' or line[0] == '+') and line[1] == ' ') {
            if (!self.in_list) {
                try self.emit("<ul>\n");
                self.in_list = true;
            }
            try self.emit("<li>");
            try self.parseInline(line[2..]);
            try self.emit("</li>\n");
            return;
        }

        // Ordered list items
        if (isOrderedListItem(line)) {
            if (!self.in_list) {
                try self.emit("<ul>\n");
                self.in_list = true;
            }
            const content = getOrderedListContent(line);
            try self.emit("<li>");
            try self.parseInline(content);
            try self.emit("</li>\n");
            return;
        }

        // Table row
        if (line[0] == '|') {
            try self.parseTableRow(line);
            return;
        }

        // Paragraph
        try self.emit("<p>");
        try self.parseInline(line);
        try self.emit("</p>\n");
    }

    fn parseHeader(self: *MarkdownParser, line: []const u8) !void {
        var level: u8 = 0;
        while (level < line.len and line[level] == '#') : (level += 1) {}
        if (level > 6) level = 6;

        const content = if (level < line.len and line[level] == ' ') line[level + 1 ..] else line[level..];
        const slug = try generateSlug(self.allocator, content);

        try self.toc.append(self.allocator, .{ .level = level, .text = content, .slug = slug });

        try self.emit("<h");
        try self.output.append(self.allocator, '0' + level);
        try self.emit(" id=\"");
        try self.emit(slug);
        try self.emit("\">");
        try self.parseInline(content);
        try self.emit("<a class=\"anchor\" href=\"#");
        try self.emit(slug);
        try self.emit("\">#</a>");
        try self.emit("</h");
        try self.output.append(self.allocator, '0' + level);
        try self.emit(">\n");
    }

    fn parseInline(self: *MarkdownParser, text: []const u8) !void {
        var i: usize = 0;
        while (i < text.len) {
            const c = text[i];

            // Image ![alt](url)
            if (c == '!' and i + 1 < text.len and text[i + 1] == '[') {
                if (std.mem.indexOf(u8, text[i + 2 ..], "](")) |bracket_end_rel| {
                    const bracket_end = i + 2 + bracket_end_rel;
                    if (std.mem.indexOfScalar(u8, text[bracket_end + 2 ..], ')')) |paren_end_rel| {
                        const paren_end = bracket_end + 2 + paren_end_rel;
                        try self.emit("<img src=\"");
                        try self.emitEscaped(text[bracket_end + 2 .. paren_end]);
                        try self.emit("\" alt=\"");
                        try self.emitEscaped(text[i + 2 .. bracket_end]);
                        try self.emit("\">");
                        i = paren_end + 1;
                        continue;
                    }
                }
            }

            // Bold **text**
            if (i + 1 < text.len and c == '*' and text[i + 1] == '*') {
                if (std.mem.indexOf(u8, text[i + 2 ..], "**")) |end| {
                    try self.emit("<strong>");
                    try self.parseInline(text[i + 2 .. i + 2 + end]);
                    try self.emit("</strong>");
                    i += 4 + end;
                    continue;
                }
            }

            // Italic *text*
            if (c == '*' and (i + 1 >= text.len or text[i + 1] != '*')) {
                if (std.mem.indexOfScalar(u8, text[i + 1 ..], '*')) |end| {
                    try self.emit("<em>");
                    try self.parseInline(text[i + 1 .. i + 1 + end]);
                    try self.emit("</em>");
                    i += 2 + end;
                    continue;
                }
            }

            // Inline code `text`
            if (c == '`') {
                if (std.mem.indexOfScalar(u8, text[i + 1 ..], '`')) |end| {
                    try self.emit("<code>");
                    try self.emitEscaped(text[i + 1 .. i + 1 + end]);
                    try self.emit("</code>");
                    i += 2 + end;
                    continue;
                }
            }

            // Link [text](url)
            if (c == '[') {
                if (std.mem.indexOf(u8, text[i..], "](")) |bracket_end| {
                    if (std.mem.indexOfScalar(u8, text[i + bracket_end + 2 ..], ')')) |paren_end| {
                        const link_text = text[i + 1 .. i + bracket_end];
                        const link_url = text[i + bracket_end + 2 .. i + bracket_end + 2 + paren_end];
                        try self.emit("<a href=\"");
                        try self.emit(link_url);
                        try self.emit("\">");
                        try self.parseInline(link_text);
                        try self.emit("</a>");
                        i += bracket_end + 3 + paren_end;
                        continue;
                    }
                }
            }

            try self.emitEscapedChar(c);
            i += 1;
        }
    }

    fn parseTableRow(self: *MarkdownParser, line: []const u8) !void {
        // Skip separator rows (|---|---|)
        if (isSeparatorRow(line)) return;

        if (!self.in_table) {
            try self.emit("<table>\n<thead>\n");
            self.in_table = true;
            // First data row is the header
            try self.emit("<tr>");
            var iter = std.mem.splitScalar(u8, line, '|');
            while (iter.next()) |cell| {
                const trimmed = std.mem.trim(u8, cell, " \t");
                if (trimmed.len > 0) {
                    try self.emit("<th>");
                    try self.parseInline(trimmed);
                    try self.emit("</th>");
                }
            }
            try self.emit("</tr>\n</thead>\n<tbody>\n");

            // Look ahead and skip separator row
            return;
        }

        try self.emit("<tr>");
        var iter = std.mem.splitScalar(u8, line, '|');
        while (iter.next()) |cell| {
            const trimmed = std.mem.trim(u8, cell, " \t");
            if (trimmed.len > 0) {
                try self.emit("<td>");
                try self.parseInline(trimmed);
                try self.emit("</td>");
            }
        }
        try self.emit("</tr>\n");
    }

    fn emit(self: *MarkdownParser, text: []const u8) !void {
        try self.output.appendSlice(self.allocator, text);
    }

    fn emitEscaped(self: *MarkdownParser, text: []const u8) !void {
        for (text) |c| try self.emitEscapedChar(c);
    }

    fn emitEscapedChar(self: *MarkdownParser, c: u8) !void {
        switch (c) {
            '<' => try self.emit("&lt;"),
            '>' => try self.emit("&gt;"),
            '&' => try self.emit("&amp;"),
            '"' => try self.emit("&quot;"),
            else => try self.output.append(self.allocator, c),
        }
    }

    fn isSeparatorRow(line: []const u8) bool {
        for (line) |c| {
            if (c != '|' and c != '-' and c != ':' and c != ' ' and c != '\t') return false;
        }
        return std.mem.indexOf(u8, line, "---") != null;
    }

    fn isOrderedListItem(line: []const u8) bool {
        var i: usize = 0;
        while (i < line.len and std.ascii.isDigit(line[i])) : (i += 1) {}
        return i > 0 and i + 1 < line.len and line[i] == '.' and line[i + 1] == ' ';
    }

    fn getOrderedListContent(line: []const u8) []const u8 {
        var i: usize = 0;
        while (i < line.len and std.ascii.isDigit(line[i])) : (i += 1) {}
        if (i + 2 <= line.len) return line[i + 2 ..]; // skip ". "
        return "";
    }
};

fn generateSlug(allocator: std.mem.Allocator, text: []const u8) ![]const u8 {
    var slug = std.ArrayListUnmanaged(u8).empty;
    errdefer slug.deinit(allocator);
    for (text) |c| {
        if (std.ascii.isAlphanumeric(c)) {
            try slug.append(allocator, std.ascii.toLower(c));
        } else if (c == ' ' or c == '-' or c == '_') {
            if (slug.items.len > 0 and slug.items[slug.items.len - 1] != '-') {
                try slug.append(allocator, '-');
            }
        }
    }
    while (slug.items.len > 0 and slug.items[slug.items.len - 1] == '-') {
        _ = slug.pop();
    }
    return try slug.toOwnedSlice(allocator);
}

// =============================================================================
// Entry Point
// =============================================================================

pub fn main(init: std.process.Init.Minimal) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const status = gpa.deinit();
        if (status == .leak) {
            std.debug.print("Memory leak detected\n", .{});
        }
    }
    const allocator = gpa.allocator();

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const args = try init.args.toSlice(arena.allocator());

    const config = parseArgs(args) catch |err| {
        if (err == error.ShowHelp) return;
        std.log.err("Invalid arguments: {t}", .{err});
        printUsage();
        return err;
    };

    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = init.environ });
    defer io_backend.deinit();
    const io = io_backend.io();

    try buildSite(allocator, io, config);
}

fn parseArgs(args: []const [:0]const u8) ArgsError!BuildConfig {
    var config = BuildConfig{
        .source_dir = "docs",
        .out_dir = "zig-out/docs",
        .manifest_path = "docs/site.json",
    };

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--source")) {
            i += 1;
            if (i >= args.len) return error.MissingValue;
            config.source_dir = args[i];
        } else if (std.mem.eql(u8, arg, "--out")) {
            i += 1;
            if (i >= args.len) return error.MissingValue;
            config.out_dir = args[i];
        } else if (std.mem.eql(u8, arg, "--manifest")) {
            i += 1;
            if (i >= args.len) return error.MissingValue;
            config.manifest_path = args[i];
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printUsage();
            return error.ShowHelp;
        } else {
            return error.InvalidArgs;
        }
    }

    return config;
}

fn printUsage() void {
    std.debug.print(
        "Usage: docs-site [--source <dir>] [--out <dir>] [--manifest <path>]\n" ++
            "Defaults:\n" ++
            "  --source docs\n" ++
            "  --out zig-out/docs\n" ++
            "  --manifest docs/site.json\n",
        .{},
    );
}

// =============================================================================
// Site Builder
// =============================================================================

fn buildSite(allocator: std.mem.Allocator, io: std.Io, config: BuildConfig) !void {
    const cwd = std.Io.Dir.cwd();

    const manifest_data = try cwd.readFileAlloc(
        io,
        config.manifest_path,
        allocator,
        .limited(5 * 1024 * 1024),
    );
    defer allocator.free(manifest_data);

    var parsed = try std.json.parseFromSlice(
        Manifest,
        allocator,
        manifest_data,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();
    const manifest = parsed.value;

    std.debug.print("=== ABI Documentation Site Generator ===\n\n", .{});

    try cwd.createDirPath(io, config.out_dir);
    try copyAssets(allocator, io, config.source_dir, config.out_dir);

    var search_entries = std.ArrayListUnmanaged(SearchEntry).empty;
    defer {
        for (search_entries.items) |entry| {
            allocator.free(entry.content);
        }
        search_entries.deinit(allocator);
    }

    var page_count: usize = 0;
    for (manifest.pages) |page| {
        writePage(allocator, io, manifest, config, page, &search_entries) catch |err| {
            std.debug.print("  [SKIP] {s}: {}\n", .{ page.slug, err });
            continue;
        };
        page_count += 1;
    }

    // Write search index
    try writeSearchIndex(allocator, io, config.out_dir, search_entries.items);

    std.debug.print("\n=== Site Generation Complete ===\n", .{});
    std.debug.print("  Pages:  {d}\n", .{page_count});
    std.debug.print("  Output: {s}/\n", .{config.out_dir});
}

fn copyAssets(allocator: std.mem.Allocator, io: std.Io, source_dir: []const u8, out_dir: []const u8) !void {
    const src_assets = try std.fs.path.join(allocator, &.{ source_dir, "assets" });
    defer allocator.free(src_assets);
    const dest_assets = try std.fs.path.join(allocator, &.{ out_dir, "assets" });
    defer allocator.free(dest_assets);

    copyDirRecursive(allocator, io, src_assets, dest_assets) catch |err| switch (err) {
        error.FileNotFound => return,
        else => return err,
    };
}

fn copyDirRecursive(allocator: std.mem.Allocator, io: std.Io, src: []const u8, dest: []const u8) !void {
    var dir = std.Io.Dir.cwd().openDir(io, src, .{ .iterate = true }) catch return error.FileNotFound;
    defer dir.close(io);

    try std.Io.Dir.cwd().createDirPath(io, dest);

    var iter = dir.iterate();
    while (true) {
        const maybe_entry = iter.next(io) catch break;
        if (maybe_entry) |entry| {
            const src_path = try std.fs.path.join(allocator, &.{ src, entry.name });
            defer allocator.free(src_path);
            const dest_path = try std.fs.path.join(allocator, &.{ dest, entry.name });
            defer allocator.free(dest_path);

            switch (entry.kind) {
                .file => try copyFile(allocator, io, src_path, dest_path),
                .directory => try copyDirRecursive(allocator, io, src_path, dest_path),
                else => {},
            }
        } else {
            break;
        }
    }
}

fn copyFile(allocator: std.mem.Allocator, io: std.Io, src: []const u8, dest: []const u8) !void {
    const content = try std.Io.Dir.cwd().readFileAlloc(io, src, allocator, .limited(5 * 1024 * 1024));
    defer allocator.free(content);
    try writeFileRaw(io, dest, content);
}

// =============================================================================
// Page Rendering
// =============================================================================

fn writePage(
    allocator: std.mem.Allocator,
    io: std.Io,
    manifest: Manifest,
    config: BuildConfig,
    page: PageConfig,
    search_entries: *std.ArrayListUnmanaged(SearchEntry),
) !void {
    const source_path = try std.fs.path.join(allocator, &.{ config.source_dir, page.source });
    defer allocator.free(source_path);
    const raw_content = try std.Io.Dir.cwd().readFileAlloc(io, source_path, allocator, .limited(10 * 1024 * 1024));
    defer allocator.free(raw_content);

    // Determine if markdown or HTML passthrough
    const is_markdown = std.mem.endsWith(u8, page.source, ".md");

    var page_body: []const u8 = undefined;
    var toc_entries: []const TocEntry = &.{};
    var search_text: []const u8 = "";
    var md_html_owned: []const u8 = "";
    var md_text_owned: []const u8 = "";
    var toc_slugs_to_free: []const TocEntry = &.{};

    if (is_markdown) {
        const fm = parseFrontMatter(raw_content);
        var parser = MarkdownParser.init(allocator, fm.body);
        defer parser.deinit();
        const result = try parser.parse();
        md_html_owned = result.html;
        md_text_owned = result.text;
        toc_entries = result.toc;
        toc_slugs_to_free = result.toc;
        page_body = md_html_owned;
        search_text = md_text_owned;
    } else {
        page_body = raw_content;
        search_text = raw_content;
    }
    defer if (is_markdown) {
        allocator.free(md_html_owned);
        allocator.free(md_text_owned);
    };

    // Add to search index
    try search_entries.append(allocator, .{
        .title = page.title,
        .slug = page.slug,
        .section = page.section,
        .content = try allocator.dupe(u8, search_text),
    });

    const file_name = try std.fmt.allocPrint(allocator, "{s}.html", .{page.slug});
    defer allocator.free(file_name);
    const output_path = try std.fs.path.join(allocator, &.{ config.out_dir, file_name });
    defer allocator.free(output_path);

    const nav_html = try buildNav(allocator, manifest, page.slug);
    defer allocator.free(nav_html);

    const toc_html = try buildToc(allocator, toc_entries);
    defer allocator.free(toc_html);

    var html = std.Io.Writer.Allocating.init(allocator);
    defer html.deinit();
    const writer = &html.writer;

    try writeDocument(writer, manifest, page, nav_html, toc_html, page_body);

    const rendered = try html.toOwnedSlice();
    defer allocator.free(rendered);

    try writeFileRaw(io, output_path, rendered);
    std.debug.print("  [OK] {s} -> {s}\n", .{ page.source, file_name });
}

fn writeDocument(
    writer: anytype,
    manifest: Manifest,
    page: PageConfig,
    nav_html: []const u8,
    toc_html: []const u8,
    page_body: []const u8,
) !void {
    try writer.writeAll("<!doctype html>\n<html lang=\"en\">\n<head>\n");
    try writer.writeAll("  <meta charset=\"utf-8\">\n");
    try writer.writeAll("  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n");
    try writer.writeAll("  <title>");
    try writeEscaped(writer, page.title);
    try writer.writeAll(" | ");
    try writeEscaped(writer, manifest.site.title);
    try writer.writeAll("</title>\n");
    try writer.writeAll("  <link rel=\"stylesheet\" href=\"");
    try writeAssetPath(writer, manifest.site.base_url, "assets/style.css");
    try writer.writeAll("\">\n");
    try writer.writeAll("</head>\n<body>\n");

    // Mobile hamburger
    try writer.writeAll("<button class=\"hamburger\" aria-label=\"Toggle navigation\">\n");
    try writer.writeAll("  <span></span><span></span><span></span>\n");
    try writer.writeAll("</button>\n");

    try writer.writeAll("<div class=\"layout\">\n");

    // Sidebar
    try writer.writeAll("  <aside class=\"sidebar\">\n");
    try writer.writeAll("    <div class=\"brand\">");
    try writeEscaped(writer, manifest.site.title);
    try writer.writeAll("</div>\n");
    if (manifest.site.tagline.len != 0) {
        try writer.writeAll("    <div class=\"tagline\">");
        try writeEscaped(writer, manifest.site.tagline);
        try writer.writeAll("</div>\n");
    }
    try writer.writeAll("    <nav class=\"nav\">\n");
    try writer.writeAll(nav_html);
    try writer.writeAll("    </nav>\n");
    try writer.writeAll("  </aside>\n");

    // Main content area with optional TOC
    try writer.writeAll("  <main class=\"content\">\n");
    try writer.writeAll("    <header class=\"page-header\">\n");
    try writer.writeAll("      <h1>");
    try writeEscaped(writer, page.title);
    try writer.writeAll("</h1>\n");
    try writer.writeAll("    </header>\n");
    try writer.writeAll("    <div class=\"page-body\">\n");
    try writer.writeAll(page_body);
    try writer.writeAll("\n    </div>\n");
    try writer.writeAll("    <footer class=\"page-footer\">");
    if (manifest.footer) |footer| {
        try writeEscaped(writer, footer.text);
    }
    try writer.writeAll("</footer>\n");
    try writer.writeAll("  </main>\n");

    // TOC sidebar (right)
    if (toc_html.len > 0) {
        try writer.writeAll("  <aside class=\"toc-sidebar\">\n");
        try writer.writeAll(toc_html);
        try writer.writeAll("  </aside>\n");
    }

    try writer.writeAll("</div>\n");
    try writer.writeAll("<script src=\"");
    try writeAssetPath(writer, manifest.site.base_url, "assets/main.js");
    try writer.writeAll("\"></script>\n");
    try writer.writeAll("</body>\n</html>\n");
}

fn buildNav(allocator: std.mem.Allocator, manifest: Manifest, active_slug: []const u8) ![]const u8 {
    var nav = std.Io.Writer.Allocating.init(allocator);
    defer nav.deinit();
    const writer = &nav.writer;

    var current_section: ?[]const u8 = null;
    for (manifest.pages) |page| {
        if (current_section == null or !std.mem.eql(u8, page.section, current_section.?)) {
            if (current_section != null) {
                try writer.writeAll("        </ul>\n      </div>\n");
            }
            current_section = page.section;
            try writer.writeAll("      <div class=\"nav-section\">\n");
            try writer.writeAll("        <div class=\"nav-title\">");
            try writeEscaped(writer, page.section);
            try writer.writeAll("</div>\n");
            try writer.writeAll("        <ul>\n");
        }
        try writer.writeAll("          <li><a class=\"nav-link");
        if (std.mem.eql(u8, page.slug, active_slug)) {
            try writer.writeAll(" active");
        }
        try writer.writeAll("\" href=\"");
        try writeLink(writer, manifest.site.base_url, page.slug);
        try writer.writeAll("\">");
        try writeEscaped(writer, page.title);
        try writer.writeAll("</a></li>\n");
    }
    if (current_section != null) {
        try writer.writeAll("        </ul>\n      </div>\n");
    }
    return nav.toOwnedSlice();
}

fn buildToc(allocator: std.mem.Allocator, entries: []const TocEntry) ![]const u8 {
    if (entries.len == 0) return try allocator.dupe(u8, "");

    var toc = std.Io.Writer.Allocating.init(allocator);
    defer toc.deinit();
    const writer = &toc.writer;

    try writer.writeAll("    <nav class=\"toc\">\n");
    try writer.writeAll("      <h3>On this page</h3>\n");
    try writer.writeAll("      <ul>\n");
    for (entries) |entry| {
        if (entry.level > 3) continue; // Only show h1-h3 in TOC
        const indent: usize = @as(usize, entry.level -| 1) * 12;
        try writer.print("        <li style=\"padding-left:{d}px\"><a href=\"#{s}\">", .{ indent, entry.slug });
        try writeEscaped(writer, entry.text);
        try writer.writeAll("</a></li>\n");
    }
    try writer.writeAll("      </ul>\n");
    try writer.writeAll("    </nav>\n");
    return toc.toOwnedSlice();
}

// =============================================================================
// Search Index
// =============================================================================

fn writeSearchIndex(
    allocator: std.mem.Allocator,
    io: std.Io,
    out_dir: []const u8,
    entries: []const SearchEntry,
) !void {
    var json = std.Io.Writer.Allocating.init(allocator);
    defer json.deinit();
    const w = &json.writer;

    try w.writeAll("[\n");
    for (entries, 0..) |entry, idx| {
        try w.writeAll("  {\"title\":\"");
        try writeJsonEscaped(w, entry.title);
        try w.writeAll("\",\"slug\":\"");
        try writeJsonEscaped(w, entry.slug);
        try w.writeAll("\",\"section\":\"");
        try writeJsonEscaped(w, entry.section);
        try w.writeAll("\",\"content\":\"");
        // Truncate content for search to first 500 chars
        const content = if (entry.content.len > 500) entry.content[0..500] else entry.content;
        try writeJsonEscaped(w, content);
        try w.writeAll("\"}");
        if (idx + 1 < entries.len) try w.writeAll(",");
        try w.writeAll("\n");
    }
    try w.writeAll("]\n");

    const json_data = try json.toOwnedSlice();
    defer allocator.free(json_data);

    const path = try std.fmt.allocPrint(allocator, "{s}/search.json", .{out_dir});
    defer allocator.free(path);
    try writeFileRaw(io, path, json_data);
    std.debug.print("  [OK] search.json\n", .{});
}

fn writeJsonEscaped(writer: anytype, text: []const u8) !void {
    for (text) |c| {
        switch (c) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll(" "),
            '\r' => {},
            '\t' => try writer.writeAll(" "),
            else => {
                if (c < 0x20) continue;
                try writer.writeByte(c);
            },
        }
    }
}

// =============================================================================
// Helpers
// =============================================================================

fn writeEscaped(writer: anytype, text: []const u8) !void {
    for (text) |ch| {
        switch (ch) {
            '&' => try writer.writeAll("&amp;"),
            '<' => try writer.writeAll("&lt;"),
            '>' => try writer.writeAll("&gt;"),
            '"' => try writer.writeAll("&quot;"),
            else => try writer.writeByte(ch),
        }
    }
}

fn writeLink(writer: anytype, base_url: []const u8, slug: []const u8) !void {
    if (base_url.len == 0) {
        try writer.print("{s}.html", .{slug});
        return;
    }
    if (base_url[base_url.len - 1] == '/') {
        try writer.print("{s}{s}.html", .{ base_url, slug });
    } else {
        try writer.print("{s}/{s}.html", .{ base_url, slug });
    }
}

fn writeAssetPath(writer: anytype, base_url: []const u8, path: []const u8) !void {
    if (base_url.len == 0) {
        try writer.writeAll(path);
        return;
    }
    if (base_url[base_url.len - 1] == '/') {
        try writer.print("{s}{s}", .{ base_url, path });
    } else {
        try writer.print("{s}/{s}", .{ base_url, path });
    }
}

fn writeFileRaw(io: std.Io, path: []const u8, content: []const u8) !void {
    var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    defer file.close(io);
    var buffer: [8192]u8 = undefined;
    var writer = file.writer(io, &buffer);
    try writer.interface.writeAll(content);
}
