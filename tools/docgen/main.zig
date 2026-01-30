//! ABI Documentation Generator
//!
//! A custom static site generator written in Zig that compiles Markdown
//! documentation into a styled HTML website.
//!
//! Usage:
//!   zig build docs-site           # Generate HTML site
//!

const std = @import("std");
const Allocator = std.mem.Allocator;

// =============================================================================
// Configuration
// =============================================================================

const Config = struct {
    input_dir: []const u8 = "docs",
    output_dir: []const u8 = "docs_html",
    base_url: []const u8 = "/abi",
    site_title: []const u8 = "ABI Framework Documentation",
    version: []const u8 = "0.16.0",
};

// =============================================================================
// Document Types
// =============================================================================

const FrontMatter = struct {
    title: []const u8 = "",
    description: []const u8 = "",
    category: []const u8 = "",
    order: u32 = 999,
};

const TocEntry = struct {
    level: u8,
    text: []const u8,
    slug: []const u8,
};

const SearchEntry = struct {
    title: []const u8,
    path: []const u8,
    content: []const u8,
};

// =============================================================================
// Markdown Parser
// =============================================================================

const MarkdownParser = struct {
    allocator: Allocator,
    source: []const u8,
    pos: usize,
    output: std.ArrayListUnmanaged(u8),
    toc: std.ArrayListUnmanaged(TocEntry),
    text_content: std.ArrayListUnmanaged(u8), // For search index
    in_code_block: bool,
    code_lang: []const u8,

    pub fn init(allocator: Allocator, source: []const u8) MarkdownParser {
        return .{
            .allocator = allocator,
            .source = source,
            .pos = 0,
            .output = .empty,
            .toc = .empty,
            .text_content = .empty,
            .in_code_block = false,
            .code_lang = "",
        };
    }

    pub fn deinit(self: *MarkdownParser) void {
        self.output.deinit(self.allocator);
        self.toc.deinit(self.allocator);
        self.text_content.deinit(self.allocator);
    }

    pub fn parse(self: *MarkdownParser) !struct { html: []const u8, toc: []const TocEntry, text: []const u8 } {
        while (self.pos < self.source.len) {
            try self.parseLine();
        }

        if (self.in_code_block) {
            try self.output.appendSlice(self.allocator, "</code></pre>\n");
        }

        return .{
            .html = try self.output.toOwnedSlice(self.allocator),
            .toc = try self.toc.toOwnedSlice(self.allocator),
            .text = try self.text_content.toOwnedSlice(self.allocator),
        };
    }

    fn parseLine(self: *MarkdownParser) !void {
        const line_end = std.mem.indexOfScalar(u8, self.source[self.pos..], '\n') orelse self.source.len - self.pos;
        const line = self.source[self.pos .. self.pos + line_end];
        self.pos += line_end + 1;

        // Collect text content for search (skip code blocks)
        if (!self.in_code_block and !std.mem.startsWith(u8, line, "```")) {
            try self.text_content.appendSlice(self.allocator, line);
            try self.text_content.append(self.allocator, ' ');
        }

        // Code block handling
        if (std.mem.startsWith(u8, line, "```")) {
            if (self.in_code_block) {
                try self.output.appendSlice(self.allocator, "</code></pre>\n");
                self.in_code_block = false;
            } else {
                self.in_code_block = true;
                self.code_lang = if (line.len > 3) line[3..] else "";
                try self.output.appendSlice(self.allocator, "<pre class=\"code-block");
                if (self.code_lang.len > 0) {
                    try self.output.appendSlice(self.allocator, " language-");
                    try self.output.appendSlice(self.allocator, self.code_lang);
                }
                try self.output.appendSlice(self.allocator, "\"><code>");
            }
            return;
        }

        if (self.in_code_block) {
            try self.appendEscaped(line);
            try self.output.append(self.allocator, '\n');
            return;
        }

        // Empty line
        if (line.len == 0) {
            try self.output.appendSlice(self.allocator, "\n");
            return;
        }

        // Headers
        if (line[0] == '#') {
            try self.parseHeader(line);
            return;
        }

        // Horizontal rule
        if (std.mem.eql(u8, line, "---") or std.mem.eql(u8, line, "***")) {
            try self.output.appendSlice(self.allocator, "<hr>\n");
            return;
        }

        // Blockquote
        if (line[0] == '>') {
            const content = if (line.len > 2) line[2..] else "";
            try self.output.appendSlice(self.allocator, "<blockquote>");
            try self.parseInline(content);
            try self.output.appendSlice(self.allocator, "</blockquote>\n");
            return;
        }

        // List items
        if (line.len > 1 and (line[0] == '-' or line[0] == '*') and line[1] == ' ') {
            try self.output.appendSlice(self.allocator, "<li>");
            try self.parseInline(line[2..]);
            try self.output.appendSlice(self.allocator, "</li>\n");
            return;
        }

        // Table row
        if (line[0] == '|') {
            try self.parseTableRow(line);
            return;
        }

        // Paragraph
        try self.output.appendSlice(self.allocator, "<p>");
        try self.parseInline(line);
        try self.output.appendSlice(self.allocator, "</p>\n");
    }

    fn parseHeader(self: *MarkdownParser, line: []const u8) !void {
        var level: u8 = 0;
        while (level < line.len and line[level] == '#') : (level += 1) {}
        if (level > 6) level = 6;

        const content = if (level < line.len and line[level] == ' ') line[level + 1 ..] else line[level..];
        const slug = try self.generateSlug(content);

        try self.toc.append(self.allocator, .{ .level = level, .text = content, .slug = slug });

        try self.output.appendSlice(self.allocator, "<h");
        try self.output.append(self.allocator, '0' + level);
        try self.output.appendSlice(self.allocator, " id=\"");
        try self.output.appendSlice(self.allocator, slug);
        try self.output.appendSlice(self.allocator, "\">");
        try self.parseInline(content);
        try self.output.appendSlice(self.allocator, "<a class=\"anchor\" href=\"#");
        try self.output.appendSlice(self.allocator, slug);
        try self.output.appendSlice(self.allocator, "\">#</a></h");
        try self.output.append(self.allocator, '0' + level);
        try self.output.appendSlice(self.allocator, ">\n");
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
                        const alt_text = text[i + 2 .. bracket_end];
                        const img_url = text[bracket_end + 2 .. paren_end];

                        try self.output.appendSlice(self.allocator, "<img src=\"");
                        try self.output.appendSlice(self.allocator, img_url);
                        try self.output.appendSlice(self.allocator, "\" alt=\"");
                        try self.output.appendSlice(self.allocator, alt_text);
                        try self.output.appendSlice(self.allocator, "\">");

                        i = paren_end + 1;
                        continue;
                    }
                }
            }

            // Bold **text**
            if (i + 1 < text.len and c == '*' and text[i + 1] == '*') {
                if (std.mem.indexOf(u8, text[i + 2 ..], "**")) |end| {
                    try self.output.appendSlice(self.allocator, "<strong>");
                    try self.parseInline(text[i + 2 .. i + 2 + end]);
                    try self.output.appendSlice(self.allocator, "</strong>");
                    i += 4 + end;
                    continue;
                }
            }

            // Inline code `text`
            if (c == '`') {
                if (std.mem.indexOfScalar(u8, text[i + 1 ..], '`')) |end| {
                    try self.output.appendSlice(self.allocator, "<code>");
                    try self.appendEscaped(text[i + 1 .. i + 1 + end]);
                    try self.output.appendSlice(self.allocator, "</code>");
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
                        try self.output.appendSlice(self.allocator, "<a href=\"");
                        try self.output.appendSlice(self.allocator, link_url);
                        try self.output.appendSlice(self.allocator, "\">");
                        try self.parseInline(link_text);
                        try self.output.appendSlice(self.allocator, "</a>");
                        i += bracket_end + 3 + paren_end;
                        continue;
                    }
                }
            }

            try self.appendEscapedChar(c);
            i += 1;
        }
    }

    fn parseTableRow(self: *MarkdownParser, line: []const u8) !void {
        if (std.mem.indexOf(u8, line, "---") != null) return;

        try self.output.appendSlice(self.allocator, "<tr>");
        var iter = std.mem.splitScalar(u8, line, '|');
        while (iter.next()) |cell| {
            const trimmed = std.mem.trim(u8, cell, " \t");
            if (trimmed.len > 0) {
                try self.output.appendSlice(self.allocator, "<td>");
                try self.parseInline(trimmed);
                try self.output.appendSlice(self.allocator, "</td>");
            }
        }
        try self.output.appendSlice(self.allocator, "</tr>\n");
    }

    fn generateSlug(self: *MarkdownParser, text: []const u8) ![]const u8 {
        var slug = std.ArrayListUnmanaged(u8).empty;
        for (text) |c| {
            if (std.ascii.isAlphanumeric(c)) {
                try slug.append(self.allocator, std.ascii.toLower(c));
            } else if (c == ' ' or c == '-') {
                if (slug.items.len > 0 and slug.items[slug.items.len - 1] != '-') {
                    try slug.append(self.allocator, '-');
                }
            }
        }
        while (slug.items.len > 0 and slug.items[slug.items.len - 1] == '-') {
            _ = slug.pop();
        }
        return try slug.toOwnedSlice(self.allocator);
    }

    fn appendEscaped(self: *MarkdownParser, text: []const u8) !void {
        for (text) |c| try self.appendEscapedChar(c);
    }

    fn appendEscapedChar(self: *MarkdownParser, c: u8) !void {
        switch (c) {
            '<' => try self.output.appendSlice(self.allocator, "&lt;"),
            '>' => try self.output.appendSlice(self.allocator, "&gt;"),
            '&' => try self.output.appendSlice(self.allocator, "&amp;"),
            '"' => try self.output.appendSlice(self.allocator, "&quot;"),
            else => try self.output.append(self.allocator, c),
        }
    }
};

// =============================================================================
// Front Matter Parser
// =============================================================================

fn parseFrontMatter(content: []const u8) struct { front_matter: FrontMatter, body: []const u8 } {
    if (!std.mem.startsWith(u8, content, "---\n")) {
        return .{ .front_matter = .{}, .body = content };
    }

    const end_marker = std.mem.indexOf(u8, content[4..], "\n---") orelse return .{ .front_matter = .{}, .body = content };
    const yaml = content[4 .. 4 + end_marker];
    const body = content[4 + end_marker + 5 ..];

    var front_matter = FrontMatter{};
    var lines = std.mem.splitScalar(u8, yaml, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (trimmed.len == 0) continue;

        if (std.mem.indexOf(u8, trimmed, ":")) |colon_pos| {
            const key = std.mem.trim(u8, trimmed[0..colon_pos], " \t\"");
            const value = std.mem.trim(u8, trimmed[colon_pos + 1 ..], " \t\"");

            if (std.mem.eql(u8, key, "title")) front_matter.title = value;
            if (std.mem.eql(u8, key, "description")) front_matter.description = value;
            if (std.mem.eql(u8, key, "category")) front_matter.category = value;
            if (std.mem.eql(u8, key, "order")) front_matter.order = std.fmt.parseInt(u32, value, 10) catch 999;
        }
    }

    return .{ .front_matter = front_matter, .body = body };
}

// =============================================================================
// HTML Generation
// =============================================================================

fn generateHtml(allocator: Allocator, title: []const u8, description: []const u8, html_content: []const u8, toc: []const TocEntry, config: Config) ![]const u8 {
    // Build TOC HTML
    var toc_html = std.ArrayListUnmanaged(u8).empty;
    defer toc_html.deinit(allocator);

    if (toc.len > 0) {
        try toc_html.appendSlice(allocator, "      <nav class=\"toc\"><h3>On this page</h3><ul>\n");
        for (toc) |entry| {
            const indent = (entry.level - 1) * 16;
            const toc_item = try std.fmt.allocPrint(allocator, "        <li style=\"padding-left: {d}px\"><a href=\"#{s}\">{s}</a></li>\n", .{ indent, entry.slug, entry.text });
            defer allocator.free(toc_item);
            try toc_html.appendSlice(allocator, toc_item);
        }
        try toc_html.appendSlice(allocator, "      </ul></nav>\n");
    }

    // Build description HTML
    const desc_html = if (description.len > 0)
        try std.fmt.allocPrint(allocator, "        <p class=\"doc-description\">{s}</p>\n", .{description})
    else
        try allocator.dupe(u8, "");
    defer allocator.free(desc_html);

    // Generate full HTML
    return std.fmt.allocPrint(allocator,
        \\<!DOCTYPE html>
        \\<html lang="en" data-theme="dark">
        \\<head>
        \\  <meta charset="UTF-8">
        \\  <meta name="viewport" content="width=device-width, initial-scale=1.0">
        \\  <title>{s} - {s}</title>
        \\  <meta name="description" content="{s}">
        \\  <link rel="preconnect" href="https://fonts.googleapis.com">
        \\  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
        \\  <link rel="stylesheet" href="{s}/assets/css/style.css">
        \\</head>
        \\<body>
        \\<nav class="navbar">
        \\  <div class="nav-container">
        \\    <a href="{s}/" class="nav-logo">
        \\      <span class="logo-text">ABI</span>
        \\      <span class="logo-version">v{s}</span>
        \\    </a>
        \\    <div class="nav-links">
        \\      <a href="{s}/">Home</a>
        \\      <a href="{s}/intro.html">Docs</a>
        \\      <a href="https://github.com/donaldfilimon/abi" target="_blank">GitHub</a>
        \\    </div>
        \\    <button class="theme-toggle" onclick="toggleTheme()"><span class="theme-icon">🌙</span></button>
        \\  </div>
        \\</nav>
        \\<div class="main-container">
        \\  <main class="content" style="margin-left: 0;">
        \\    <article class="doc-article">
        \\      <header class="doc-header">
        \\        <h1>{s}</h1>
        \\{s}      </header>
        \\{s}      <div class="doc-content">{s}</div>
        \\    </article>
        \\  </main>
        \\</div>
        \\<footer class="footer" style="margin-left: 0;">
        \\  <div class="footer-content">
        \\    <div class="footer-section">
        \\      <h4>ABI Framework</h4>
        \\      <p>Modern Zig framework for AI services and high-performance systems.</p>
        \\    </div>
        \\  </div>
        \\  <div class="footer-bottom"><p>&copy; 2026 ABI Framework. Built with Zig.</p></div>
        \\</footer>
        \\<script src="{s}/assets/js/main.js"></script>
        \\</body>
        \\</html>
        \\
    , .{ title, config.site_title, description, config.base_url, config.base_url, config.version, config.base_url, config.base_url, title, desc_html, toc_html.items, html_content, config.base_url });
}

// =============================================================================
// Main
// =============================================================================

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const config = Config{};

    std.debug.print("🚀 ABI Documentation Generator\n", .{});
    std.debug.print("   Input:  {s}\n", .{config.input_dir});
    std.debug.print("   Output: {s}\n\n", .{config.output_dir});

    // Initialize I/O backend for Zig 0.16
    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    // Create output directories
    std.Io.Dir.cwd().createDirPath(io, config.output_dir) catch {};
    const css_dir = try std.fmt.allocPrint(allocator, "{s}/assets/css", .{config.output_dir});
    defer allocator.free(css_dir);
    std.Io.Dir.cwd().createDirPath(io, css_dir) catch {};
    const js_dir = try std.fmt.allocPrint(allocator, "{s}/assets/js", .{config.output_dir});
    defer allocator.free(js_dir);
    std.Io.Dir.cwd().createDirPath(io, js_dir) catch {};

    // Copy assets
    const css_path = try std.fmt.allocPrint(allocator, "{s}/assets/css/style.css", .{config.output_dir});
    defer allocator.free(css_path);
    try writeFile(io, allocator, css_path, @embedFile("templates/style.css"));
    const js_path = try std.fmt.allocPrint(allocator, "{s}/assets/js/main.js", .{config.output_dir});
    defer allocator.free(js_path);
    try writeFile(io, allocator, js_path, @embedFile("templates/main.js"));
    std.debug.print("📦 Assets copied\n", .{});

    // Search index
    var search_index = std.ArrayListUnmanaged(SearchEntry).empty;
    defer {
        for (search_index.items) |item| {
            allocator.free(item.title);
            allocator.free(item.path);
            allocator.free(item.content);
        }
        search_index.deinit(allocator);
    }

    // Process markdown files
    var doc_count: usize = 0;
    try processDirectory(allocator, io, config, config.input_dir, "", &doc_count, &search_index);

    // Write search index
    try writeSearchIndex(io, allocator, config.output_dir, search_index.items);

    std.debug.print("\n✅ Generated {d} pages\n", .{doc_count});
    std.debug.print("   Open {s}/index.html to view\n", .{config.output_dir});
}

fn processDirectory(allocator: Allocator, io: anytype, config: Config, base: []const u8, rel: []const u8, count: *usize, search_index: *std.ArrayListUnmanaged(SearchEntry)) !void {
    const path = if (rel.len > 0) try std.fmt.allocPrint(allocator, "{s}/{s}", .{ base, rel }) else base;
    defer if (rel.len > 0) allocator.free(path);

    var dir = std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true }) catch return;
    defer dir.close(io);

    var iter = dir.iterate();
    while (try iter.next(io)) |entry| {
        const name = entry.name;
        if (name[0] == '.' or name[0] == '_') continue;
        if (std.mem.eql(u8, name, "plans") or std.mem.eql(u8, name, "archive")) continue;

        const child_rel = if (rel.len > 0) try std.fmt.allocPrint(allocator, "{s}/{s}", .{ rel, name }) else try allocator.dupe(u8, name);
        defer allocator.free(child_rel);

        if (entry.kind == .directory) {
            try processDirectory(allocator, io, config, base, child_rel, count, search_index);
        } else if (entry.kind == .file and std.mem.endsWith(u8, name, ".md")) {
            try processMarkdownFile(allocator, io, config, path, name, count, search_index, rel);
        }
    }
}

fn processMarkdownFile(allocator: Allocator, io: anytype, config: Config, dir_path: []const u8, filename: []const u8, count: *usize, search_index: *std.ArrayListUnmanaged(SearchEntry), rel_dir: []const u8) !void {
    const full_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ dir_path, filename });
    defer allocator.free(full_path);

    const content = std.Io.Dir.cwd().readFileAlloc(io, full_path, allocator, .limited(1024 * 1024)) catch return;
    defer allocator.free(content);

    const parsed = parseFrontMatter(content);
    var parser = MarkdownParser.init(allocator, parsed.body);
    defer parser.deinit();
    const result = try parser.parse();
    defer allocator.free(result.html);
    defer {
        for (result.toc) |entry| allocator.free(entry.slug);
        allocator.free(result.toc);
    }
    defer allocator.free(result.text);

    const title = if (parsed.front_matter.title.len > 0) parsed.front_matter.title else filename[0 .. filename.len - 3];
    const html = try generateHtml(allocator, title, parsed.front_matter.description, result.html, result.toc, config);
    defer allocator.free(html);

    const html_name = try std.mem.replaceOwned(u8, allocator, filename, ".md", ".html");
    defer allocator.free(html_name);

    // Construct relative path for search index
    const rel_path = if (rel_dir.len > 0)
        try std.fmt.allocPrint(allocator, "{s}/{s}", .{ rel_dir, html_name })
    else
        try allocator.dupe(u8, html_name);
    defer allocator.free(rel_path);

    // Add to search index
    try search_index.append(allocator, .{
        .title = try allocator.dupe(u8, title),
        .path = try allocator.dupe(u8, rel_path),
        .content = try allocator.dupe(u8, result.text),
    });

    const output_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ config.output_dir, html_name });
    defer allocator.free(output_path);

    try writeFile(io, allocator, output_path, html);
    std.debug.print("   ✓ {s}\n", .{filename});
    count.* += 1;
}

fn writeSearchIndex(io: anytype, allocator: Allocator, output_dir: []const u8, items: []const SearchEntry) !void {
    // Format JSON using std.json.fmt (Zig 0.16 API)
    const json_str = try std.fmt.allocPrint(allocator, "{}", .{std.json.fmt(items, .{ .whitespace = .indent_2 })});
    defer allocator.free(json_str);

    const path = try std.fmt.allocPrint(allocator, "{s}/search.json", .{output_dir});
    defer allocator.free(path);
    try writeFile(io, allocator, path, json_str);
    std.debug.print("🔍 Search index generated\n", .{});
}

fn writeFile(io: anytype, allocator: Allocator, path: []const u8, content: []const u8) !void {
    _ = allocator;
    var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    defer file.close(io);
    try file.writeStreamingAll(io, content);
}
