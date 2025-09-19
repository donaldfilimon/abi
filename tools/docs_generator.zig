const std = @import("std");

/// Minimal documentation generator used by CI and local builds.
///
/// The previous implementation tried to fabricate a complete GitHub Pages
/// site in Zig code. That approach made the tool brittle, hard to maintain,
/// and it produced markdown that quickly drifted away from the real project
/// documentation that now lives in the repository. The new generator focuses
/// on the parts that actually benefit from automation:
///   * keep the docs folders ready for publishing (no Jekyll processing)
///   * build a small searchable index over the generated markdown pages
///   * extract a quick overview of public declarations from the source tree
///   * leave a placeholder for compiler generated docs so contributors know
///     how to refresh them when the Zig toolchain is available
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        switch (gpa.deinit()) {
            .ok => {},
            .leak => std.log.warn("docs generator leaked memory", .{}),
        }
    }

    var generator = Generator{ .allocator = gpa.allocator() };
    try generator.run();
}

const Generator = struct {
    allocator: std.mem.Allocator,

    fn run(self: *Generator) !void {
        try self.ensureLayout();
        try self.generateCodeIndex();
        try self.generateSearchIndex();
        try self.ensureZigDocsPlaceholder();
    }

    fn ensureLayout(self: *Generator) !void {
        const cwd = std.fs.cwd();
        try cwd.makePath("docs/generated");
        try cwd.makePath("docs/zig-docs");

        // GitHub Pages must see this file for the pre-built docs directory.
        var nojekyll = try cwd.createFile("docs/.nojekyll", .{ .truncate = true });
        defer nojekyll.close();
    }

    fn generateCodeIndex(self: *Generator) !void {
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const a = arena.allocator();

        var files = std.ArrayList([]const u8).init(a);
        defer files.deinit();
        try collectZigFiles(a, "src", &files);

        var out = try std.fs.cwd().createFile("docs/generated/CODE_API_INDEX.md", .{ .truncate = true });
        defer out.close();
        var writer = out.writer();

        try writer.writeAll("# Code API Index\n\n");
        if (files.items.len == 0) {
            try writer.writeAll("No Zig source files found under `src/`.\n");
            return;
        }
        try writer.print("Indexed {d} files under `src/`.\n\n", .{files.items.len});

        var decls = std.ArrayList(Declaration).init(a);
        defer decls.deinit();

        for (files.items) |rel| {
            decls.clearRetainingCapacity();
            try scanFile(a, rel, &decls);
            if (decls.items.len == 0) continue;

            try writer.print("## {s}\n\n", .{rel});
            for (decls.items) |decl| {
                try writer.print("- {s} `{s}`\n", .{ decl.kind, decl.name });
                if (decl.doc.len > 0) {
                    try writer.writeAll("\n");
                    try writer.writeAll(decl.doc);
                    if (decl.doc[decl.doc.len - 1] != '\n') try writer.writeByte('\n');
                }
                if (decl.signature.len > 0) {
                    try writer.print("```zig\n{s}\n```\n", .{decl.signature});
                }
                try writer.writeByte('\n');
            }
        }
    }

    fn generateSearchIndex(self: *Generator) !void {
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const a = arena.allocator();

        var docs_dir = std.fs.cwd().openDir("docs/generated", .{ .iterate = true }) catch |err| switch (err) {
            error.FileNotFound => return,
            else => return err,
        };
        defer docs_dir.close();

        var files = std.ArrayList([]const u8).init(a);
        defer files.deinit();

        var it = docs_dir.iterate();
        while (it.next() catch null) |entry| {
            if (entry.kind != .file) continue;
            if (!std.mem.endsWith(u8, entry.name, ".md")) continue;
            const rel = try std.fs.path.join(a, &[_][]const u8{ "generated", entry.name });
            try files.append(rel);
        }

        if (files.items.len == 0) return;

        var out = try std.fs.cwd().createFile("docs/generated/search_index.json", .{ .truncate = true });
        defer out.close();

        try out.writeAll("[\n");
        var first = true;
        for (files.items) |rel| {
            const full = try std.fs.path.join(a, &[_][]const u8{ "docs", rel });
            var title_buf: []const u8 = "";
            var excerpt_buf: []const u8 = "";
            getTitleAndExcerpt(a, full, &title_buf, &excerpt_buf) catch {
                title_buf = std.fs.path.basename(rel);
                excerpt_buf = "";
            };

            if (!first) {
                try out.writeAll(",\n");
            } else {
                first = false;
            }

            var rel_web = try a.dupe(u8, rel);
            for (rel_web) |*ch| {
                if (ch.* == std.fs.path.sep) ch.* = '/';
            }

            try out.writeAll("  {\"file\": ");
            try writeJsonString(out, rel_web);
            try out.writeAll(", \"title\": ");
            try writeJsonString(out, title_buf);
            try out.writeAll(", \"excerpt\": ");
            try writeJsonString(out, excerpt_buf);
            try out.writeAll("}");
        }
        try out.writeAll("\n]\n");
    }

    fn ensureZigDocsPlaceholder(self: *Generator) !void {
        const cwd = std.fs.cwd();
        try cwd.makePath("docs/zig-docs");
        const info = "<!DOCTYPE html>\n<html lang=\"en\">\n<head><meta charset=\"utf-8\"><title>Zig Docs</title></head>\n" ++
            "<body><h1>Zig documentation not generated</h1>" ++
            "<p>Run <code>zig doc src/root.zig --output-dir docs/zig-docs</code> " ++
            "with a matching Zig toolchain to refresh this folder.</p></body></html>\n";
        var out = try cwd.createFile("docs/zig-docs/index.html", .{ .truncate = true });
        defer out.close();
        try out.writeAll(info);
    }
};

const Declaration = struct {
    name: []const u8,
    kind: []const u8,
    signature: []const u8,
    doc: []const u8,
};

fn collectZigFiles(allocator: std.mem.Allocator, dir_path: []const u8, out: *std.ArrayList([]const u8)) !void {
    var dir = std.fs.cwd().openDir(dir_path, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => return,
        else => return err,
    };
    defer dir.close();

    var it = dir.iterate();
    while (it.next() catch null) |entry| {
        switch (entry.kind) {
            .directory => {
                if (entry.name.len == 0 or std.mem.eql(u8, entry.name, ".") or std.mem.eql(u8, entry.name, "..")) continue;
                if (std.mem.eql(u8, entry.name, "zig-cache") or std.mem.eql(u8, entry.name, "zig-out")) continue;
                const sub = try std.fs.path.join(allocator, &[_][]const u8{ dir_path, entry.name });
                try collectZigFiles(allocator, sub, out);
            },
            .file => {
                if (!std.mem.endsWith(u8, entry.name, ".zig")) continue;
                const rel = try std.fs.path.join(allocator, &[_][]const u8{ dir_path, entry.name });
                try out.append(rel);
            },
            else => {},
        }
    }
}

fn scanFile(allocator: std.mem.Allocator, rel_path: []const u8, decls: *std.ArrayList(Declaration)) !void {
    const data = try std.fs.cwd().readFileAlloc(allocator, rel_path, 1 << 20);
    defer allocator.free(data);

    var it = std.mem.splitScalar(u8, data, '\n');
    var doc_buf = std.ArrayList(u8).init(allocator);
    defer doc_buf.deinit();

    while (it.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (trimmed.len == 0) {
            doc_buf.clearRetainingCapacity();
            continue;
        }

        if (std.mem.startsWith(u8, trimmed, "///")) {
            const doc_line = std.mem.trim(u8, trimmed[3..], " \t");
            try doc_buf.appendSlice(doc_line);
            try doc_buf.append('\n');
            continue;
        }

        if (isPubDecl(trimmed)) {
            const kind = detectKind(trimmed);
            const name = extractName(allocator, trimmed) catch {
                doc_buf.clearRetainingCapacity();
                continue;
            };
            const sig = try allocator.dupe(u8, trimmed);
            const doc = try allocator.dupe(u8, doc_buf.items);
            doc_buf.clearRetainingCapacity();

            try decls.append(.{
                .name = name,
                .kind = kind,
                .signature = sig,
                .doc = doc,
            });
            continue;
        }

        if (!std.mem.startsWith(u8, trimmed, "//")) {
            doc_buf.clearRetainingCapacity();
        }
    }
}

fn isPubDecl(line: []const u8) bool {
    if (!std.mem.startsWith(u8, line, "pub ")) return false;
    return std.mem.indexOfAny(u8, line[4..], "fctuv") != null or
        std.mem.startsWith(u8, line, "pub usingnamespace") or
        std.mem.indexOf(u8, line, " struct") != null or
        std.mem.indexOf(u8, line, " enum") != null;
}

fn detectKind(line: []const u8) []const u8 {
    if (std.mem.startsWith(u8, line, "pub fn ")) return "fn";
    if (std.mem.startsWith(u8, line, "pub const ")) {
        if (std.mem.indexOf(u8, line, " struct") != null) return "type";
        if (std.mem.indexOf(u8, line, " enum") != null) return "type";
        return "const";
    }
    if (std.mem.startsWith(u8, line, "pub var ")) return "var";
    if (std.mem.startsWith(u8, line, "pub usingnamespace")) return "usingnamespace";
    return "pub";
}

fn extractName(allocator: std.mem.Allocator, line: []const u8) ![]const u8 {
    var rest = line;
    if (std.mem.startsWith(u8, rest, "pub fn ")) {
        rest = rest[7..];
    } else if (std.mem.startsWith(u8, rest, "pub const ")) {
        rest = rest[10..];
    } else if (std.mem.startsWith(u8, rest, "pub var ")) {
        rest = rest[8..];
    } else if (std.mem.startsWith(u8, rest, "pub usingnamespace ")) {
        rest = rest[18..];
    } else if (std.mem.startsWith(u8, rest, "pub ")) {
        rest = rest[4..];
    }

    var i: usize = 0;
    while (i < rest.len) : (i += 1) {
        const c = rest[i];
        const is_ident = (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or (c >= '0' and c <= '9') or c == '_' or c == '.';
        if (!is_ident) break;
    }

    const ident = std.mem.trim(u8, rest[0..i], " \t");
    if (ident.len == 0) return error.Invalid;
    return allocator.dupe(u8, ident);
}

fn getTitleAndExcerpt(allocator: std.mem.Allocator, path: []const u8, title_out: *[]const u8, excerpt_out: *[]const u8) !void {
    const data = try std.fs.cwd().readFileAlloc(allocator, path, 1 << 20);
    defer allocator.free(data);

    var it = std.mem.splitScalar(u8, data, '\n');
    var first_heading: ?[]const u8 = null;
    var in_code = false;

    var excerpt = std.ArrayList(u8).init(allocator);
    defer excerpt.deinit();

    while (it.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (std.mem.startsWith(u8, trimmed, "```")) {
            in_code = !in_code;
            continue;
        }
        if (in_code) continue;

        if (first_heading == null and std.mem.startsWith(u8, trimmed, "#")) {
            var idx: usize = 0;
            while (idx < trimmed.len and trimmed[idx] == '#') idx += 1;
            const after = std.mem.trim(u8, trimmed[idx..], " \t");
            if (after.len > 0) first_heading = after;
            continue;
        }

        if (trimmed.len == 0) continue;
        if (trimmed[0] == '#') continue;
        if (trimmed[0] == '|') continue;

        if (excerpt.items.len > 0) try excerpt.append(' ');
        var j: usize = 0;
        while (j < trimmed.len and excerpt.items.len < 300) : (j += 1) {
            const c = trimmed[j];
            if (c == '`') continue;
            try excerpt.append(c);
        }
        if (excerpt.items.len >= 300) break;
    }

    if (first_heading) |h| {
        title_out.* = try allocator.dupe(u8, h);
    } else {
        title_out.* = try allocator.dupe(u8, std.fs.path.basename(path));
    }
    excerpt_out.* = try allocator.dupe(u8, excerpt.items);
}

fn writeJsonString(out: std.fs.File, s: []const u8) !void {
    try out.writeAll("\"");
    var i: usize = 0;
    while (i < s.len) : (i += 1) {
        const c = s[i];
        switch (c) {
            '\\' => try out.writeAll("\\\\"),
            '"' => try out.writeAll("\\\""),
            '\n' => try out.writeAll("\\n"),
            '\r' => try out.writeAll("\\r"),
            '\t' => try out.writeAll("\\t"),
            else => {
                var buf: [1]u8 = .{c};
                try out.writeAll(buf[0..]);
            },
        }
    }
    try out.writeAll("\"");
}
