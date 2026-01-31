const std = @import("std");

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

pub fn main(init: std.process.Init.Minimal) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const status = gpa.deinit();
        if (status == .leak) {
            std.debug.print("Memory leak detected\n", .{});
        }
    }
    const allocator = gpa.allocator();

    var args_iter = init.args.iterateAllocator(allocator) catch |err| {
        std.log.err("Failed to read args: {t}", .{err});
        return err;
    };
    defer args_iter.deinit();

    const config = parseArgs(&args_iter) catch |err| {
        if (err == error.ShowHelp) return;
        std.log.err("Invalid arguments: {t}", .{err});
        printUsage();
        return err;
    };

    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    try buildSite(allocator, io, config);
}

fn parseArgs(args: *std.process.Args.Iterator) ArgsError!BuildConfig {
    var config = BuildConfig{
        .source_dir = "docs",
        .out_dir = "zig-out/docs",
        .manifest_path = "docs/site.json",
    };

    _ = args.skip();
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--source")) {
            const value = args.next() orelse return error.MissingValue;
            config.source_dir = value;
        } else if (std.mem.eql(u8, arg, "--out")) {
            const value = args.next() orelse return error.MissingValue;
            config.out_dir = value;
        } else if (std.mem.eql(u8, arg, "--manifest")) {
            const value = args.next() orelse return error.MissingValue;
            config.manifest_path = value;
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

fn buildSite(allocator: std.mem.Allocator, io: std.Io, config: BuildConfig) !void {
    const manifest_data = try std.Io.Dir.cwd().readFileAlloc(
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

    try std.Io.Dir.cwd().createDirPath(io, config.out_dir);
    try copyAssets(allocator, io, config.source_dir, config.out_dir);

    for (manifest.pages) |page| {
        try writePage(allocator, io, manifest, config, page);
    }
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
    try writeFile(io, dest, content);
}

fn writePage(
    allocator: std.mem.Allocator,
    io: std.Io,
    manifest: Manifest,
    config: BuildConfig,
    page: PageConfig,
) !void {
    const source_path = try std.fs.path.join(allocator, &.{ config.source_dir, page.source });
    defer allocator.free(source_path);
    const page_body = try std.Io.Dir.cwd().readFileAlloc(io, source_path, allocator, .limited(10 * 1024 * 1024));
    defer allocator.free(page_body);

    const file_name = try std.fmt.allocPrint(allocator, "{s}.html", .{page.slug});
    defer allocator.free(file_name);
    const output_path = try std.fs.path.join(allocator, &.{ config.out_dir, file_name });
    defer allocator.free(output_path);

    const nav_html = try buildNav(allocator, manifest, page.slug);
    defer allocator.free(nav_html);

    var html = std.ArrayList(u8).init(allocator);
    defer html.deinit();
    const writer = html.writer();

    try writeDocument(writer, manifest, page, nav_html, page_body);

    const rendered = try html.toOwnedSlice();
    defer allocator.free(rendered);

    try writeFile(io, output_path, rendered);
}

fn writeDocument(
    writer: anytype,
    manifest: Manifest,
    page: PageConfig,
    nav_html: []const u8,
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
    try writer.writeAll("<div class=\"layout\">\n");
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
    try writer.writeAll("</div>\n");
    try writer.writeAll("<script src=\"");
    try writeAssetPath(writer, manifest.site.base_url, "assets/main.js");
    try writer.writeAll("\"></script>\n");
    try writer.writeAll("</body>\n</html>\n");
}

fn buildNav(allocator: std.mem.Allocator, manifest: Manifest, active_slug: []const u8) ![]const u8 {
    var nav = std.ArrayList(u8).init(allocator);
    const writer = nav.writer();

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

fn writeFile(io: std.Io, path: []const u8, content: []const u8) !void {
    var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    defer file.close(io);
    try file.writer(io).writeAll(content);
}
