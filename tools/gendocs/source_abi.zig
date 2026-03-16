const std = @import("std");
const module_catalog = @import("module_catalog");
const model = @import("model.zig");

pub fn discoverModules(allocator: std.mem.Allocator, io: std.Io, cwd: std.Io.Dir) ![]model.ModuleDoc {
    var modules = std.ArrayListUnmanaged(model.ModuleDoc).empty;
    errdefer {
        for (modules.items) |mod| mod.deinit(allocator);
        modules.deinit(allocator);
    }

    _ = io;
    _ = cwd;

    for (module_catalog.public_modules) |entry| {
        const parse_result = parseModuleFile(allocator, entry.path) catch {
            try modules.append(allocator, .{
                .name = try allocator.dupe(u8, entry.name),
                .path = try allocator.dupe(u8, entry.path),
                .description = try allocator.dupe(u8, entry.description),
                .category = categorizeByPath(trimSourcePrefix(entry.path), entry.name),
                .build_flag = try allocator.dupe(u8, entry.build_flag orelse "always-on"),
                .symbols = try allocator.dupe(model.SymbolDoc, &.{}),
            });
            continue;
        };

        const description = if (parse_result.module_doc.len > 0)
            parse_result.module_doc
        else
            try allocator.dupe(u8, entry.description);

        try modules.append(allocator, .{
            .name = try allocator.dupe(u8, entry.name),
            .path = try allocator.dupe(u8, entry.path),
            .description = description,
            .category = categorizeByPath(trimSourcePrefix(entry.path), entry.name),
            .build_flag = try allocator.dupe(u8, entry.build_flag orelse "always-on"),
            .symbols = parse_result.symbols,
        });
    }

    insertionSortModules(modules.items);
    return try modules.toOwnedSlice(allocator);
}

const ModuleParseResult = struct {
    symbols: []model.SymbolDoc,
    module_doc: []const u8,
};

fn extractModuleDoc(allocator: std.mem.Allocator, source: []const u8) ![]const u8 {
    var doc_buf = std.ArrayListUnmanaged(u8).empty;
    errdefer doc_buf.deinit(allocator);

    var lines = std.mem.splitScalar(u8, source, '\n');
    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \t\r");
        if (std.mem.startsWith(u8, line, "//!")) {
            const content = std.mem.trimStart(u8, if (line.len > 3) line[3..] else "", " ");
            if (doc_buf.items.len > 0) try doc_buf.append(allocator, '\n');
            try doc_buf.appendSlice(allocator, content);
            continue;
        }
        if (line.len == 0) continue;
        break;
    }

    if (doc_buf.items.len == 0) {
        doc_buf.deinit(allocator);
        return "";
    }
    return doc_buf.toOwnedSlice(allocator);
}

fn parseModuleFile(allocator: std.mem.Allocator, path: []const u8) !ModuleParseResult {
    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();
    const cwd = std.Io.Dir.cwd();

    const source = try cwd.readFileAlloc(io, path, allocator, .limited(4 * 1024 * 1024));
    defer allocator.free(source);

    const module_doc = try extractModuleDoc(allocator, source);
    const symbols = try parseSymbolsFromSource(allocator, source);
    return .{ .symbols = symbols, .module_doc = module_doc };
}

pub fn parseModuleSymbols(allocator: std.mem.Allocator, path: []const u8) ![]model.SymbolDoc {
    const result = try parseModuleFile(allocator, path);
    if (result.module_doc.len > 0) allocator.free(result.module_doc);
    return result.symbols;
}

fn parseSymbolsFromSource(allocator: std.mem.Allocator, source: []const u8) ![]model.SymbolDoc {
    var symbols = std.ArrayListUnmanaged(model.SymbolDoc).empty;
    errdefer {
        for (symbols.items) |sym| sym.deinit(allocator);
        symbols.deinit(allocator);
    }

    var lines = std.mem.splitScalar(u8, source, '\n');
    var line_no: usize = 0;

    var doc_buf = std.ArrayListUnmanaged(u8).empty;
    defer doc_buf.deinit(allocator);

    var sig_buf = std.ArrayListUnmanaged(u8).empty;
    defer sig_buf.deinit(allocator);

    while (lines.next()) |raw_line| {
        line_no += 1;
        const line = std.mem.trim(u8, raw_line, " \t\r");

        if (std.mem.startsWith(u8, line, "///")) {
            const content = std.mem.trimStart(u8, if (line.len > 3) line[3..] else "", " ");
            if (doc_buf.items.len > 0) try doc_buf.append(allocator, '\n');
            try doc_buf.appendSlice(allocator, content);
            continue;
        }

        if (std.mem.startsWith(u8, line, "pub ")) {
            sig_buf.clearRetainingCapacity();
            try sig_buf.appendSlice(allocator, line);
            const decl_line = line_no;

            while (!endsDeclaration(sig_buf.items)) {
                const continuation = lines.next() orelse break;
                line_no += 1;
                const trimmed = std.mem.trim(u8, continuation, " \t\r");
                if (trimmed.len == 0) continue;
                try sig_buf.append(allocator, ' ');
                try sig_buf.appendSlice(allocator, trimmed);
            }

            const signature = trimDeclSignature(sig_buf.items);
            if (signature.len > 0 and doc_buf.items.len > 0) {
                const doc_text = std.mem.trim(u8, doc_buf.items, " \t\r\n");
                const anchor = try slugify(allocator, signature);
                try symbols.append(allocator, .{
                    .signature = try allocator.dupe(u8, signature),
                    .doc = try allocator.dupe(u8, doc_text),
                    .kind = detectItemType(signature),
                    .line = decl_line,
                    .anchor = anchor,
                });
            }
            doc_buf.clearRetainingCapacity();
            continue;
        }

        if (line.len > 0 and !std.mem.startsWith(u8, line, "//")) {
            doc_buf.clearRetainingCapacity();
        }
    }

    insertionSortSymbols(symbols.items);
    return try symbols.toOwnedSlice(allocator);
}

fn endsDeclaration(decl: []const u8) bool {
    var depth: usize = 0;
    for (decl) |ch| {
        switch (ch) {
            '(' => depth += 1,
            ')' => depth -|= 1,
            else => {},
        }
    }

    if (depth != 0) return false;

    const trimmed = std.mem.trim(u8, decl, " \t\r\n");
    if (trimmed.len == 0) return false;
    const last = trimmed[trimmed.len - 1];
    if (last == ';' or last == '{') return true;
    if (std.mem.indexOf(u8, trimmed, " = ") != null and last != ',' and last != '.') return true;

    return false;
}

fn trimDeclSignature(line: []const u8) []const u8 {
    var end = line.len;
    var depth: usize = 0;

    for (line, 0..) |c, i| {
        switch (c) {
            '(' => depth += 1,
            ')' => depth -|= 1,
            else => {},
        }
        if (depth == 0) {
            if (c == '{' or c == ';') {
                end = i;
                break;
            }
            if (c == '=' and i + 1 < line.len and line[i + 1] != '>') {
                end = i;
                break;
            }
        }
    }

    return std.mem.trim(u8, line[0..end], " \t\r\n");
}

fn detectItemType(line: []const u8) model.SymbolKind {
    if (std.mem.indexOf(u8, line, "pub fn ") != null) return .function;
    if (std.mem.indexOf(u8, line, "pub var ") != null) return .variable;

    if (std.mem.indexOf(u8, line, "pub const ") != null) {
        if (std.mem.indexOf(u8, line, " struct") != null or
            std.mem.indexOf(u8, line, " enum") != null or
            std.mem.indexOf(u8, line, " union") != null or
            std.mem.indexOf(u8, line, " @import") != null)
        {
            return .type_def;
        }
        return .constant;
    }

    return .constant;
}

fn slugify(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    var prev_dash = false;
    for (text) |ch| {
        if (std.ascii.isAlphanumeric(ch)) {
            try out.append(allocator, std.ascii.toLower(ch));
            prev_dash = false;
        } else if (!prev_dash) {
            try out.append(allocator, '-');
            prev_dash = true;
        }
    }

    while (out.items.len > 0 and out.items[out.items.len - 1] == '-') {
        _ = out.pop();
    }

    if (out.items.len == 0) {
        try out.appendSlice(allocator, "symbol");
    }

    return out.toOwnedSlice(allocator);
}

fn trimSourcePrefix(path: []const u8) []const u8 {
    if (std.mem.startsWith(u8, path, "src/")) return path["src/".len..];
    return path;
}

fn categorizeByPath(path: []const u8, name: []const u8) model.Category {
    if (std.mem.startsWith(u8, path, "features/ai")) {
        return .ai;
    }

    if (std.mem.startsWith(u8, path, "features/gpu") or
        std.mem.startsWith(u8, path, "inference/") or
        std.mem.eql(u8, name, "runtime") or
        std.mem.eql(u8, name, "simd") or
        std.mem.eql(u8, name, "benchmarks"))
    {
        return .compute;
    }

    if (std.mem.eql(u8, name, "database") or
        std.mem.eql(u8, name, "cache") or
        std.mem.eql(u8, name, "storage") or
        std.mem.eql(u8, name, "search"))
    {
        return .data;
    }

    if (std.mem.startsWith(u8, path, "core/") or
        std.mem.eql(u8, name, "config") or
        std.mem.eql(u8, name, "framework") or
        std.mem.eql(u8, name, "errors") or
        std.mem.eql(u8, name, "registry"))
    {
        return .core;
    }

    if (std.mem.eql(u8, name, "network") or
        std.mem.eql(u8, name, "web") or
        std.mem.eql(u8, name, "cloud") or
        std.mem.eql(u8, name, "gateway") or
        std.mem.eql(u8, name, "pages") or
        std.mem.eql(u8, name, "messaging") or
        std.mem.eql(u8, name, "observability") or
        std.mem.eql(u8, name, "ha") or
        std.mem.eql(u8, name, "mcp") or
        std.mem.eql(u8, name, "acp") or
        std.mem.eql(u8, name, "mobile"))
    {
        return .infrastructure;
    }

    return .utilities;
}

fn insertionSortModules(items: []model.ModuleDoc) void {
    var i: usize = 1;
    while (i < items.len) : (i += 1) {
        const value = items[i];
        var j = i;
        while (j > 0 and model.compareModules({}, value, items[j - 1])) : (j -= 1) {
            items[j] = items[j - 1];
        }
        items[j] = value;
    }
}

fn insertionSortSymbols(items: []model.SymbolDoc) void {
    var i: usize = 1;
    while (i < items.len) : (i += 1) {
        const value = items[i];
        var j = i;
        while (j > 0 and model.compareSymbols({}, value, items[j - 1])) : (j -= 1) {
            items[j] = items[j - 1];
        }
        items[j] = value;
    }
}

test "trimDeclSignature handles assignment and braces" {
    try std.testing.expectEqualStrings("pub fn hello(x: usize) void", trimDeclSignature("pub fn hello(x: usize) void {"));
    try std.testing.expectEqualStrings("pub const Name", trimDeclSignature("pub const Name = \"x\";"));
}

test "slugify is stable" {
    const slug = try slugify(std.testing.allocator, "pub fn helloWorld(x: usize) !void");
    defer std.testing.allocator.free(slug);
    try std.testing.expect(std.mem.startsWith(u8, slug, "pub-fn-helloworld"));
}

test "endsDeclaration handles multiline function and const init" {
    try std.testing.expect(!endsDeclaration("pub fn render(\n"));
    try std.testing.expect(endsDeclaration("pub fn render(a: usize) void {"));
    try std.testing.expect(endsDeclaration("pub const gpu = if (build_options.feat_gpu) @import(\"features/gpu/mod.zig\") else @import(\"features/gpu/stub.zig\");"));
}

test "trimSourcePrefix drops src prefix when present" {
    try std.testing.expectEqualStrings("services/runtime/mod.zig", trimSourcePrefix("src/services/runtime/mod.zig"));
    try std.testing.expectEqualStrings("services/runtime/mod.zig", trimSourcePrefix("services/runtime/mod.zig"));
}
