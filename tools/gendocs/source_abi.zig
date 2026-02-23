const std = @import("std");
const model = @import("model.zig");

pub fn discoverModules(allocator: std.mem.Allocator, io: std.Io, cwd: std.Io.Dir) ![]model.ModuleDoc {
    const source = try cwd.readFileAlloc(io, "src/abi.zig", allocator, .limited(4 * 1024 * 1024));
    defer allocator.free(source);

    var modules = std.ArrayListUnmanaged(model.ModuleDoc).empty;
    errdefer {
        for (modules.items) |mod| mod.deinit(allocator);
        modules.deinit(allocator);
    }

    var lines = std.mem.splitScalar(u8, source, '\n');
    var line_buf = std.ArrayListUnmanaged(u8).empty;
    defer line_buf.deinit(allocator);

    var doc_buf = std.ArrayListUnmanaged(u8).empty;
    defer doc_buf.deinit(allocator);

    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \t\r");

        if (std.mem.startsWith(u8, line, "///")) {
            const content = std.mem.trimStart(u8, if (line.len > 3) line[3..] else "", " ");
            if (doc_buf.items.len > 0) try doc_buf.append(allocator, '\n');
            try doc_buf.appendSlice(allocator, content);
            continue;
        }

        if (std.mem.startsWith(u8, line, "pub const ")) {
            line_buf.clearRetainingCapacity();
            try line_buf.appendSlice(allocator, line);

            while (!endsDeclaration(line_buf.items)) {
                const continuation = lines.next() orelse break;
                const trimmed = std.mem.trim(u8, continuation, " \t\r");
                if (trimmed.len == 0) continue;
                try line_buf.append(allocator, ' ');
                try line_buf.appendSlice(allocator, trimmed);
            }

            const declaration = line_buf.items;
            const module_info = try parseModuleDeclaration(allocator, declaration, doc_buf.items);
            if (module_info) |mod| {
                try modules.append(allocator, mod);
            }
            doc_buf.clearRetainingCapacity();
            continue;
        }

        if (line.len > 0 and !std.mem.startsWith(u8, line, "//")) {
            doc_buf.clearRetainingCapacity();
        }
    }

    insertionSortModules(modules.items);
    return try modules.toOwnedSlice(allocator);
}

fn parseModuleDeclaration(
    allocator: std.mem.Allocator,
    declaration: []const u8,
    docs: []const u8,
) !?model.ModuleDoc {
    const after_const = declaration["pub const ".len..];
    const eq_pos = std.mem.indexOf(u8, after_const, " = ") orelse return null;
    const name = std.mem.trim(u8, after_const[0..eq_pos], " \t");
    const rhs = after_const[eq_pos + 3 ..];

    const import_path = parseImportPath(rhs) orelse return null;
    if (std.mem.eql(u8, import_path, "build_options") or
        std.mem.eql(u8, import_path, "builtin") or
        std.mem.eql(u8, import_path, "std"))
    {
        return null;
    }

    const build_flag = parseBuildFlag(rhs) orelse "always-on";
    const full_path = try std.fmt.allocPrint(allocator, "src/{s}", .{import_path});
    errdefer allocator.free(full_path);

    const parse_result = parseModuleFile(allocator, full_path) catch {
        return .{
            .name = try allocator.dupe(u8, name),
            .path = full_path,
            .description = try allocator.dupe(u8, std.mem.trim(u8, docs, " \t\r\n")),
            .category = categorizeByPath(import_path, name),
            .build_flag = try allocator.dupe(u8, build_flag),
            .symbols = try allocator.dupe(model.SymbolDoc, &.{}),
        };
    };

    // Use /// doc from abi.zig if available, otherwise fall back to //! module doc
    const trimmed_docs = std.mem.trim(u8, docs, " \t\r\n");
    const description = if (trimmed_docs.len > 0) blk: {
        if (parse_result.module_doc.len > 0) allocator.free(parse_result.module_doc);
        break :blk try allocator.dupe(u8, trimmed_docs);
    } else if (parse_result.module_doc.len > 0) blk: {
        break :blk parse_result.module_doc;
    } else blk: {
        break :blk try allocator.dupe(u8, "");
    };

    return .{
        .name = try allocator.dupe(u8, name),
        .path = full_path,
        .description = description,
        .category = categorizeByPath(import_path, name),
        .build_flag = try allocator.dupe(u8, build_flag),
        .symbols = parse_result.symbols,
    };
}

fn parseBuildFlag(rhs: []const u8) ?[]const u8 {
    const needle = "if (build_options.";
    const start = std.mem.indexOf(u8, rhs, needle) orelse return null;
    const rest = rhs[start + needle.len ..];
    const end = std.mem.indexOfScalar(u8, rest, ')') orelse return null;
    return std.mem.trim(u8, rest[0..end], " \t");
}

fn parseImportPath(rhs: []const u8) ?[]const u8 {
    const import_start = std.mem.indexOf(u8, rhs, "@import(\"") orelse return null;
    const tail = rhs[import_start + "@import(\"".len ..];
    const import_end = std.mem.indexOfScalar(u8, tail, '"') orelse return null;
    return tail[0..import_end];
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

fn categorizeByPath(path: []const u8, name: []const u8) model.Category {
    if (std.mem.startsWith(u8, path, "features/ai")) {
        return .ai;
    }

    if (std.mem.startsWith(u8, path, "features/gpu") or
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

test "parseBuildFlag extracts enable flag" {
    const flag = parseBuildFlag("if (build_options.enable_gpu) @import(\"features/gpu/mod.zig\")") orelse "";
    try std.testing.expectEqualStrings("enable_gpu", flag);
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
    try std.testing.expect(endsDeclaration("pub const gpu = if (build_options.enable_gpu) @import(\"features/gpu/mod.zig\") else @import(\"features/gpu/stub.zig\");"));
}
