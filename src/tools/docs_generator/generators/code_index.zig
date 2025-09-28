const std = @import("std");

const Declaration = struct {
    name: []u8,
    kind: []u8,
    signature: []u8,
    doc: []u8,
};

fn docPathLessThan(_: void, lhs: []const u8, rhs: []const u8) bool {
    return std.mem.lessThan(u8, lhs, rhs);
}

pub fn generateCodeApiIndex(allocator: std.mem.Allocator) !void {
    // Use an arena for all temporary allocations in scanning to avoid leaks and simplify ownership
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var files = std.ArrayListUnmanaged([]const u8){};
    defer files.deinit(a);

    try collectZigFiles(a, "src", &files);
    std.mem.sort([]const u8, files.items, {}, struct {
        fn lessThan(_: void, lhs: []const u8, rhs: []const u8) bool {
            return std.mem.lessThan(u8, lhs, rhs);
        }
    }.lessThan);

    std.sort.block([]const u8, files.items, {}, docPathLessThan);

    var out = try std.fs.cwd().createFile("docs/generated/CODE_API_INDEX.md", .{ .truncate = true });
    defer out.close();

    const writef = struct {
        fn go(file: std.fs.File, alloc2: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !void {
            const s = try std.fmt.allocPrint(alloc2, fmt, args);
            defer alloc2.free(s);
            try file.writeAll(s);
        }
    }.go;

    try writef(out, a, "# Code API Index (Scanned)\n\n", .{});
    try writef(out, a, "Scanned {d} Zig files under `src/`. This index lists public declarations discovered along with leading doc comments.\n\n", .{files.items.len});

    var decls = std.ArrayListUnmanaged(Declaration){};
    defer decls.deinit(a);

    for (files.items) |rel| {
        decls.clearRetainingCapacity();
        try scanFile(a, rel, &decls);
        if (decls.items.len == 0) continue;

        try writef(out, a, "## {s}\n\n", .{rel});
        for (decls.items) |d| {
            try writef(out, a, "- {s} `{s}`\n\n", .{ d.kind, d.name });
            if (d.doc.len > 0) {
                try writef(out, a, "{s}\n\n", .{d.doc});
            }
            if (d.signature.len > 0) {
                try writef(out, a, "```zig\n{s}\n```\n\n", .{d.signature});
            }
        }
    }
}

fn collectZigFiles(allocator: std.mem.Allocator, dir_path: []const u8, out_files: *std.ArrayListUnmanaged([]const u8)) !void {
    var stack = std.ArrayListUnmanaged([]u8){};
    defer {
        for (stack.items) |p| allocator.free(p);
        stack.deinit(allocator);
    }
    try stack.append(allocator, try allocator.dupe(u8, dir_path));

    while (stack.items.len > 0) {
        const idx = stack.items.len - 1;
        const path = stack.items[idx];
        _ = stack.pop();
        defer allocator.free(path);

        var dir = std.fs.cwd().openDir(path, .{ .iterate = true }) catch continue;
        defer dir.close();

        var it = dir.iterate();
        while (it.next() catch null) |entry| {
            if (entry.kind == .file) {
                if (std.mem.endsWith(u8, entry.name, ".zig")) {
                    const rel = try std.fs.path.join(allocator, &[_][]const u8{ path, entry.name });
                    try out_files.append(allocator, rel);
                }
            } else if (entry.kind == .directory) {
                if (std.mem.eql(u8, entry.name, ".") or std.mem.eql(u8, entry.name, "..")) continue;
                const sub = try std.fs.path.join(allocator, &[_][]const u8{ path, entry.name });
                try stack.append(allocator, sub);
            }
        }
    }
}

fn scanFile(allocator: std.mem.Allocator, rel_path: []const u8, decls: *std.ArrayListUnmanaged(Declaration)) !void {
    const file = try std.fs.cwd().openFile(rel_path, .{});
    defer file.close();

    const max_bytes: usize = 16 * 1024 * 1024;
    const data = try std.fs.cwd().readFileAlloc(allocator, rel_path, max_bytes);
    defer allocator.free(data);

    var it = std.mem.splitScalar(u8, data, '\n');

    var doc_buf: std.ArrayListUnmanaged(u8) = .empty;
    defer doc_buf.deinit(allocator);

    while (it.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (std.mem.startsWith(u8, trimmed, "///")) {
            // accumulate doc lines
            const doc_line = std.mem.trim(u8, trimmed[3..], " \t");
            try doc_buf.appendSlice(allocator, doc_line);
            try doc_buf.append(allocator, '\n');
            continue;
        }

        // Identify public declarations after doc comments
        if (isPubDecl(trimmed)) {
            const kind = detectKind(trimmed);
            const name = extractName(allocator, trimmed) catch {
                // reset doc buffer and continue
                doc_buf.clearRetainingCapacity();
                continue;
            };
            const sig = try allocator.dupe(u8, trimmed);
            const doc = try allocator.dupe(u8, doc_buf.items);
            doc_buf.clearRetainingCapacity();

            try decls.append(allocator, .{
                .name = name,
                .kind = try allocator.dupe(u8, kind),
                .signature = sig,
                .doc = doc,
            });
            continue;
        } else {
            // reset doc buffer if we encounter a non-doc, non-decl line
            if (trimmed.len > 0 and !std.mem.startsWith(u8, trimmed, "//")) {
                doc_buf.clearRetainingCapacity();
            }
        }
    }
}

fn isPubDecl(line: []const u8) bool {
    // consider pub fn/const/var/type usingnamespace
    if (!std.mem.startsWith(u8, line, "pub ")) return false;
    return std.mem.indexOfAny(u8, line[4..], "fctuv") != null // quick filter
    or std.mem.startsWith(u8, line, "pub usingnamespace") or std.mem.indexOf(u8, line, " struct") != null or std.mem.indexOf(u8, line, " enum") != null;
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

fn extractName(allocator: std.mem.Allocator, line: []const u8) ![]u8 {
    // naive name extraction: after `pub fn|const|var` read identifier
    var rest: []const u8 = line;
    if (std.mem.startsWith(u8, rest, "pub fn ")) rest = rest[7..] else if (std.mem.startsWith(u8, rest, "pub const ")) rest = rest[10..] else if (std.mem.startsWith(u8, rest, "pub var ")) rest = rest[8..] else if (std.mem.startsWith(u8, rest, "pub usingnamespace ")) rest = rest[18..] else if (std.mem.startsWith(u8, rest, "pub ")) rest = rest[4..];

    // identifier: letters, digits, underscore
    var i: usize = 0;
    while (i < rest.len) : (i += 1) {
        const c = rest[i];
        const is_id = (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or (c >= '0' and c <= '9') or (c == '_') or (c == '.');
        if (!is_id) break;
    }
    const ident = std.mem.trim(u8, rest[0..i], " \t");
    if (ident.len == 0) return error.Invalid;
    return allocator.dupe(u8, ident);
}
