const std = @import("std");
const model = @import("model.zig");

pub const skill_footer =
    \\## Zig Skill
    \\Use the `$zig` Codex skill for ABI Zig 0.16-dev syntax updates, modular build graph guidance, and targeted validation workflows.
;

pub fn render(
    allocator: std.mem.Allocator,
    modules: []const model.ModuleDoc,
    outputs: *std.ArrayListUnmanaged(model.OutputFile),
) !void {
    try renderIndex(allocator, modules, outputs);
    for (modules) |mod| {
        try renderModule(allocator, mod, outputs);
    }
}

fn renderIndex(
    allocator: std.mem.Allocator,
    modules: []const model.ModuleDoc,
    outputs: *std.ArrayListUnmanaged(model.OutputFile),
) !void {
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(allocator);

    try buf.appendSlice(allocator,
        \\# ABI Framework API Reference
        \\
        \\> Comprehensive API documentation auto-generated from source code.
        \\
        \\---
        \\
        \\## Quick Links
        \\
    );
    try model.appendTableHeader(allocator, &buf, &.{ "Module", "Category", "Description", "Build Flag" });

    for (modules) |mod| {
        const row = [_][]const u8{
            try std.fmt.allocPrint(allocator, "[{s}]({s}.md)", .{ mod.name, mod.name }),
            mod.category.name(),
            summary(mod.description),
            try std.fmt.allocPrint(allocator, "`{s}`", .{mod.build_flag}),
        };
        defer allocator.free(row[0]);
        defer allocator.free(row[3]);
        try model.appendTableRow(allocator, &buf, &row);
    }

    try buf.appendSlice(allocator, "\n---\n\n");

    const categories = [_]model.Category{ .core, .compute, .ai, .data, .infrastructure, .utilities };
    for (categories) |category| {
        var has = false;
        for (modules) |mod| {
            if (mod.category != category) continue;
            if (!has) {
                has = true;
                try appendFmt(allocator, &buf, "## {s}\n\n", .{category.name()});
            }
            try appendFmt(allocator, &buf, "### [{s}]({s}.md)\n\n", .{ mod.name, mod.name });
            if (mod.description.len > 0) {
                try appendFmt(allocator, &buf, "{s}\n\n", .{mod.description});
            }
            try appendFmt(allocator, &buf, "**Source:** [`{s}`](../../{s})", .{ mod.path, mod.path });
            if (!std.mem.eql(u8, mod.build_flag, "always-on")) {
                try appendFmt(allocator, &buf, " | **Flag:** `-D{s}`", .{mod.build_flag});
            }
            try buf.appendSlice(allocator, "\n\n");
        }
    }

    try buf.appendSlice(allocator,
        \\---
        \\
        \\*Generated automatically by `zig build gendocs`*
        \\
        \\
    );
    try buf.appendSlice(allocator, skill_footer);
    try buf.append(allocator, '\n');

    try model.pushOutput(allocator, outputs, "docs/api/index.md", buf.items);
}

fn renderModule(
    allocator: std.mem.Allocator,
    mod: model.ModuleDoc,
    outputs: *std.ArrayListUnmanaged(model.OutputFile),
) !void {
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(allocator);

    try appendFmt(allocator, &buf, "# {s}\n\n", .{mod.name});
    if (mod.description.len > 0) {
        try appendFmt(allocator, &buf, "> {s}\n\n", .{mod.description});
    }

    try appendFmt(allocator, &buf, "**Source:** [`{s}`](../../{s})\n\n", .{ mod.path, mod.path });
    if (!std.mem.eql(u8, mod.build_flag, "always-on")) {
        try appendFmt(allocator, &buf, "**Build flag:** `-D{s}=true`\n\n", .{mod.build_flag});
    } else {
        try buf.appendSlice(allocator, "**Availability:** Always enabled\n\n");
    }

    try buf.appendSlice(allocator, "---\n\n## API\n\n");

    if (mod.symbols.len == 0) {
        try buf.appendSlice(allocator, "No documented public symbols were discovered.\n\n");
    } else {
        for (mod.symbols) |symbol| {
            try appendFmt(allocator, &buf, "### <a id=\"{s}\"></a>`{s}`\n\n", .{ symbol.anchor, symbol.signature });
            try appendFmt(allocator, &buf, "<sup>**{s}**</sup> | [source](../../{s}#L{d})\n\n", .{
                symbol.kind.badge(),
                mod.path,
                symbol.line,
            });
            try appendFmt(allocator, &buf, "{s}\n\n", .{symbol.doc});
        }
    }

    try buf.appendSlice(allocator,
        \\---
        \\
        \\*Generated automatically by `zig build gendocs`*
        \\
        \\
    );
    try buf.appendSlice(allocator, skill_footer);
    try buf.append(allocator, '\n');

    const out_path = try std.fmt.allocPrint(allocator, "docs/api/{s}.md", .{mod.name});
    defer allocator.free(out_path);
    try model.pushOutput(allocator, outputs, out_path, buf.items);
}

fn summary(text: []const u8) []const u8 {
    const one_line = model.lineSummary(text);
    if (one_line.len == 0) return "—";
    return one_line;
}

fn appendFmt(
    allocator: std.mem.Allocator,
    buf: *std.ArrayListUnmanaged(u8),
    comptime fmt: []const u8,
    args: anytype,
) !void {
    const s = try std.fmt.allocPrint(allocator, fmt, args);
    defer allocator.free(s);
    try buf.appendSlice(allocator, s);
}
