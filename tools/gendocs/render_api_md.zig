const std = @import("std");
const model = @import("model.zig");

pub const skill_footer = model.generated_footer;

pub fn render(
    allocator: std.mem.Allocator,
    modules: []const model.ModuleDoc,
    outputs: *std.ArrayListUnmanaged(model.OutputFile),
) !void {
    try renderIndex(allocator, modules, outputs);
    for (modules) |mod| {
        try renderModule(allocator, mod, outputs);
    }
    try renderCoverage(allocator, modules, outputs);
}

fn renderCoverage(
    allocator: std.mem.Allocator,
    modules: []const model.ModuleDoc,
    outputs: *std.ArrayListUnmanaged(model.OutputFile),
) !void {
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(allocator);

    try buf.appendSlice(allocator,
        \\# API Documentation Coverage
        \\
        \\> Per-module documentation coverage of public symbols.
        \\
        \\---
        \\
    );
    try model.appendTableHeader(allocator, &buf, &.{ "Module", "Documented", "Total", "Coverage" });

    var total_documented: usize = 0;
    var total_symbols: usize = 0;

    for (modules) |mod| {
        const documented = countDocumented(mod.symbols);
        const total = mod.symbols.len;
        total_documented += documented;
        total_symbols += total;

        const pct = if (total > 0) (documented * 100) / total else 100;
        const marker: []const u8 = if (pct < 50) " ⚠" else "";

        const pct_cell = try std.fmt.allocPrint(allocator, "{d}%{s}", .{ pct, marker });
        defer allocator.free(pct_cell);
        const doc_cell = try std.fmt.allocPrint(allocator, "{d}", .{documented});
        defer allocator.free(doc_cell);
        const total_cell = try std.fmt.allocPrint(allocator, "{d}", .{total});
        defer allocator.free(total_cell);

        const link_cell = try std.fmt.allocPrint(allocator, "[{s}]({s}.md)", .{ mod.name, mod.name });
        defer allocator.free(link_cell);
        try model.appendTableRow(allocator, &buf, &.{ link_cell, doc_cell, total_cell, pct_cell });
    }

    const overall_pct = if (total_symbols > 0) (total_documented * 100) / total_symbols else 100;
    try appendFmt(allocator, &buf, "\n**Overall: {d}/{d} symbols documented ({d}%)**\n", .{ total_documented, total_symbols, overall_pct });

    try buf.appendSlice(allocator, model.generated_footer);

    try model.pushOutput(allocator, outputs, "docs/api/coverage.md", buf.items);

    // Print summary to stderr for build visibility
    std.debug.print("API doc coverage: {d}/{d} symbols ({d}%)\n", .{ total_documented, total_symbols, overall_pct });
}

fn countDocumented(symbols: []const model.SymbolDoc) usize {
    var count: usize = 0;
    for (symbols) |sym| {
        if (sym.doc.len > 0) count += 1;
    }
    return count;
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

    try buf.appendSlice(allocator, skill_footer);

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

    try buf.appendSlice(allocator, skill_footer);

    const out_path = try std.fmt.allocPrint(allocator, "docs/api/{s}.md", .{mod.name});
    defer allocator.free(out_path);
    try model.pushOutput(allocator, outputs, out_path, buf.items);
}

fn summary(text: []const u8) []const u8 {
    const one_line = model.lineSummary(text);
    if (one_line.len == 0) return "—";
    return one_line;
}

const appendFmt = model.appendFmt;
