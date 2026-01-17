//! API Documentation Generator
//!
//! Generates markdown documentation from Zig doc comments.
//! Usage: zig build gendocs

const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const output_dir = "docs/api";

    // Ensure output directory exists
    std.fs.cwd().makePath(output_dir) catch |err| {
        std.debug.print("Warning: Could not create output directory {s}: {}\n", .{ output_dir, err });
    };

    // Generate docs for main modules
    const modules = [_][]const u8{
        "src/abi.zig",
        "src/compute/gpu/unified.zig",
        "src/features/ai/mod.zig",
        "src/features/database/mod.zig",
        "src/features/network/mod.zig",
        "src/compute/runtime/mod.zig",
    };

    var success_count: usize = 0;
    for (modules) |module| {
        if (generateModuleDoc(allocator, module, output_dir)) {
            success_count += 1;
        } else |err| {
            std.debug.print("Warning: Failed to generate docs for {s}: {}\n", .{ module, err });
        }
    }

    std.debug.print("Documentation generated: {}/{} modules in {s}/\n", .{ success_count, modules.len, output_dir });
}

fn generateModuleDoc(allocator: std.mem.Allocator, module_path: []const u8, output_dir: []const u8) !void {
    // Read source file
    const source = std.fs.cwd().readFileAlloc(allocator, module_path, 1024 * 1024) catch |err| {
        return err;
    };
    defer allocator.free(source);

    // Extract module name
    const basename = std.fs.path.basename(module_path);
    const name = std.mem.sliceTo(basename, '.');

    // Create output file
    const output_path = try std.fmt.allocPrint(allocator, "{s}/{s}.md", .{ output_dir, name });
    defer allocator.free(output_path);

    var file = try std.fs.cwd().createFile(output_path, .{});
    defer file.close();

    var writer = file.writer();

    // Write header
    try writer.print("# {s} Module\n\n", .{name});
    try writer.print("**Source:** `{s}`\n\n", .{module_path});

    // Extract and write doc comments
    try extractDocComments(source, writer);

    // Extract and document public declarations
    try extractPublicDeclarations(source, writer);
}

fn extractDocComments(source: []const u8, writer: anytype) !void {
    var lines = std.mem.splitScalar(u8, source, '\n');
    var in_module_doc = false;
    var wrote_overview = false;

    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");

        if (std.mem.startsWith(u8, trimmed, "//!")) {
            if (!wrote_overview) {
                try writer.writeAll("## Overview\n\n");
                wrote_overview = true;
            }
            // Module-level doc comment
            const content = if (trimmed.len > 3) trimmed[3..] else "";
            const clean = std.mem.trim(u8, content, " ");
            if (clean.len > 0) {
                try writer.print("{s}\n", .{clean});
            } else {
                try writer.writeAll("\n");
            }
            in_module_doc = true;
        } else if (in_module_doc and trimmed.len == 0) {
            try writer.writeAll("\n");
        } else if (!std.mem.startsWith(u8, trimmed, "//!")) {
            if (in_module_doc) {
                try writer.writeAll("\n");
                in_module_doc = false;
            }
        }
    }
}

fn extractPublicDeclarations(source: []const u8, writer: anytype) !void {
    var lines = std.mem.splitScalar(u8, source, '\n');
    var doc_buffer: [4096]u8 = undefined;
    var doc_len: usize = 0;
    var found_any = false;

    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");

        // Collect item doc comments
        if (std.mem.startsWith(u8, trimmed, "///")) {
            const content = if (trimmed.len > 3) trimmed[3..] else "";
            const clean = std.mem.trim(u8, content, " ");
            if (doc_len + clean.len + 1 < doc_buffer.len) {
                if (doc_len > 0) {
                    doc_buffer[doc_len] = ' ';
                    doc_len += 1;
                }
                @memcpy(doc_buffer[doc_len .. doc_len + clean.len], clean);
                doc_len += clean.len;
            }
            continue;
        }

        // Check for public declarations
        if (std.mem.startsWith(u8, trimmed, "pub const ") or
            std.mem.startsWith(u8, trimmed, "pub fn ") or
            std.mem.startsWith(u8, trimmed, "pub var "))
        {
            if (!found_any) {
                try writer.writeAll("## Public API\n\n");
                found_any = true;
            }

            // Extract declaration name
            var decl_start: usize = 0;
            if (std.mem.startsWith(u8, trimmed, "pub const ")) {
                decl_start = 10;
            } else if (std.mem.startsWith(u8, trimmed, "pub fn ")) {
                decl_start = 7;
            } else if (std.mem.startsWith(u8, trimmed, "pub var ")) {
                decl_start = 8;
            }

            const rest = trimmed[decl_start..];
            var name_end: usize = 0;
            for (rest, 0..) |c, i| {
                if (c == ' ' or c == '(' or c == ':' or c == '=') {
                    name_end = i;
                    break;
                }
            }
            if (name_end == 0) name_end = rest.len;
            const decl_name = rest[0..name_end];

            // Write declaration
            try writer.print("### `{s}`\n\n", .{decl_name});

            if (doc_len > 0) {
                try writer.print("{s}\n\n", .{doc_buffer[0..doc_len]});
            }
        }

        // Reset doc buffer for non-doc lines
        if (!std.mem.startsWith(u8, trimmed, "///")) {
            doc_len = 0;
        }
    }
}
