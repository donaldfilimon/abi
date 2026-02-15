//! Automatic API Documentation Generator
//!
//! Auto-discovers modules from `src/abi.zig` and generates markdown API
//! documentation from Zig doc comments. No hardcoded module list needed —
//! adding a new feature to `abi.zig` automatically generates its docs.
//!
//! ## Usage
//! ```bash
//! zig build gendocs
//! ```
//!
//! ## Output
//! - `docs/api/index.md` — Categorized module index
//! - `docs/api/<module>.md` — Per-module API reference

const std = @import("std");

// =============================================================================
// Types
// =============================================================================

const Module = struct {
    name: []const u8,
    path: []const u8,
    doc_comment: []const u8,
    category: Category,
    build_flag: []const u8,

    const Category = enum {
        core,
        compute,
        ai,
        data,
        infrastructure,
        utilities,
    };

    fn categoryName(self: Module) []const u8 {
        return switch (self.category) {
            .core => "Core Framework",
            .compute => "Compute & Runtime",
            .ai => "AI & Machine Learning",
            .data => "Data & Storage",
            .infrastructure => "Infrastructure",
            .utilities => "Utilities",
        };
    }

    fn categoryOrder(self: Module) u8 {
        return switch (self.category) {
            .core => 0,
            .compute => 1,
            .ai => 2,
            .data => 3,
            .infrastructure => 4,
            .utilities => 5,
        };
    }
};

const DocItem = struct {
    signature: []const u8,
    doc: []const u8,
    item_type: ItemType,

    const ItemType = enum {
        function,
        constant,
        type_def,
        variable,
    };
};

// =============================================================================
// Module Auto-Discovery
// =============================================================================

fn discoverModules(allocator: std.mem.Allocator, io: std.Io, cwd: std.Io.Dir) ![]Module {
    const source = try cwd.readFileAlloc(io, "src/abi.zig", allocator, .limited(2 * 1024 * 1024));
    defer allocator.free(source);

    var modules = std.ArrayListUnmanaged(Module).empty;
    errdefer modules.deinit(allocator);

    var lines = std.mem.splitScalar(u8, source, '\n');
    var prev_doc: []const u8 = "";

    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \r\t");

        // Track doc comments (/// lines before pub const)
        if (std.mem.startsWith(u8, line, "///")) {
            prev_doc = if (line.len > 4) line[4..] else "";
            continue;
        }

        // Pattern 1: pub const X = if (build_options.enable_Y) @import(...)
        if (std.mem.startsWith(u8, line, "pub const ")) {
            const after_const = line["pub const ".len..];

            // Extract name
            const eq_pos = std.mem.indexOf(u8, after_const, " = ") orelse {
                prev_doc = "";
                continue;
            };
            const name = after_const[0..eq_pos];
            const rhs = after_const[eq_pos + 3 ..];

            // Comptime-gated: if (build_options.enable_X)
            if (std.mem.startsWith(u8, rhs, "if (build_options.enable_")) {
                // Extract the build flag
                const flag_start = "if (build_options.".len;
                const flag_end = std.mem.indexOfScalar(u8, rhs[flag_start..], ')') orelse {
                    prev_doc = "";
                    continue;
                };
                const build_flag = rhs[flag_start .. flag_start + flag_end];

                // Extract path from @import
                const import_start = std.mem.indexOf(u8, rhs, "@import(\"") orelse {
                    prev_doc = "";
                    continue;
                };
                const path_start = import_start + "@import(\"".len;
                const path_end = std.mem.indexOfScalar(u8, rhs[path_start..], '"') orelse {
                    prev_doc = "";
                    continue;
                };
                const rel_path = rhs[path_start .. path_start + path_end];

                // Build full path
                const full_path = std.fmt.allocPrint(allocator, "src/{s}", .{rel_path}) catch {
                    prev_doc = "";
                    continue;
                };

                const category = categorizeByPath(rel_path, name);

                try modules.append(allocator, .{
                    .name = try allocator.dupe(u8, name),
                    .path = full_path,
                    .doc_comment = try allocator.dupe(u8, prev_doc),
                    .category = category,
                    .build_flag = try allocator.dupe(u8, build_flag),
                });
            }
            // Pattern 2: pub const X = @import("services/...") — always-on
            else if (std.mem.startsWith(u8, rhs, "@import(\"")) {
                const path_start = "@import(\"".len;
                const path_end = std.mem.indexOfScalar(u8, rhs[path_start..], '"') orelse {
                    prev_doc = "";
                    continue;
                };
                const rel_path = rhs[path_start .. path_start + path_end];

                // Skip non-module imports (build_options, builtin, std)
                if (std.mem.eql(u8, rel_path, "build_options") or
                    std.mem.eql(u8, rel_path, "builtin") or
                    std.mem.eql(u8, rel_path, "std"))
                {
                    prev_doc = "";
                    continue;
                }

                const full_path = std.fmt.allocPrint(allocator, "src/{s}", .{rel_path}) catch {
                    prev_doc = "";
                    continue;
                };

                const category = categorizeByPath(rel_path, name);

                try modules.append(allocator, .{
                    .name = try allocator.dupe(u8, name),
                    .path = full_path,
                    .doc_comment = try allocator.dupe(u8, prev_doc),
                    .category = category,
                    .build_flag = try allocator.dupe(u8, "always-on"),
                });
            }

            prev_doc = "";
        } else if (line.len > 0 and !std.mem.startsWith(u8, line, "//")) {
            prev_doc = "";
        }
    }

    return try modules.toOwnedSlice(allocator);
}

fn categorizeByPath(path: []const u8, name: []const u8) Module.Category {
    // AI modules
    if (std.mem.startsWith(u8, path, "features/ai") or
        std.mem.eql(u8, name, "inference") or
        std.mem.eql(u8, name, "training") or
        std.mem.eql(u8, name, "reasoning"))
    {
        return .ai;
    }

    // Compute
    if (std.mem.startsWith(u8, path, "features/gpu") or
        std.mem.eql(u8, name, "runtime") or
        std.mem.eql(u8, name, "simd") or
        std.mem.eql(u8, name, "benchmarks"))
    {
        return .compute;
    }

    // Data
    if (std.mem.eql(u8, name, "database") or
        std.mem.eql(u8, name, "cache") or
        std.mem.eql(u8, name, "storage") or
        std.mem.eql(u8, name, "search"))
    {
        return .data;
    }

    // Core
    if (std.mem.startsWith(u8, path, "core/") or
        std.mem.eql(u8, name, "config") or
        std.mem.eql(u8, name, "framework") or
        std.mem.eql(u8, name, "errors") or
        std.mem.eql(u8, name, "registry"))
    {
        return .core;
    }

    // Infrastructure
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

    // Utilities
    if (std.mem.eql(u8, name, "shared") or
        std.mem.eql(u8, name, "platform") or
        std.mem.eql(u8, name, "tasks") or
        std.mem.eql(u8, name, "connectors") or
        std.mem.eql(u8, name, "auth") or
        std.mem.eql(u8, name, "analytics"))
    {
        return .utilities;
    }

    return .utilities;
}

// =============================================================================
// Main
// =============================================================================

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();
    const cwd = std.Io.Dir.cwd();

    // Ensure output directory exists
    cwd.createDirPath(io, "docs/api") catch |err| {
        if (err != error.PathAlreadyExists) {
            std.debug.print("Warning: Could not create docs/api directory: {}\n", .{err});
        }
    };

    std.debug.print("=== ABI Auto Documentation Generator ===\n\n", .{});

    // Auto-discover modules
    const modules = try discoverModules(allocator, io, cwd);
    defer {
        for (modules) |mod| {
            allocator.free(mod.name);
            allocator.free(mod.path);
            allocator.free(mod.doc_comment);
            allocator.free(mod.build_flag);
        }
        allocator.free(modules);
    }

    std.debug.print("  Discovered {d} modules from src/abi.zig\n\n", .{modules.len});

    var successful = std.ArrayListUnmanaged(Module).empty;
    defer successful.deinit(allocator);
    var failed_count: usize = 0;

    for (modules) |mod| {
        generateModuleDoc(allocator, io, cwd, mod) catch |err| {
            std.debug.print("  [SKIP] {s}: {}\n", .{ mod.name, err });
            failed_count += 1;
            continue;
        };
        successful.append(allocator, mod) catch {};
    }

    // Generate index
    generateIndex(allocator, io, cwd, successful.items) catch |err| {
        std.debug.print("Failed to generate index: {}\n", .{err});
    };

    std.debug.print("\n=== Documentation Generation Complete ===\n", .{});
    std.debug.print("  Generated: {d} modules\n", .{successful.items.len});
    std.debug.print("  Skipped:   {d} modules\n", .{failed_count});
    std.debug.print("  Output:    docs/api/\n", .{});
}

// =============================================================================
// Doc Generation
// =============================================================================

fn generateModuleDoc(allocator: std.mem.Allocator, io: std.Io, cwd: std.Io.Dir, mod: Module) !void {
    const source = cwd.readFileAlloc(io, mod.path, allocator, .limited(2 * 1024 * 1024)) catch |err| {
        return err;
    };
    defer allocator.free(source);

    const out_name = try std.fmt.allocPrint(allocator, "docs/api/{s}.md", .{mod.name});
    defer allocator.free(out_name);

    var file = cwd.createFile(io, out_name, .{}) catch |err| {
        return err;
    };
    defer file.close(io);

    // Header
    try print(allocator, io, file, "# {s}\n\n", .{mod.name});

    if (mod.doc_comment.len > 0) {
        try print(allocator, io, file, "> {s}\n\n", .{mod.doc_comment});
    }

    try print(allocator, io, file, "**Source:** [`{s}`](../../{s})\n\n", .{ mod.path, mod.path });

    if (!std.mem.eql(u8, mod.build_flag, "always-on")) {
        try print(allocator, io, file, "**Build flag:** `-D{s}=true`\n\n", .{mod.build_flag});
    } else {
        try file.writeStreamingAll(io, "**Availability:** Always enabled\n\n");
    }

    try file.writeStreamingAll(io, "---\n\n");

    // Parse source for doc items
    var lines = std.mem.splitScalar(u8, source, '\n');
    var doc_buffer = std.ArrayListUnmanaged(u8).empty;
    defer doc_buffer.deinit(allocator);

    var in_module_docs = true;
    var module_doc_written = false;

    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \r\t");

        // Module-level doc comments (//!)
        if (std.mem.startsWith(u8, line, "//!")) {
            if (in_module_docs) {
                const content = if (line.len > 3) line[3..] else "";
                const trimmed_content = std.mem.trimStart(u8, content, " ");
                try print(allocator, io, file, "{s}\n", .{trimmed_content});
                module_doc_written = true;
            }
            continue;
        }

        // End of module docs section
        if (in_module_docs and module_doc_written and line.len > 0 and !std.mem.startsWith(u8, line, "//")) {
            in_module_docs = false;
            try file.writeStreamingAll(io, "\n---\n\n## API\n\n");
        }

        // Item-level doc comments (///)
        if (std.mem.startsWith(u8, line, "///")) {
            const content = if (line.len > 3) line[3..] else "";
            const trimmed_content = std.mem.trimStart(u8, content, " ");
            try doc_buffer.appendSlice(allocator, trimmed_content);
            try doc_buffer.append(allocator, '\n');
            continue;
        }

        // Public declarations
        if (std.mem.startsWith(u8, line, "pub ")) {
            if (doc_buffer.items.len > 0) {
                const sig = extractDeclSignature(line);
                const item_type = detectItemType(line);

                const type_badge = switch (item_type) {
                    .function => "fn",
                    .constant => "const",
                    .type_def => "type",
                    .variable => "var",
                };

                try print(allocator, io, file, "### `{s}`\n\n", .{sig});
                try print(allocator, io, file, "<sup>**{s}**</sup>\n\n", .{type_badge});
                try print(allocator, io, file, "{s}\n", .{doc_buffer.items});

                doc_buffer.clearRetainingCapacity();
            }
        } else if (line.len > 0 and !std.mem.startsWith(u8, line, "//")) {
            doc_buffer.clearRetainingCapacity();
        }
    }

    // Footer
    try file.writeStreamingAll(io,
        \\---
        \\
        \\*Generated automatically by `zig build gendocs`*
        \\
    );

    std.debug.print("  [OK] {s} -> {s}\n", .{ mod.name, out_name });
}

fn generateIndex(allocator: std.mem.Allocator, io: std.Io, cwd: std.Io.Dir, generated_modules: []const Module) !void {
    var file = cwd.createFile(io, "docs/api/index.md", .{}) catch |err| {
        return err;
    };
    defer file.close(io);

    try file.writeStreamingAll(io,
        \\# ABI Framework API Reference
        \\
        \\> Comprehensive API documentation auto-generated from source code.
        \\
        \\---
        \\
        \\## Quick Links
        \\
        \\| Module | Category | Description |
        \\|--------|----------|-------------|
        \\
    );

    for (generated_modules) |mod| {
        try print(allocator, io, file, "| [{s}]({s}.md) | {s} | {s} |\n", .{
            mod.name,
            mod.name,
            mod.categoryName(),
            if (mod.doc_comment.len > 0) mod.doc_comment else "—",
        });
    }

    try file.writeStreamingAll(io, "\n---\n\n");

    // Categorized sections
    const categories = [_]Module.Category{ .core, .compute, .ai, .data, .infrastructure, .utilities };

    for (categories) |cat| {
        var has_items = false;
        for (generated_modules) |mod| {
            if (mod.category == cat) {
                if (!has_items) {
                    try print(allocator, io, file, "## {s}\n\n", .{mod.categoryName()});
                    has_items = true;
                }
                try print(allocator, io, file, "### [{s}]({s}.md)\n\n", .{ mod.name, mod.name });
                if (mod.doc_comment.len > 0) {
                    try print(allocator, io, file, "{s}\n\n", .{mod.doc_comment});
                }
                try print(allocator, io, file, "**Source:** [`{s}`](../../{s})", .{ mod.path, mod.path });
                if (!std.mem.eql(u8, mod.build_flag, "always-on")) {
                    try print(allocator, io, file, " | **Flag:** `-D{s}`", .{mod.build_flag});
                }
                try file.writeStreamingAll(io, "\n\n");
            }
        }
    }

    try file.writeStreamingAll(io,
        \\---
        \\
        \\*Generated automatically by `zig build gendocs`*
        \\
    );

    std.debug.print("  [OK] API index -> docs/api/index.md\n", .{});
}

// =============================================================================
// Helpers
// =============================================================================

fn print(allocator: std.mem.Allocator, io: std.Io, file: std.Io.File, comptime fmt: []const u8, args: anytype) !void {
    const s = try std.fmt.allocPrint(allocator, fmt, args);
    defer allocator.free(s);
    try file.writeStreamingAll(io, s);
}

fn extractDeclSignature(line: []const u8) []const u8 {
    var end = line.len;
    var depth: usize = 0;

    for (line, 0..) |c, i| {
        if (c == '(') depth += 1;
        if (c == ')') depth -|= 1;

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

    return std.mem.trim(u8, line[0..end], " \t");
}

fn detectItemType(line: []const u8) DocItem.ItemType {
    if (std.mem.indexOf(u8, line, "pub fn ")) |_| return .function;
    if (std.mem.indexOf(u8, line, "pub const ")) |_| {
        if (std.mem.indexOf(u8, line, "= struct") != null or
            std.mem.indexOf(u8, line, "= enum") != null or
            std.mem.indexOf(u8, line, "= union") != null or
            std.mem.indexOf(u8, line, "= @import") != null)
        {
            return .type_def;
        }
        return .constant;
    }
    if (std.mem.indexOf(u8, line, "pub var ")) |_| return .variable;
    return .constant;
}
