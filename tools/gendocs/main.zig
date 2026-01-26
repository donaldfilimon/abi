//! Automatic API Documentation Generator
//!
//! Generates comprehensive markdown documentation from Zig doc comments.
//! Supports module-level (`//!`) and item-level (`///`) documentation.
//!
//! ## Usage
//! ```bash
//! zig build gendocs
//! ```
//!
//! ## Output
//! - `docs/api/index.md` - API index with all modules
//! - `docs/api/<module>.md` - Individual module documentation

const std = @import("std");

/// Module definition for documentation generation
const Module = struct {
    path: []const u8,
    name: []const u8,
    category: Category,
    description: []const u8,

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
};

/// All modules to document
const modules = [_]Module{
    // Core Framework
    .{ .path = "src/abi.zig", .name = "abi", .category = .core, .description = "Main framework entry point and public API" },
    .{ .path = "src/config/mod.zig", .name = "config", .category = .core, .description = "Unified configuration system with builder pattern" },
    .{ .path = "src/framework.zig", .name = "framework", .category = .core, .description = "Framework orchestration and lifecycle management" },

    // Compute & Runtime
    .{ .path = "src/runtime/mod.zig", .name = "runtime", .category = .compute, .description = "Runtime infrastructure (engine, scheduling, memory)" },
    .{ .path = "src/runtime/engine/mod.zig", .name = "runtime-engine", .category = .compute, .description = "Work-stealing task execution engine" },
    .{ .path = "src/runtime/scheduling/mod.zig", .name = "runtime-scheduling", .category = .compute, .description = "Futures, cancellation, and task groups" },
    .{ .path = "src/runtime/memory/mod.zig", .name = "runtime-memory", .category = .compute, .description = "Memory pools and custom allocators" },
    .{ .path = "src/runtime/concurrency/mod.zig", .name = "runtime-concurrency", .category = .compute, .description = "Lock-free concurrent primitives" },

    // GPU
    .{ .path = "src/gpu/mod.zig", .name = "gpu", .category = .compute, .description = "GPU acceleration framework (Vulkan, CUDA, Metal, WebGPU)" },

    // AI & Machine Learning
    .{ .path = "src/ai/mod.zig", .name = "ai", .category = .ai, .description = "AI module with agents, LLM, embeddings, and training" },
    .{ .path = "src/ai/agents/mod.zig", .name = "ai-agents", .category = .ai, .description = "Agent runtime and orchestration" },
    .{ .path = "src/ai/embeddings/mod.zig", .name = "ai-embeddings", .category = .ai, .description = "Vector embeddings generation" },
    .{ .path = "src/ai/llm/mod.zig", .name = "ai-llm", .category = .ai, .description = "Local LLM inference" },
    .{ .path = "src/ai/training/mod.zig", .name = "ai-training", .category = .ai, .description = "Training pipelines and fine-tuning" },
    .{ .path = "src/connectors/mod.zig", .name = "connectors", .category = .ai, .description = "API connectors (OpenAI, Ollama, Anthropic, HuggingFace)" },

    // Data & Storage
    .{ .path = "src/database/mod.zig", .name = "database", .category = .data, .description = "Vector database (WDBX with HNSW/IVF-PQ)" },

    // Infrastructure
    .{ .path = "src/network/mod.zig", .name = "network", .category = .infrastructure, .description = "Distributed compute and Raft consensus" },
    .{ .path = "src/ha/mod.zig", .name = "ha", .category = .infrastructure, .description = "High availability (backup, PITR, replication)" },
    .{ .path = "src/observability/mod.zig", .name = "observability", .category = .infrastructure, .description = "Metrics, tracing, and monitoring" },
    .{ .path = "src/registry/mod.zig", .name = "registry", .category = .infrastructure, .description = "Plugin registry (comptime, runtime, dynamic)" },
    .{ .path = "src/web/mod.zig", .name = "web", .category = .infrastructure, .description = "Web utilities and HTTP support" },

    // Utilities
    .{ .path = "src/shared/security/mod.zig", .name = "security", .category = .utilities, .description = "TLS, mTLS, API keys, and RBAC" },
};

/// Documentation item (function, type, constant)
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

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();
    const cwd = std.Io.Dir.cwd();

    // Ensure docs/api directory exists for index
    cwd.createDirPath(io, "docs/api") catch |err| {
        if (err != error.PathAlreadyExists) {
            std.debug.print("Warning: Could not create docs/api directory: {}\n", .{err});
        }
    };

    std.debug.print("=== ABI Auto Documentation Generator ===\n\n", .{});

    var successful_modules = std.ArrayListUnmanaged(Module){};
    defer successful_modules.deinit(allocator);

    var failed_count: usize = 0;

    // Generate documentation for each module
    for (modules) |mod| {
        generateModuleDoc(allocator, io, cwd, mod) catch |err| {
            std.debug.print("  [SKIP] {s}: {}\n", .{ mod.name, err });
            failed_count += 1;
            continue;
        };
        successful_modules.append(allocator, mod) catch {};
    }

    // Generate index
    generateIndex(allocator, io, cwd, successful_modules.items) catch |err| {
        std.debug.print("Failed to generate index: {}\n", .{err});
    };

    std.debug.print("\n=== Documentation Generation Complete ===\n", .{});
    std.debug.print("  Generated: {d} modules\n", .{successful_modules.items.len});
    std.debug.print("  Skipped:   {d} modules\n", .{failed_count});
    std.debug.print("  Output:    docs/api/\n", .{});
}

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

    // Write header
    try print(allocator, io, file,
        \\# {s} API Reference
        \\
        \\> {s}
        \\
        \\**Source:** [`{s}`](../../{s})
        \\
        \\---
        \\
        \\
    , .{ mod.name, mod.description, mod.path, mod.path });

    // Parse and write module-level docs
    var lines = std.mem.splitScalar(u8, source, '\n');
    var doc_buffer = std.ArrayListUnmanaged(u8){};
    defer doc_buffer.deinit(allocator);

    var items = std.ArrayListUnmanaged(DocItem){};
    defer {
        for (items.items) |item| {
            allocator.free(item.signature);
            allocator.free(item.doc);
        }
        items.deinit(allocator);
    }

    var in_module_docs = true;
    var module_doc_written = false;

    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \r\t");

        // Module-level doc comments (//!)
        if (std.mem.startsWith(u8, trimmed, "//!")) {
            if (in_module_docs) {
                const content = if (trimmed.len > 3) trimmed[3..] else "";
                const trimmed_content = std.mem.trimStart(u8, content, " ");
                try print(allocator, io, file, "{s}\n", .{trimmed_content});
                module_doc_written = true;
            }
            continue;
        }

        // End of module docs section
        if (in_module_docs and module_doc_written and trimmed.len > 0 and !std.mem.startsWith(u8, trimmed, "//")) {
            in_module_docs = false;
            try file.writeStreamingAll(io, "\n---\n\n## API\n\n");
        }

        // Item-level doc comments (///)
        if (std.mem.startsWith(u8, trimmed, "///")) {
            const content = if (trimmed.len > 3) trimmed[3..] else "";
            const trimmed_content = std.mem.trimStart(u8, content, " ");
            try doc_buffer.appendSlice(allocator, trimmed_content);
            try doc_buffer.append(allocator, '\n');
            continue;
        }

        // Public declarations
        if (std.mem.startsWith(u8, trimmed, "pub ")) {
            if (doc_buffer.items.len > 0) {
                const sig = extractDeclSignature(trimmed);
                const item_type = detectItemType(trimmed);

                // Write the item directly
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
        } else if (trimmed.len > 0 and !std.mem.startsWith(u8, trimmed, "//")) {
            // Non-doc, non-pub line - clear buffer
            doc_buffer.clearRetainingCapacity();
        }
    }

    // Write footer
    try print(allocator, io, file,
        \\---
        \\
        \\*Generated automatically by `zig build gendocs`*
        \\
    , .{});

    std.debug.print("  [OK] {s} -> {s}\n", .{ mod.name, out_name });
}

fn generateIndex(allocator: std.mem.Allocator, io: std.Io, cwd: std.Io.Dir, generated_modules: []const Module) !void {
    var file = cwd.createFile(io, "docs/api/index.md", .{}) catch |err| {
        return err;
    };
    defer file.close(io);

    try print(allocator, io, file,
        \\# ABI Framework API Reference
        \\
        \\> Comprehensive API documentation auto-generated from source code
        \\
        \\---
        \\
        \\## Quick Links
        \\
        \\| Module | Description |
        \\|--------|-------------|
        \\
    , .{});

    // Quick links table
    for (generated_modules) |mod| {
        try print(allocator, io, file, "| [{s}](../api/{s}.md) | {s} |\n", .{ mod.name, mod.name, mod.description });
    }

    try file.writeStreamingAll(io, "\n---\n\n");

    // Categorized sections
    const categories = [_]Module.Category{ .core, .compute, .ai, .data, .infrastructure, .utilities };

    for (categories) |cat| {
        var has_items = false;
        for (generated_modules) |mod| {
            if (mod.category == cat) {
                if (!has_items) {
                    const cat_name = mod.categoryName();
                    try print(allocator, io, file, "## {s}\n\n", .{cat_name});
                    has_items = true;
                }
                try print(allocator, io, file, "### [{s}](../api/{s}.md)\n\n", .{ mod.name, mod.name });
                try print(allocator, io, file, "{s}\n\n", .{mod.description});
                try print(allocator, io, file, "**Source:** [`{s}`](../../{s})\n\n", .{ mod.path, mod.path });
            }
        }
    }

    // Footer
    try print(allocator, io, file,
        \\---
        \\
        \\## Additional Resources
        \\
        \\- [Getting Started Guide](../tutorials/getting-started.md)
        \\- [Architecture Overview](../architecture/overview.md)
        \\- [Feature Flags](../feature-flags.md)
        \\- [Troubleshooting](../troubleshooting.md)
        \\
        \\---
        \\
        \\*Generated automatically by `zig build gendocs`*
        \\
    , .{});

    std.debug.print("  [OK] API index -> docs/api/index.md\n", .{});
}

fn print(allocator: std.mem.Allocator, io: std.Io, file: std.Io.File, comptime fmt: []const u8, args: anytype) !void {
    const s = try std.fmt.allocPrint(allocator, fmt, args);
    defer allocator.free(s);
    try file.writeStreamingAll(io, s);
}

fn extractDeclSignature(line: []const u8) []const u8 {
    // Extract signature up to opening brace, semicolon, or equals
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
        // Check if it's a type definition
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
