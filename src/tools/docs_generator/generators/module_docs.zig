const std = @import("std");

const Module = struct {
    title: []const u8,
    namespace: []const u8,
    slug: []const u8,
    summary: []const u8,
};

const modules = [_]Module{
    .{ .title = "Database", .namespace = "abi.database", .slug = "database", .summary = "Storage primitives and vector search orchestration." },
    .{ .title = "AI", .namespace = "abi.ai", .slug = "ai", .summary = "Inference helpers for the multi-agent runtime." },
    .{ .title = "SIMD", .namespace = "abi.simd", .slug = "simd", .summary = "Portable SIMD accelerated math utilities." },
    .{ .title = "HTTP Client", .namespace = "abi.http_client", .slug = "http_client", .summary = "Minimal HTTP client used by integrations and tests." },
    .{ .title = "Plugins", .namespace = "abi.plugins", .slug = "plugins", .summary = "Extension points for custom storage and scoring logic." },
};

pub fn generateModuleDocs(allocator: std.mem.Allocator) !void {
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();
    var writer = buffer.writer();

    try writer.writeAll("# ABI Module Reference\n\n");
    try writer.writeAll("This document lists the primary Zig namespaces that make up ABI. Each entry links back to the public API directory so the markdown can be browsed locally or on GitHub Pages.\n\n");

    for (modules) |module| {
        try writer.print("## {s}\n\n", .{module.title});
        try writer.print("- Namespace: `{s}`\n", .{module.namespace});
        try writer.print("- Summary: {s}\n", .{module.summary});
        try writer.print("- Reference: [docs/api/{s}.md](../api/{s}.md)\n\n", .{ module.slug, module.slug });
    }

    var file = try std.fs.cwd().createFile("docs/generated/MODULE_REFERENCE.md", .{ .truncate = true });
    defer file.close();
    try file.writeAll(buffer.items);
}
