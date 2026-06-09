const std = @import("std");

fn expectContains(haystack: []const u8, needle: []const u8) !void {
    try std.testing.expect(std.mem.indexOf(u8, haystack, needle) != null);
}

fn expectNotContains(haystack: []const u8, needle: []const u8) !void {
    try std.testing.expect(std.mem.indexOf(u8, haystack, needle) == null);
}

test "public overview docs avoid unsupported external benchmark claims" {
    const allocator = std.testing.allocator;
    const public_paths = [_][]const u8{
        "README.md",
        "CHANGELOG.md",
        "docs/index.md",
        "docs/contracts/public-api.md",
        "docs/spec/multi-persona-technical.md",
    };

    for (public_paths) |path| {
        const content = try std.Io.Dir.cwd().readFileAlloc(std.Options.debug_io, path, allocator, .limited(1024 * 1024));
        defer allocator.free(content);

        inline for (.{
            "12,000 QPS",
            "8.2 ms",
            "295x",
            "13x",
            "SQuAD",
            "CodeSearchNet",
            "15 kWh",
            "10,000 req/s",
            "50 ms",
            "GPT-4o",
            "OpenAI o1",
            "Swift 6",
        }) |claim| {
            try expectNotContains(content, claim);
        }
    }
}

test "external claims audit records repo-backed replacement language" {
    const audit = try std.Io.Dir.cwd().readFileAlloc(
        std.Options.debug_io,
        "docs/contracts/external-claims-audit.md",
        std.testing.allocator,
        .limited(1024 * 1024),
    );
    defer std.testing.allocator.free(audit);

    try expectContains(audit, "Zig `0.17.0-dev.813+2153f8143`");
    try expectContains(audit, "in-process vector/key-value/block store");
    try expectContains(audit, "Not currently proven by repo source or tests");
    try expectContains(audit, "does not currently prove distributed sharding");
}

test "README walkthrough documents current CLI and MCP surfaces" {
    const readme = try std.Io.Dir.cwd().readFileAlloc(
        std.Options.debug_io,
        "README.md",
        std.testing.allocator,
        .limited(1024 * 1024),
    );
    defer std.testing.allocator.free(readme);

    try expectContains(readme, "0.17.0-dev.813+2153f8143");
    try expectContains(readme, "./zig-out/bin/abi scheduler status");
    try expectContains(readme, "./zig-out/bin/abi agent plan");
    try expectContains(readme, "./zig-out/bin/abi agent train all");
    try expectContains(readme, "./zig-out/bin/abi wdbx query zig-out/local-memory.jsonl");
    try expectContains(readme, "one-shot scheduler probe");
    try expectContains(readme, "scheduler-backed AI helper surface");
    try expectContains(readme, "ai_train");
    try expectContains(readme, "wdbx_query");
    try expectContains(readme, "scheduler_info");
    try expectContains(readme, "plugin_run");
    try expectContains(readme, "does not perform live network dispatch");
}

test "master spec keeps GPU acceleration claim boundary explicit" {
    const spec = try std.Io.Dir.cwd().readFileAlloc(
        std.Options.debug_io,
        "docs/superpowers/specs/ABI-MASTER-SPEC.md",
        std.testing.allocator,
        .limited(1024 * 1024),
    );
    defer std.testing.allocator.free(spec);

    try expectContains(spec, "batchCosineSimilarity");
    try expectContains(spec, "deterministically fall back to vectorized CPU");
    try expectContains(spec, "does not prove distributed sharding");
    try expectContains(spec, "does not prove distributed sharding, AES/RBAC");
    try expectContains(spec, "general native GPU acceleration");
    try expectContains(spec, "native Metal execution is claimed only when the runtime backend reports initialized native kernels");
    try expectNotContains(spec, "real GPU acceleration for HNSW cosine *has* been implemented");
}
