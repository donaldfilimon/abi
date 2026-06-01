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

    try expectContains(audit, "Zig `0.17.0-dev.329+21b7ceb5e`");
    try expectContains(audit, "in-process vector/key-value/block store");
    try expectContains(audit, "Not currently proven by repo source or tests");
    try expectContains(audit, "does not currently prove distributed sharding");
}
