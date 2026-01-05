const std = @import("std");
const explore = @import("mod.zig");

test "explore agent creation" {
    const allocator = std.testing.allocator;

    var agent = explore.createDefaultAgent(allocator);
    agent.deinit();

    try std.testing.expect(true);
}

test "explore config default" {
    const config = explore.ExploreConfig.defaultForLevel(.quick);
    try std.testing.expect(config.level == .quick);
    try std.testing.expect(config.max_files == 1000);
}

test "explore config thorough" {
    const config = explore.ExploreConfig.defaultForLevel(.thorough);
    try std.testing.expect(config.level == .thorough);
    try std.testing.expect(config.max_files == 10000);
}

test "explore result creation" {
    const allocator = std.testing.allocator;

    var result = explore.ExploreResult.init(allocator, "test query", .medium);
    defer result.deinit();

    try std.testing.expectEqualStrings("test query", result.query);
    try std.testing.expect(result.level == .medium);
    try std.testing.expect(result.matches.items.len == 0);
}

test "search pattern literal" {
    const allocator = std.testing.allocator;

    var compiler = explore.search.PatternCompiler.init(allocator);

    var pattern = try compiler.compile("test", explore.search.PatternType.literal, false);
    defer pattern.deinit(allocator);

    try std.testing.expect(pattern.pattern_type == .literal);
}

test "glob matching" {
    try std.testing.expect(explore.search.matchesGlob("*.zig", "main.zig"));
    try std.testing.expect(!explore.search.matchesGlob("*.zig", "main.c"));
    try std.testing.expect(explore.search.matchesGlob("main*", "main.zig"));
}

test "file stats creation" {
    const allocator = std.testing.allocator;
    _ = allocator;

    const test_path = ".";
    const file_stats = explore.fs.FileStats{
        .path = test_path,
        .size_bytes = 0,
        .mtime = 0,
        .ctime = 0,
        .is_directory = true,
        .is_symlink = false,
        .mode = 0,
    };

    try std.testing.expect(file_stats.is_directory);
}

test "file type detection" {
    try std.testing.expectEqualStrings("source", explore.fs.determineFileType("main.zig"));
    try std.testing.expectEqualStrings("source", explore.fs.determineFileType("main.c"));
    try std.testing.expectEqualStrings("test", explore.fs.determineFileType("main.test.zig"));
    try std.testing.expectEqualStrings("documentation", explore.fs.determineFileType("README.md"));
    try std.testing.expectEqualStrings("config", explore.fs.determineFileType("config.json"));
}
