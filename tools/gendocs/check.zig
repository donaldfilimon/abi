const std = @import("std");
const model = @import("model.zig");

pub const CheckError = error{DriftDetected};
pub const VerifyOptions = struct {
    untracked_markdown: bool = false,
};

pub fn verifyOutputs(
    allocator: std.mem.Allocator,
    io: std.Io,
    cwd: std.Io.Dir,
    expected: []const model.OutputFile,
    options: VerifyOptions,
) !void {
    var expected_map = std.StringHashMapUnmanaged(void).empty;
    defer expected_map.deinit(allocator);

    var missing = std.ArrayListUnmanaged([]const u8).empty;
    defer {
        for (missing.items) |path| allocator.free(path);
        missing.deinit(allocator);
    }
    var changed = std.ArrayListUnmanaged([]const u8).empty;
    defer {
        for (changed.items) |path| allocator.free(path);
        changed.deinit(allocator);
    }
    var extra = std.ArrayListUnmanaged([]const u8).empty;
    defer {
        for (extra.items) |path| allocator.free(path);
        extra.deinit(allocator);
    }

    for (expected) |file| {
        try validateOutputPolicy(file.path, file.content);
        try expected_map.put(allocator, file.path, {});

        if (!shouldEnforceDrift(file.path, options)) {
            continue;
        }

        const current = cwd.readFileAlloc(io, file.path, allocator, .limited(16 * 1024 * 1024)) catch |err| {
            if (err == error.FileNotFound) {
                try missing.append(allocator, try allocator.dupe(u8, file.path));
                continue;
            }
            return err;
        };
        defer allocator.free(current);

        if (!std.mem.eql(u8, current, file.content)) {
            try changed.append(allocator, try allocator.dupe(u8, file.path));
        }
    }

    try collectExtraFiles(allocator, io, cwd, "docs/api", &expected_map, true, &extra);
    try collectExtraFiles(allocator, io, cwd, "docs/_docs", &expected_map, true, &extra);
    try collectExtraFiles(allocator, io, cwd, "docs/plans", &expected_map, true, &extra);
    try collectExtraFiles(allocator, io, cwd, "docs", &expected_map, false, &extra);
    if (options.untracked_markdown) {
        filterGeneratedMarkdownExtras(allocator, &extra);
    }

    // Cross-file link validation (warnings only, not blocking)
    try validateInternalLinks(allocator, expected);

    if (missing.items.len == 0 and changed.items.len == 0 and extra.items.len == 0) {
        std.debug.print("OK: docs are up to date\n", .{});
        return;
    }

    std.debug.print("ERROR: docs drift detected\n", .{});
    if (missing.items.len > 0) {
        std.debug.print("  Missing files:\n", .{});
        for (missing.items) |path| std.debug.print("    - {s}\n", .{path});
    }
    if (changed.items.len > 0) {
        std.debug.print("  Changed files:\n", .{});
        for (changed.items) |path| std.debug.print("    - {s}\n", .{path});
    }
    if (extra.items.len > 0) {
        std.debug.print("  Extra files:\n", .{});
        for (extra.items) |path| std.debug.print("    - {s}\n", .{path});
    }

    return CheckError.DriftDetected;
}

fn shouldEnforceDrift(path: []const u8, options: VerifyOptions) bool {
    if (!options.untracked_markdown) return true;
    return !isGeneratedMarkdownPath(path);
}

fn isGeneratedMarkdownPath(path: []const u8) bool {
    if (!std.mem.endsWith(u8, path, ".md")) return false;
    return std.mem.startsWith(u8, path, "docs/api/") or
        std.mem.startsWith(u8, path, "docs/_docs/") or
        std.mem.startsWith(u8, path, "docs/plans/");
}

fn filterGeneratedMarkdownExtras(
    allocator: std.mem.Allocator,
    extras: *std.ArrayListUnmanaged([]const u8),
) void {
    var i: usize = 0;
    while (i < extras.items.len) {
        if (isGeneratedMarkdownPath(extras.items[i])) {
            allocator.free(extras.orderedRemove(i));
        } else {
            i += 1;
        }
    }
}

fn validateOutputPolicy(path: []const u8, content: []const u8) !void {
    if (!std.mem.endsWith(u8, path, ".md")) return;

    if (std.mem.indexOf(u8, content, "](/") != null) {
        std.debug.print("ERROR: forbidden root-absolute markdown link in {s}\n", .{path});
        return CheckError.DriftDetected;
    }
    if (std.mem.indexOf(u8, content, "/Users/") != null) {
        std.debug.print("ERROR: forbidden local filesystem path in {s}\n", .{path});
        return CheckError.DriftDetected;
    }
    if (hasLeftoverPlaceholder(content)) {
        std.debug.print("ERROR: unreplaced template placeholder in {s}\n", .{path});
        return CheckError.DriftDetected;
    }
}

/// Detect `{{...}}` placeholders that survived template expansion.
fn hasLeftoverPlaceholder(content: []const u8) bool {
    var cursor: usize = 0;
    while (cursor + 4 < content.len) {
        const open = std.mem.indexOfPos(u8, content, cursor, "{{") orelse return false;
        // Skip code fences (``` blocks often contain template examples)
        if (isInsideCodeFence(content, open)) {
            cursor = open + 2;
            continue;
        }
        const close = std.mem.indexOfPos(u8, content, open + 2, "}}") orelse return false;
        // Require at least one uppercase letter (template vars are UPPER_SNAKE)
        const inner = content[open + 2 .. close];
        for (inner) |c| {
            if (std.ascii.isUpper(c)) return true;
        }
        cursor = close + 2;
    }
    return false;
}

/// Simple heuristic: check if position is inside a fenced code block.
fn isInsideCodeFence(content: []const u8, pos: usize) bool {
    var fence_count: usize = 0;
    var cursor: usize = 0;
    while (cursor < pos) {
        if (cursor + 3 <= content.len and std.mem.eql(u8, content[cursor .. cursor + 3], "```")) {
            fence_count += 1;
            cursor += 3;
        } else {
            cursor += 1;
        }
    }
    // Odd fence count means we're inside a code block
    return fence_count % 2 == 1;
}

/// Validate that relative markdown links point to files in the expected output set.
pub fn validateInternalLinks(
    allocator: std.mem.Allocator,
    expected: []const model.OutputFile,
) !void {
    var expected_set = std.StringHashMapUnmanaged(void).empty;
    defer expected_set.deinit(allocator);

    for (expected) |file| {
        try expected_set.put(allocator, file.path, {});
    }
    // Root files are valid link targets
    try expected_set.put(allocator, "README.md", {});
    try expected_set.put(allocator, "AGENTS.md", {});
    try expected_set.put(allocator, "LICENSE", {});

    var broken_count: usize = 0;
    for (expected) |file| {
        if (!std.mem.endsWith(u8, file.path, ".md")) continue;
        const dir = std.fs.path.dirname(file.path) orelse "";

        var cursor: usize = 0;
        while (cursor < file.content.len) {
            const link_target = extractNextLink(file.content, &cursor) orelse break;
            // Skip external URLs, anchors, and non-md links
            if (link_target.len == 0) continue;
            if (link_target[0] == '#') continue;
            if (std.mem.startsWith(u8, link_target, "http://")) continue;
            if (std.mem.startsWith(u8, link_target, "https://")) continue;
            if (!std.mem.endsWith(u8, link_target, ".md") and
                std.mem.indexOf(u8, link_target, ".md#") == null) continue;

            // Strip fragment
            const path_only = if (std.mem.indexOfScalar(u8, link_target, '#')) |hash|
                link_target[0..hash]
            else
                link_target;

            // Resolve relative to file's directory
            const resolved = resolvePath(allocator, dir, path_only) catch continue;
            defer allocator.free(resolved);

            if (!expected_set.contains(resolved)) {
                std.debug.print("WARN: broken link in {s}: {s} (resolved: {s})\n", .{ file.path, link_target, resolved });
                broken_count += 1;
            }
        }
    }

    if (broken_count > 0) {
        std.debug.print("WARN: {d} broken internal link(s) found\n", .{broken_count});
    }
}

/// Extract the next markdown link target `[...](TARGET)` from content.
fn extractNextLink(content: []const u8, cursor: *usize) ?[]const u8 {
    while (cursor.* < content.len) {
        const open = std.mem.indexOfPos(u8, content, cursor.*, "](") orelse return null;
        const target_start = open + 2;
        const close = std.mem.indexOfScalarPos(u8, content, target_start, ')') orelse {
            cursor.* = target_start;
            continue;
        };
        cursor.* = close + 1;
        return content[target_start..close];
    }
    return null;
}

fn collectExtraFiles(
    allocator: std.mem.Allocator,
    io: std.Io,
    cwd: std.Io.Dir,
    root_path: []const u8,
    expected_map: *const std.StringHashMapUnmanaged(void),
    markdown_only: bool,
    extras: *std.ArrayListUnmanaged([]const u8),
) !void {
    var root = cwd.openDir(io, root_path, .{ .iterate = true }) catch |err| {
        if (err == error.FileNotFound) return;
        return err;
    };
    defer root.close(io);

    try walkExtra(allocator, io, cwd, root, root_path, expected_map, markdown_only, extras);
}

fn walkExtra(
    allocator: std.mem.Allocator,
    io: std.Io,
    cwd: std.Io.Dir,
    dir: std.Io.Dir,
    prefix: []const u8,
    expected_map: *const std.StringHashMapUnmanaged(void),
    markdown_only: bool,
    extras: *std.ArrayListUnmanaged([]const u8),
) !void {
    var it = dir.iterate();
    while (true) {
        const maybe_entry = try it.next(io);
        const entry = maybe_entry orelse break;

        const full_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ prefix, entry.name });
        defer allocator.free(full_path);

        switch (entry.kind) {
            .file => {
                if (markdown_only and !std.mem.endsWith(u8, entry.name, ".md")) continue;
                if (!shouldTrackExtraFile(full_path)) continue;
                if (!expected_map.contains(full_path)) {
                    try extras.append(allocator, try allocator.dupe(u8, full_path));
                }
            },
            .directory => {
                if (shouldSkipDir(full_path)) continue;
                var child = try cwd.openDir(io, full_path, .{ .iterate = true });
                defer child.close(io);
                try walkExtra(allocator, io, cwd, child, full_path, expected_map, markdown_only, extras);
            },
            else => {},
        }
    }
}

/// Resolve a relative path against a base directory, collapsing `..` components.
fn resolvePath(allocator: std.mem.Allocator, base: []const u8, rel: []const u8) ![]u8 {
    var parts = std.ArrayListUnmanaged([]const u8).empty;
    defer parts.deinit(allocator);

    // Start with base directory components
    var base_it = std.mem.splitScalar(u8, base, '/');
    while (base_it.next()) |part| {
        if (part.len == 0 or std.mem.eql(u8, part, ".")) continue;
        try parts.append(allocator, part);
    }

    // Apply relative path components
    var rel_it = std.mem.splitScalar(u8, rel, '/');
    while (rel_it.next()) |part| {
        if (part.len == 0 or std.mem.eql(u8, part, ".")) continue;
        if (std.mem.eql(u8, part, "..")) {
            if (parts.items.len > 0) _ = parts.pop();
        } else {
            try parts.append(allocator, part);
        }
    }

    // Join back
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    for (parts.items, 0..) |part, idx| {
        if (idx > 0) try out.append(allocator, '/');
        try out.appendSlice(allocator, part);
    }
    return out.toOwnedSlice(allocator);
}

fn shouldTrackExtraFile(path: []const u8) bool {
    if (std.mem.endsWith(u8, path, "/docs_engine.wasm")) return false;
    if (std.mem.endsWith(u8, path, "/docs_engine.component.wasm")) return false;
    if (std.mem.endsWith(u8, path, "/docs_engine.wit")) return false;
    return true;
}

fn shouldSkipDir(path: []const u8) bool {
    if (std.mem.endsWith(u8, path, "/data") and std.mem.indexOf(u8, path, "api-app") != null) return true;
    return false;
}

pub fn writeOutputs(
    allocator: std.mem.Allocator,
    io: std.Io,
    cwd: std.Io.Dir,
    expected: []const model.OutputFile,
) !void {
    _ = allocator;
    for (expected) |file| {
        if (std.fs.path.dirname(file.path)) |dir_path| {
            cwd.createDirPath(io, dir_path) catch |err| {
                if (err != error.PathAlreadyExists) return err;
            };
        }
        var out_file = try cwd.createFile(io, file.path, .{ .truncate = true });
        defer out_file.close(io);
        try out_file.writeStreamingAll(io, file.content);
    }
}

test "shouldTrackExtraFile ignores wasm artifacts" {
    try std.testing.expect(!shouldTrackExtraFile("docs/data/docs_engine.wasm"));
    try std.testing.expect(shouldTrackExtraFile("docs/plans/index.md"));
}

test "generated markdown classifier and drift mode" {
    try std.testing.expect(isGeneratedMarkdownPath("docs/api/index.md"));
    try std.testing.expect(isGeneratedMarkdownPath("docs/_docs/architecture.md"));
    try std.testing.expect(isGeneratedMarkdownPath("docs/plans/index.md"));
    try std.testing.expect(!isGeneratedMarkdownPath("docs/data/features.zon"));
    try std.testing.expect(!isGeneratedMarkdownPath("README.md"));

    try std.testing.expect(!shouldEnforceDrift("docs/api/index.md", .{ .untracked_markdown = true }));
    try std.testing.expect(shouldEnforceDrift("docs/data/features.zon", .{ .untracked_markdown = true }));
    try std.testing.expect(shouldEnforceDrift("docs/api/index.md", .{}));
}

test "filterGeneratedMarkdownExtras removes only generated markdown paths" {
    const allocator = std.testing.allocator;
    var extras = std.ArrayListUnmanaged([]const u8).empty;
    defer {
        for (extras.items) |item| allocator.free(item);
        extras.deinit(allocator);
    }

    try extras.append(allocator, try allocator.dupe(u8, "docs/api/index.md"));
    try extras.append(allocator, try allocator.dupe(u8, "docs/_docs/architecture.md"));
    try extras.append(allocator, try allocator.dupe(u8, "docs/data/features.zon"));
    try extras.append(allocator, try allocator.dupe(u8, "docs/index.html"));

    filterGeneratedMarkdownExtras(allocator, &extras);

    try std.testing.expectEqual(@as(usize, 2), extras.items.len);
    try std.testing.expectEqualStrings("docs/data/features.zon", extras.items[0]);
    try std.testing.expectEqualStrings("docs/index.html", extras.items[1]);
}

test "hasLeftoverPlaceholder detects unreplaced template vars" {
    try std.testing.expect(hasLeftoverPlaceholder("Hello {{TITLE}} world"));
    try std.testing.expect(hasLeftoverPlaceholder("{{AUTO_CONTENT}}"));
    // Lowercase-only placeholders are not template vars
    try std.testing.expect(!hasLeftoverPlaceholder("{{lowercase}}"));
    // No placeholders
    try std.testing.expect(!hasLeftoverPlaceholder("Hello world"));
    // Inside code fence should be ignored
    try std.testing.expect(!hasLeftoverPlaceholder("```\n{{TITLE}}\n```"));
}

test "resolvePath collapses parent references" {
    {
        const resolved = try resolvePath(std.testing.allocator, "docs/_docs", "../api/index.md");
        defer std.testing.allocator.free(resolved);
        try std.testing.expectEqualStrings("docs/api/index.md", resolved);
    }
    {
        const resolved = try resolvePath(std.testing.allocator, "docs/api", "coverage.md");
        defer std.testing.allocator.free(resolved);
        try std.testing.expectEqualStrings("docs/api/coverage.md", resolved);
    }
    {
        const resolved = try resolvePath(std.testing.allocator, "docs/_docs", "../plans/index.md");
        defer std.testing.allocator.free(resolved);
        try std.testing.expectEqualStrings("docs/plans/index.md", resolved);
    }
}

test "extractNextLink parses markdown links" {
    const content = "See [API](../api/index.md) and [home](https://example.com)";
    var cursor: usize = 0;

    const first = extractNextLink(content, &cursor);
    try std.testing.expect(first != null);
    try std.testing.expectEqualStrings("../api/index.md", first.?);

    const second = extractNextLink(content, &cursor);
    try std.testing.expect(second != null);
    try std.testing.expectEqualStrings("https://example.com", second.?);

    const third = extractNextLink(content, &cursor);
    try std.testing.expect(third == null);
}

test "validateOutputPolicy rejects root-absolute and local links" {
    try std.testing.expectError(
        CheckError.DriftDetected,
        validateOutputPolicy("docs/_docs/roadmap.md", "](/api/)"),
    );
    try std.testing.expectError(
        CheckError.DriftDetected,
        validateOutputPolicy("docs/_docs/roadmap.md", "/Users/donaldfilimon/abi"),
    );
    try validateOutputPolicy("docs/_docs/roadmap.md", "[OK](../plans/index.md)");
}
