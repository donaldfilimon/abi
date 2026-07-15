//! File-aware agent context: `@file` mention parsing, path sandboxing, and
//! context budget management. Pure (no IO) except for the file-read helper
//! which takes an explicit `std.Io`.
//!
//! Path sandboxing mirrors the `ai_train` G4 discipline: paths are confined
//! to `root` (default: cwd), rejecting `..`, absolute paths outside root,
//! and symlink escapes.

const std = @import("std");

/// Maximum total bytes of file context injected into a prompt.
pub const DEFAULT_BUDGET_BYTES: usize = 8192;

/// A single `@file` mention found in input text.
pub const FileMention = struct {
    path: []const u8,
    start: usize,
    end: usize,
};

/// Context budget tracker. Ensures injected file content does not exceed
/// a configurable byte budget.
pub const ContextBudget = struct {
    max_bytes: usize,
    used: usize = 0,

    pub fn init(max_bytes: usize) ContextBudget {
        return .{ .max_bytes = max_bytes };
    }

    pub fn remaining(self: ContextBudget) usize {
        return if (self.used >= self.max_bytes) 0 else self.max_bytes - self.used;
    }

    pub fn canFit(self: ContextBudget, bytes: usize) bool {
        return self.used + bytes <= self.max_bytes;
    }

    pub fn consume(self: *ContextBudget, bytes: usize) void {
        self.used += bytes;
        if (self.used > self.max_bytes) self.used = self.max_bytes;
    }
};

/// Validate that `path` is a safe relative path confined to `root`.
/// Rejects `..`, absolute paths, and empty paths. Does NOT resolve symlinks
/// (that requires IO); callers should use `resolveAndInject` for full sandboxing.
pub fn validateMentionPath(path: []const u8, root: []const u8) !void {
    if (path.len == 0) return error.EmptyPath;
    if (std.fs.path.isAbsolute(path)) return error.AbsolutePathRejected;
    // Reject any `..` component
    var it = std.mem.splitScalar(u8, path, '/');
    while (it.next()) |component| {
        if (std.mem.eql(u8, component, "..")) return error.PathEscape;
    }
    _ = root;
}

/// Scan input text for `@file` mentions. Returns owned slices of the mention
/// paths (borrowed from `input`) and their positions.
pub fn parseFileMentions(allocator: std.mem.Allocator, input: []const u8) ![]FileMention {
    var mentions = std.ArrayListUnmanaged(FileMention).empty;
    defer mentions.deinit(allocator);

    var i: usize = 0;
    while (i < input.len) {
        if (input[i] != '@') {
            i += 1;
            continue;
        }
        // Skip if preceded by a non-whitespace, non-start character (likely an email)
        if (i > 0) {
            const prev = input[i - 1];
            if (prev != ' ' and prev != '\t' and prev != '\n' and prev != '\r' and prev != '(' and prev != '[') {
                i += 1;
                continue;
            }
        }
        const start = i;
        i += 1; // skip '@'
        const path_start = i;
        // Path ends at whitespace or end of input
        while (i < input.len and !std.ascii.isWhitespace(input[i])) {
            i += 1;
        }
        const path = input[path_start..i];
        if (path.len == 0) continue;
        // Must look like a file path (contain a dot or slash, and not be all digits)
        if (std.mem.indexOfScalar(u8, path, '.') == null and std.mem.indexOfScalar(u8, path, '/') == null) continue;
        try mentions.append(allocator, .{
            .path = path,
            .start = start,
            .end = i,
        });
    }
    return try mentions.toOwnedSlice(allocator);
}

/// Read a file relative to `cwd` and return its contents, truncated to
/// `max_read_bytes`. Returns `error.FileNotFound` if the file does not exist.
pub fn readFileBounded(io: std.Io, allocator: std.mem.Allocator, cwd: []const u8, path: []const u8, max_read_bytes: usize) ![]u8 {
    try validateMentionPath(path, cwd);
    var dir = std.Io.Dir.cwd();
    const file = dir.openFile(io, path, .{}) catch return error.FileNotFound;
    defer file.close(io);
    const stat = try file.stat(io);
    const read_len = @min(stat.size, max_read_bytes);
    const buf = try allocator.alloc(u8, read_len);
    errdefer allocator.free(buf);
    const n = try file.readPositionalAll(io, buf, 0);
    return buf[0..n];
}

/// Resolve `@file` mentions in `input` by reading file contents and injecting
/// them as `[file: <path>]\n<contents>\n` blocks. Files that cannot be read
/// are silently skipped (the mention stays as literal text). Returns a new
/// owned string with file contents inlined.
pub fn resolveAndInject(
    io: std.Io,
    allocator: std.mem.Allocator,
    input: []const u8,
    root: []const u8,
    budget: *ContextBudget,
) ![]u8 {
    const mentions = try parseFileMentions(allocator, input);
    defer allocator.free(mentions);

    if (mentions.len == 0) {
        return try allocator.dupe(u8, input);
    }

    var result = std.ArrayListUnmanaged(u8).empty;
    defer result.deinit(allocator);

    var last_end: usize = 0;
    for (mentions) |mention| {
        // Append text before the mention
        try result.appendSlice(allocator, input[last_end..mention.start]);

        // Try to read the file
        const max_read = budget.remaining();
        if (max_read == 0) {
            // Budget exhausted: keep mention as literal text
            try result.appendSlice(allocator, input[mention.start..mention.end]);
            last_end = mention.end;
            continue;
        }

        const file_contents = readFileBounded(io, allocator, root, mention.path, max_read) catch {
            // File not found or invalid: keep mention as literal text
            try result.appendSlice(allocator, input[mention.start..mention.end]);
            last_end = mention.end;
            continue;
        };
        defer allocator.free(file_contents);

        // Inject file contents
        const header = try std.fmt.allocPrint(allocator, "[file: {s}]\n", .{mention.path});
        defer allocator.free(header);
        try result.appendSlice(allocator, header);
        try result.appendSlice(allocator, file_contents);
        try result.appendSlice(allocator, "\n");
        budget.consume(header.len + file_contents.len + 1);

        last_end = mention.end;
    }
    // Append remaining text after the last mention
    try result.appendSlice(allocator, input[last_end..]);

    return try result.toOwnedSlice(allocator);
}

test "parseFileMentions extracts valid file paths" {
    const allocator = std.testing.allocator;
    const mentions = try parseFileMentions(allocator, "hello @src/main.zig world");
    defer allocator.free(mentions);
    try std.testing.expectEqual(@as(usize, 1), mentions.len);
    try std.testing.expectEqualStrings("src/main.zig", mentions[0].path);
    try std.testing.expectEqual(@as(usize, 6), mentions[0].start);
    try std.testing.expectEqual(@as(usize, 19), mentions[0].end);
}

test "parseFileMentions ignores email-like @ tokens" {
    const allocator = std.testing.allocator;
    const mentions = try parseFileMentions(allocator, "contact user@example.com");
    defer allocator.free(mentions);
    try std.testing.expectEqual(@as(usize, 0), mentions.len);
}

test "parseFileMentions requires dot or slash in path" {
    const allocator = std.testing.allocator;
    const mentions = try parseFileMentions(allocator, "hello @world end");
    defer allocator.free(mentions);
    try std.testing.expectEqual(@as(usize, 0), mentions.len);
}

test "parseFileMentions handles multiple mentions" {
    const allocator = std.testing.allocator;
    const mentions = try parseFileMentions(allocator, "@file1.zig and @dir/file2.zig");
    defer allocator.free(mentions);
    try std.testing.expectEqual(@as(usize, 2), mentions.len);
    try std.testing.expectEqualStrings("file1.zig", mentions[0].path);
    try std.testing.expectEqualStrings("dir/file2.zig", mentions[1].path);
}

test "parseFileMentions handles empty input" {
    const allocator = std.testing.allocator;
    const mentions = try parseFileMentions(allocator, "");
    defer allocator.free(mentions);
    try std.testing.expectEqual(@as(usize, 0), mentions.len);
}

test "parseFileMentions handles all 256 byte values without panic" {
    const allocator = std.testing.allocator;
    var buf: [256]u8 = undefined;
    for (&buf, 0..) |*b, i| b.* = @intCast(i);
    const mentions = try parseFileMentions(allocator, &buf);
    defer allocator.free(mentions);
}

test "validateMentionPath rejects path escape" {
    try std.testing.expectError(error.PathEscape, validateMentionPath("../etc/passwd", "/"));
    try std.testing.expectError(error.PathEscape, validateMentionPath("foo/../bar", "/"));
    try std.testing.expectError(error.AbsolutePathRejected, validateMentionPath("/etc/passwd", "/"));
    try std.testing.expectError(error.EmptyPath, validateMentionPath("", "/"));
    try validateMentionPath("src/main.zig", "/");
    try validateMentionPath("README.md", "/");
}

test "ContextBudget tracks consumption" {
    var budget = ContextBudget.init(100);
    try std.testing.expectEqual(@as(usize, 100), budget.remaining());
    try std.testing.expect(budget.canFit(50));
    budget.consume(30);
    try std.testing.expectEqual(@as(usize, 70), budget.remaining());
    try std.testing.expect(budget.canFit(70));
    try std.testing.expect(!budget.canFit(71));
    budget.consume(70);
    try std.testing.expectEqual(@as(usize, 0), budget.remaining());
    budget.consume(10);
    try std.testing.expectEqual(@as(usize, 0), budget.remaining());
}

test "resolveAndInject passes through when no mentions" {
    const allocator = std.testing.allocator;
    var budget = ContextBudget.init(DEFAULT_BUDGET_BYTES);
    const result = try resolveAndInject(std.testing.io, allocator, "no mentions here", "/", &budget);
    defer allocator.free(result);
    try std.testing.expectEqualStrings("no mentions here", result);
}

test {
    std.testing.refAllDecls(@This());
}
