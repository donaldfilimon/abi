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

/// Caps for bounded workspace tree listings (cwd-sandboxed).
pub const TREE_MAX_DEPTH: usize = 3;
pub const TREE_MAX_ENTRIES: usize = 64;
pub const TREE_DEFAULT_BUDGET_BYTES: usize = 2048;
pub const GIT_DIFF_DEFAULT_BUDGET_BYTES: usize = 2048;

/// Priority buckets for multi-source context. Higher-priority sources consume
/// first; leftovers may be used by lower tiers when a tier underspends.
pub const BudgetShares = struct {
    open_bytes: usize,
    at_file_bytes: usize,
    tree_bytes: usize,
    git_bytes: usize,

    /// Split `total` with priority open > @file > tree > git (40/35/15/10).
    pub fn fairShare(total: usize) BudgetShares {
        if (total == 0) return .{ .open_bytes = 0, .at_file_bytes = 0, .tree_bytes = 0, .git_bytes = 0 };
        const open = (total * 40) / 100;
        const at_file = (total * 35) / 100;
        const tree = (total * 15) / 100;
        const git = total - open - at_file - tree;
        return .{ .open_bytes = open, .at_file_bytes = at_file, .tree_bytes = tree, .git_bytes = git };
    }
};

fn shouldSkipTreeName(name: []const u8) bool {
    if (name.len == 0 or name[0] == '.') return true;
    if (std.mem.eql(u8, name, "zig-cache") or std.mem.eql(u8, name, ".zig-cache")) return true;
    if (std.mem.eql(u8, name, "zig-out") or std.mem.eql(u8, name, "node_modules")) return true;
    return false;
}

/// Recursively list files under `root` (relative paths), capped by depth and
/// entry count. Paths are cwd-sandboxed (no `..` / absolute escape). Returns an
/// owned multi-line listing; caller frees.
pub fn listWorkspaceTree(
    io: std.Io,
    allocator: std.mem.Allocator,
    root: []const u8,
    max_depth: usize,
    max_entries: usize,
) ![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    var count: usize = 0;
    try walkTree(io, allocator, root, "", 0, max_depth, max_entries, &count, &out);
    if (out.items.len == 0) {
        try out.appendSlice(allocator, "(empty tree)\n");
    }
    return try out.toOwnedSlice(allocator);
}

fn walkTree(
    io: std.Io,
    allocator: std.mem.Allocator,
    root: []const u8,
    rel: []const u8,
    depth: usize,
    max_depth: usize,
    max_entries: usize,
    count: *usize,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    if (depth > max_depth or count.* >= max_entries) return;

    // `rel` is a cwd-relative path (includes `root` after the first level).
    const open_path = if (rel.len == 0) root else rel;
    var dir = try std.Io.Dir.cwd().openDir(io, open_path, .{ .iterate = true });
    defer dir.close(io);

    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (count.* >= max_entries) break;
        if (shouldSkipTreeName(entry.name)) continue;

        const child_rel = if (rel.len == 0)
            (if (std.mem.eql(u8, root, "."))
                try allocator.dupe(u8, entry.name)
            else
                try std.fmt.allocPrint(allocator, "{s}/{s}", .{ root, entry.name }))
        else
            try std.fmt.allocPrint(allocator, "{s}/{s}", .{ rel, entry.name });
        defer allocator.free(child_rel);

        try validateMentionPath(child_rel, ".");

        switch (entry.kind) {
            .directory => {
                if (depth >= max_depth) continue;
                try walkTree(io, allocator, root, child_rel, depth + 1, max_depth, max_entries, count, out);
            },
            .file => {
                try out.appendSlice(allocator, child_rel);
                try out.append(allocator, '\n');
                count.* += 1;
            },
            else => {},
        }
    }
}

/// Truncate `text` to `max_bytes`, preferring a trailing newline boundary.
fn truncateToBudget(allocator: std.mem.Allocator, text: []const u8, max_bytes: usize) ![]u8 {
    if (text.len <= max_bytes) return try allocator.dupe(u8, text);
    var cut = max_bytes;
    while (cut > 0 and text[cut - 1] != '\n') cut -= 1;
    if (cut == 0) cut = max_bytes;
    return try allocator.dupe(u8, text[0..cut]);
}

/// Run `git diff --stat` (or full `git diff --color=never`) and return owned
/// stdout truncated to `max_bytes`. Missing git / non-repo → empty string.
pub fn readGitDiffBudgeted(
    io: std.Io,
    allocator: std.mem.Allocator,
    max_bytes: usize,
    stat_only: bool,
) ![]u8 {
    if (max_bytes == 0) return try allocator.dupe(u8, "");

    const argv: []const []const u8 = if (stat_only)
        &.{ "git", "diff", "--stat" }
    else
        &.{ "git", "diff", "--color=never" };

    var child = std.process.spawn(io, .{
        .argv = argv,
        .cwd = .inherit,
        .stdin = .ignore,
        .stdout = .pipe,
        .stderr = .ignore,
    }) catch return try allocator.dupe(u8, "");
    defer child.kill(io);

    var output = std.ArrayListUnmanaged(u8).empty;
    defer output.deinit(allocator);
    var buf: [4096]u8 = undefined;
    while (true) {
        const n = std.Io.File.readStreaming(child.stdout.?, io, &.{&buf}) catch break;
        if (n == 0) break;
        try output.appendSlice(allocator, buf[0..n]);
        if (output.items.len >= max_bytes) break;
    }
    _ = child.wait(io) catch {};

    return try truncateToBudget(allocator, output.items, max_bytes);
}

pub const AgentContextOptions = struct {
    include_tree: bool = true,
    include_git_diff: bool = true,
    git_stat_only: bool = true,
    open_path: []const u8 = "",
    open_content: []const u8 = "",
    tree_max_depth: usize = TREE_MAX_DEPTH,
    tree_max_entries: usize = TREE_MAX_ENTRIES,
};

/// Build a budgeted agent context: open file (priority) + `@file` mentions +
/// optional workspace tree + optional git diff/--stat. Returns owned text.
pub fn buildAgentContext(
    io: std.Io,
    allocator: std.mem.Allocator,
    input: []const u8,
    root: []const u8,
    total_budget: usize,
    opts: AgentContextOptions,
) ![]u8 {
    const shares = BudgetShares.fairShare(total_budget);
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    // 1. Open file (highest priority)
    var open_used: usize = 0;
    if (opts.open_path.len > 0 and opts.open_content.len > 0 and shares.open_bytes > 0) {
        const body = try truncateToBudget(allocator, opts.open_content, shares.open_bytes);
        defer allocator.free(body);
        const header = try std.fmt.allocPrint(allocator, "[open: {s}]\n", .{opts.open_path});
        defer allocator.free(header);
        try out.appendSlice(allocator, header);
        try out.appendSlice(allocator, body);
        try out.append(allocator, '\n');
        open_used = header.len + body.len + 1;
    }

    // 2. @file mentions
    var at_budget = ContextBudget.init(shares.at_file_bytes + (shares.open_bytes -| open_used));
    const injected = try resolveAndInject(io, allocator, input, root, &at_budget);
    defer allocator.free(injected);
    try out.appendSlice(allocator, injected);

    // 3. Workspace tree
    if (opts.include_tree and shares.tree_bytes > 0) {
        const tree = listWorkspaceTree(io, allocator, root, opts.tree_max_depth, opts.tree_max_entries) catch null;
        if (tree) |listing| {
            defer allocator.free(listing);
            const clipped = try truncateToBudget(allocator, listing, shares.tree_bytes);
            defer allocator.free(clipped);
            try out.appendSlice(allocator, "[workspace-tree]\n");
            try out.appendSlice(allocator, clipped);
            if (clipped.len == 0 or clipped[clipped.len - 1] != '\n') try out.append(allocator, '\n');
        }
    }

    // 4. Git diff / --stat
    if (opts.include_git_diff and shares.git_bytes > 0) {
        const diff = try readGitDiffBudgeted(io, allocator, shares.git_bytes, opts.git_stat_only);
        defer allocator.free(diff);
        if (diff.len > 0) {
            try out.appendSlice(allocator, if (opts.git_stat_only) "[git-diff-stat]\n" else "[git-diff]\n");
            try out.appendSlice(allocator, diff);
            if (diff[diff.len - 1] != '\n') try out.append(allocator, '\n');
        }
    }

    return try out.toOwnedSlice(allocator);
}

test "BudgetShares.fairShare partitions total with open priority" {
    const s = BudgetShares.fairShare(1000);
    try std.testing.expectEqual(@as(usize, 400), s.open_bytes);
    try std.testing.expectEqual(@as(usize, 350), s.at_file_bytes);
    try std.testing.expectEqual(@as(usize, 150), s.tree_bytes);
    try std.testing.expectEqual(@as(usize, 100), s.git_bytes);
    try std.testing.expectEqual(@as(usize, 1000), s.open_bytes + s.at_file_bytes + s.tree_bytes + s.git_bytes);
}

test "listWorkspaceTree returns sandboxed relative paths" {
    const allocator = std.testing.allocator;
    const listing = try listWorkspaceTree(std.testing.io, allocator, "src/features/ai", 1, 8);
    defer allocator.free(listing);
    try std.testing.expect(listing.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, listing, "..") == null);
}

test "truncateToBudget respects max bytes" {
    const allocator = std.testing.allocator;
    const clipped = try truncateToBudget(allocator, "aaa\nbbb\nccc\n", 8);
    defer allocator.free(clipped);
    try std.testing.expect(clipped.len <= 8);
}

test {
    std.testing.refAllDecls(@This());
}
