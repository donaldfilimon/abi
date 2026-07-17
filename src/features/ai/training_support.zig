const std = @import("std");
const builtin = @import("builtin");
const foundation_io = @import("../../foundation/io/mod.zig");
const core_memory = @import("../../core/memory.zig");
const temp_path = @import("../../foundation/temp_path.zig");
const env = @import("../../foundation/env.zig");
const helpers = @import("helpers.zig");
const types = @import("types.zig");
const point_neural_net = @import("point_neural_net.zig");

const Point = point_neural_net.Point;

/// Optional absolute root for dataset/artifact paths (TM-003). When unset,
/// the process cwd is the confine root. Absolute paths outside the root and
/// symlink escapes that resolve outside the root are rejected.
pub const TRAIN_DATA_ROOT_ENV = "ABI_TRAIN_DATA_ROOT";

pub const DatasetSummary = struct {
    available: bool = false,
    records: usize = 0,
    bytes: usize = 0,
};

pub fn validateTrainingConfig(config: types.TrainingConfig) !void {
    if (config.profile.len == 0) return error.InvalidTrainingProfile;
    _ = parseAgentProfile(config.profile) catch return error.InvalidTrainingProfile;
    if (config.dataset.path.len == 0) return error.InvalidDatasetPath;
    if (config.artifact_dir.len == 0) return error.InvalidArtifactPath;
}

/// Profile + non-empty path checks, then confine dataset/artifact under the
/// training data root (cwd or `ABI_TRAIN_DATA_ROOT`).
pub fn validateTrainingConfigConfined(allocator: std.mem.Allocator, config: types.TrainingConfig) !void {
    try validateTrainingConfig(config);
    const root = try trainingDataRoot(allocator);
    defer allocator.free(root);
    try confineTrainingPath(allocator, config.dataset.path, root);
    try confineTrainingPath(allocator, config.artifact_dir, root);
}

/// Resolve the training data root: env override if set, else absolute cwd.
/// Caller owns the returned slice (free with `allocator.free`).
pub fn trainingDataRoot(allocator: std.mem.Allocator) ![]u8 {
    if (env.get(TRAIN_DATA_ROOT_ENV)) |root| {
        if (root.len == 0) return error.InvalidTrainingDataRoot;
        return try resolveExistingOrAbsolute(allocator, root);
    }
    // currentPathAlloc returns a sentinel slice; re-dupe as a plain `[]u8` so
    // callers can free with `allocator.free` without sentinel size skew.
    const cwd_z = try std.process.currentPathAlloc(std.Options.debug_io, allocator);
    defer allocator.free(cwd_z);
    return try allocator.dupe(u8, cwd_z);
}

/// Reject `..` segments, absolute paths outside `root`, and realpath escapes
/// (symlink to a location outside the root). `root` must be absolute.
pub fn confineTrainingPath(allocator: std.mem.Allocator, path: []const u8, root: []const u8) !void {
    if (path.len == 0) return error.InvalidDatasetPath;
    try rejectDotDot(path);

    const candidate = if (std.fs.path.isAbsolute(path))
        try allocator.dupe(u8, path)
    else if (root.len > 0 and (root[root.len - 1] == '/' or (builtin.target.os.tag == .windows and root[root.len - 1] == '\\')))
        try std.fmt.allocPrint(allocator, "{s}{s}", .{ root, path })
    else
        try std.fmt.allocPrint(allocator, "{s}/{s}", .{ root, path });
    defer allocator.free(candidate);

    // Prefer realpath so symlink escapes surface as PathOutsideRoot.
    var resolved_buf: [foundation_io.MAX_PATH_LEN]u8 = undefined;
    if (std.Io.Dir.realPathFile(.cwd(), std.Options.debug_io, candidate, &resolved_buf)) |len| {
        if (!pathIsUnderRoot(resolved_buf[0..len], root)) return error.PathOutsideRoot;
        return;
    } else |_| {
        // Path does not exist yet (common for artifact_dir): check lexical prefix.
        if (!pathIsUnderRoot(candidate, root)) return error.PathOutsideRoot;
    }
}

fn rejectDotDot(path: []const u8) !void {
    var it = std.mem.splitAny(u8, path, "/\\");
    while (it.next()) |segment| {
        if (std.mem.eql(u8, segment, "..")) return error.PathTraversal;
    }
}

fn pathIsUnderRoot(path: []const u8, root: []const u8) bool {
    if (path.len < root.len) return false;
    if (!std.mem.startsWith(u8, path, root)) return false;
    if (path.len == root.len) return true;
    const next = path[root.len];
    if (next == '/') return true;
    if (builtin.target.os.tag == .windows and next == '\\') return true;
    // Avoid `/tmp/evil` matching root `/tmp/ev`.
    return false;
}

fn resolveExistingOrAbsolute(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    var buf: [foundation_io.MAX_PATH_LEN]u8 = undefined;
    if (std.Io.Dir.realPathFile(.cwd(), std.Options.debug_io, path, &buf)) |len| {
        return try allocator.dupe(u8, buf[0..len]);
    } else |_| {
        if (std.fs.path.isAbsolute(path)) return try allocator.dupe(u8, path);
        return error.InvalidTrainingDataRoot;
    }
}

pub fn inspectDataset(allocator: std.mem.Allocator, dataset: types.DatasetSpec) !DatasetSummary {
    return inspectDatasetTracked(allocator, dataset, null);
}

pub fn inspectDatasetTracked(
    allocator: std.mem.Allocator,
    dataset: types.DatasetSpec,
    tracker: ?*core_memory.MemoryTracker,
) !DatasetSummary {
    if (tracker) |t| {
        var tracking = core_memory.TrackingAllocator.init(allocator, t);
        return inspectDatasetWithAllocator(tracking.allocator(), dataset);
    }
    return inspectDatasetWithAllocator(allocator, dataset);
}

fn inspectDatasetWithAllocator(allocator: std.mem.Allocator, dataset: types.DatasetSpec) !DatasetSummary {
    const path = foundation_io.resolvePath(allocator, dataset.path) catch |err| switch (err) {
        error.FileNotFound => return .{ .available = false, .records = 0, .bytes = dataset.path.len },
        else => return err,
    };
    defer allocator.free(path);

    const data = foundation_io.asyncReadFile(allocator, path) catch |err| switch (err) {
        error.FileNotFound => return .{ .available = false, .records = 0, .bytes = dataset.path.len },
        else => return err,
    };
    defer allocator.free(data);

    return .{
        .available = true,
        .records = try countDatasetRecords(allocator, dataset.format, data),
        .bytes = data.len,
    };
}

pub fn parseAgentProfile(name: []const u8) !types.AgentProfile {
    inline for (types.known_profiles) |p| {
        if (std.mem.eql(u8, name, p.label())) return p;
    }
    return error.UnknownAgentProfile;
}

pub fn profileEmbedding(agent: types.AgentProfile) [helpers.EMBED_DIM]f32 {
    // Derive each persona's signature vector from its label via the shared
    // embedding, so profile vectors share the dimensionality and feature space
    // of every other stored vector (no dimension mismatch across train/complete).
    return helpers.textEmbedding(agent.label());
}

fn countDatasetRecords(allocator: std.mem.Allocator, format: types.DatasetFormat, data: []const u8) !usize {
    return switch (format) {
        .text => helpers.countNonEmptyLines(data),
        .csv => blk: {
            const lines = helpers.countNonEmptyLines(data);
            break :blk if (lines > 0) lines - 1 else 0;
        },
        .jsonl => try countJsonlRecords(allocator, data),
    };
}

fn countJsonlRecords(allocator: std.mem.Allocator, data: []const u8) !usize {
    var records: usize = 0;
    var lines = std.mem.splitScalar(u8, data, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (trimmed.len == 0) continue;
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, trimmed, .{});
        defer parsed.deinit();
        records += 1;
    }
    return records;
}

/// Extract a single training text from a parsed JSON value. Prefers the
/// `input`/`text`/`content` string fields; a bare JSON string is used directly;
/// anything else yields null (caller skips the record).
fn extractTextFromJson(allocator: std.mem.Allocator, value: std.json.Value) !?[]const u8 {
    switch (value) {
        .string => |s| return try allocator.dupe(u8, s),
        .object => |obj| {
            if (obj.get("input")) |v| if (v == .string) return try allocator.dupe(u8, v.string);
            if (obj.get("text")) |v| if (v == .string) return try allocator.dupe(u8, v.string);
            if (obj.get("content")) |v| if (v == .string) return try allocator.dupe(u8, v.string);
            return null;
        },
        else => return null,
    }
}

/// Read a training dataset and turn each record into a 3-D `Point` via
/// `Point.fromText`. Returns `null` when the file is missing, empty, or yields
/// no parseable text records. Caller owns the returned slice of `Point`s.
pub fn datasetToPoints(allocator: std.mem.Allocator, dataset: types.DatasetSpec) !?[]Point {
    const path = foundation_io.resolvePath(allocator, dataset.path) catch return null;
    defer allocator.free(path);
    const data = foundation_io.asyncReadFile(allocator, path) catch return null;
    defer allocator.free(data);

    var texts = std.ArrayListUnmanaged([]const u8).empty;
    errdefer {
        for (texts.items) |t| allocator.free(t);
        texts.deinit(allocator);
    }

    switch (dataset.format) {
        .text => {
            var lines = std.mem.splitScalar(u8, data, '\n');
            while (lines.next()) |line| {
                const trimmed = std.mem.trim(u8, line, " \t\r");
                if (trimmed.len == 0) continue;
                try texts.append(allocator, try allocator.dupe(u8, trimmed));
            }
        },
        .jsonl => {
            var lines = std.mem.splitScalar(u8, data, '\n');
            while (lines.next()) |line| {
                const trimmed = std.mem.trim(u8, line, " \t\r");
                if (trimmed.len == 0) continue;
                const parsed = std.json.parseFromSlice(std.json.Value, allocator, trimmed, .{}) catch continue;
                const text = extractTextFromJson(allocator, parsed.value) catch {
                    parsed.deinit();
                    continue;
                };
                parsed.deinit();
                if (text) |t| try texts.append(allocator, t);
            }
        },
        .csv => {
            var lines = std.mem.splitScalar(u8, data, '\n');
            var first = true;
            while (lines.next()) |line| {
                const trimmed = std.mem.trim(u8, line, " \t\r");
                if (trimmed.len == 0) continue;
                if (first) {
                    first = false;
                    continue; // header row
                }
                var cell_iter = std.mem.splitScalar(u8, trimmed, ',');
                const cell = std.mem.trim(u8, cell_iter.first(), " \"");
                if (cell.len == 0) continue;
                try texts.append(allocator, try allocator.dupe(u8, cell));
            }
        },
    }

    if (texts.items.len == 0) return null;

    const points = try allocator.alloc(Point, texts.items.len);
    errdefer allocator.free(points);
    for (texts.items, 0..) |t, i| points[i] = Point.fromText(t);
    for (texts.items) |t| allocator.free(t);
    texts.deinit(allocator);
    return points;
}

test "profile parsing accepts known profiles" {
    try std.testing.expectEqual(types.AgentProfile.abbey, try parseAgentProfile("abbey"));
    try std.testing.expectEqual(types.AgentProfile.aviva, try parseAgentProfile("aviva"));
    try std.testing.expectEqual(types.AgentProfile.abi, try parseAgentProfile("abi"));
    try std.testing.expectError(error.UnknownAgentProfile, parseAgentProfile("unknown"));
}

test "dataset record counting handles text and csv" {
    try std.testing.expectEqual(@as(usize, 2), try countDatasetRecords(std.testing.allocator, .text, "a\n\nb\n"));
    try std.testing.expectEqual(@as(usize, 2), try countDatasetRecords(std.testing.allocator, .csv, "h\n1\n2\n"));
}

test "dataset record counting handles jsonl and surfaces malformed lines" {
    const allocator = std.testing.allocator;
    // Valid JSONL: one object per non-empty line; blank/whitespace lines skipped.
    try std.testing.expectEqual(@as(usize, 2), try countDatasetRecords(allocator, .jsonl, "{\"a\":1}\n\n{\"b\":2}\n"));
    try std.testing.expectEqual(@as(usize, 0), try countDatasetRecords(allocator, .jsonl, "\n  \n"));
    // Current contract: a malformed JSONL line aborts the whole count (the parse
    // error propagates) rather than being silently skipped. Pinned so a future
    // skip-malformed change is a deliberate decision, not an accidental drift.
    try std.testing.expect(if (countDatasetRecords(allocator, .jsonl, "{\"a\":1}\nnot json\n")) |_| false else |_| true);
}

test "pathIsUnderRoot requires separator boundary" {
    try std.testing.expect(pathIsUnderRoot("/data/train", "/data"));
    try std.testing.expect(pathIsUnderRoot("/data", "/data"));
    try std.testing.expect(!pathIsUnderRoot("/data-evil/x", "/data"));
    try std.testing.expect(!pathIsUnderRoot("/etc/passwd", "/data"));
}

test "confineTrainingPath rejects traversal absolute outside and accepts under root" {
    const allocator = std.testing.allocator;
    const root = try trainingDataRoot(allocator);
    defer allocator.free(root);

    try std.testing.expectError(error.PathTraversal, confineTrainingPath(allocator, "../secret", root));
    try std.testing.expectError(error.PathOutsideRoot, confineTrainingPath(allocator, "/etc/passwd", root));

    // Relative paths are joined under the root and accepted even when missing.
    try confineTrainingPath(allocator, "zig-cache/agents", root);
    try confineTrainingPath(allocator, "datasets/local-training.jsonl", root);
}

test "confineTrainingPath rejects symlink escape when present" {
    if (builtin.target.os.tag == .windows) return;

    const allocator = std.testing.allocator;
    const root = try temp_path.tempFilePath(allocator, "abi_train_root", "dir");
    defer allocator.free(root);
    // tempFilePath returns a file-shaped name; create it as a directory root.
    std.Io.Dir.createDirPath(.cwd(), std.Options.debug_io, root) catch {
        std.Io.Dir.deleteFileAbsolute(std.Options.debug_io, root) catch |err| std.log.warn("training_support setup: {s}", .{@errorName(err)});
        try std.Io.Dir.createDirPath(.cwd(), std.Options.debug_io, root);
    };
    defer std.Io.Dir.deleteTree(.cwd(), std.Options.debug_io, root) catch |err| std.log.warn("training_support cleanup: {s}", .{@errorName(err)});

    const link_path = try std.fmt.allocPrint(allocator, "{s}/escape", .{root});
    defer allocator.free(link_path);
    // Point a symlink at a location outside the root.
    std.Io.Dir.symLinkAbsolute(std.Options.debug_io, "/etc", link_path, .{}) catch return;
    defer std.Io.Dir.deleteFileAbsolute(std.Options.debug_io, link_path) catch |err| std.log.warn("training_support test cleanup: {s}", .{@errorName(err)});

    try std.testing.expectError(error.PathOutsideRoot, confineTrainingPath(allocator, "escape", root));
    // Direct child name under root (non-symlink) is fine when missing or present.
    try confineTrainingPath(allocator, "artifacts", root);
}

test "inspectDatasetTracked accounts read and JSON parse transients" {
    const allocator = std.testing.allocator;
    const path = try temp_path.tempFilePath(allocator, "abi_training_support_tracker", "jsonl");
    defer allocator.free(path);
    defer std.Io.Dir.deleteFileAbsolute(std.testing.io, path) catch |err| switch (err) {
        error.FileNotFound => {},
        else => std.log.warn("training_support test cleanup failed: {s}", .{@errorName(err)}),
    };

    try foundation_io.asyncWriteFile(path, "{\"input\":\"a\"}\n{\"input\":\"b\"}\n");

    var tracker = core_memory.MemoryTracker.init(allocator);
    defer tracker.deinit();

    const summary = try inspectDatasetTracked(allocator, .{ .path = path, .format = .jsonl }, &tracker);

    try std.testing.expect(summary.available);
    try std.testing.expectEqual(@as(usize, 2), summary.records);
    try std.testing.expect(summary.bytes > 0);
    try std.testing.expect(tracker.getTotalAllocated() > 0);
    try std.testing.expectEqual(tracker.getTotalAllocated(), tracker.getTotalFreed());
    try std.testing.expectEqual(@as(usize, 0), tracker.getCurrentUsage());
    try std.testing.expectEqual(@as(usize, 0), tracker.getRecordCount());
}

test "datasetToPoints parses jsonl/text/csv records into points" {
    const allocator = std.testing.allocator;

    const jsonl = try temp_path.tempFilePath(allocator, "abi_ds_jsonl", "jsonl");
    defer allocator.free(jsonl);
    defer std.Io.Dir.deleteFileAbsolute(std.testing.io, jsonl) catch |err| std.log.warn("training_support test cleanup: {s}", .{@errorName(err)});
    try foundation_io.asyncWriteFile(jsonl, "{\"input\":\"hello world\"}\n{\"input\":\"foo bar\"}\n");
    const jp = (try datasetToPoints(allocator, .{ .path = jsonl, .format = .jsonl })).?;
    defer allocator.free(jp);
    try std.testing.expectEqual(@as(usize, 2), jp.len);

    const text = try temp_path.tempFilePath(allocator, "abi_ds_text", "txt");
    defer allocator.free(text);
    defer std.Io.Dir.deleteFileAbsolute(std.testing.io, text) catch |err| std.log.warn("training_support test cleanup: {s}", .{@errorName(err)});
    try foundation_io.asyncWriteFile(text, "alpha\nbeta\ngamma\n");
    const tp = (try datasetToPoints(allocator, .{ .path = text, .format = .text })).?;
    defer allocator.free(tp);
    try std.testing.expectEqual(@as(usize, 3), tp.len);

    const csv = try temp_path.tempFilePath(allocator, "abi_ds_csv", "csv");
    defer allocator.free(csv);
    defer std.Io.Dir.deleteFileAbsolute(std.testing.io, csv) catch |err| std.log.warn("training_support test cleanup: {s}", .{@errorName(err)});
    try foundation_io.asyncWriteFile(csv, "text\nred\ngreen\n");
    const cp = (try datasetToPoints(allocator, .{ .path = csv, .format = .csv })).?;
    defer allocator.free(cp);
    try std.testing.expectEqual(@as(usize, 2), cp.len);

    // Missing file yields null.
    try std.testing.expect((try datasetToPoints(allocator, .{ .path = "/no/such/file.jsonl", .format = .jsonl })) == null);
}

test {
    std.testing.refAllDecls(@This());
}
