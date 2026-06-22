const std = @import("std");
const foundation_io = @import("../../foundation/io/mod.zig");
const helpers = @import("helpers.zig");
const types = @import("types.zig");

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

pub fn inspectDataset(allocator: std.mem.Allocator, dataset: types.DatasetSpec) !DatasetSummary {
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

test {
    std.testing.refAllDecls(@This());
}
