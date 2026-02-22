//! Persistent Ralph skills store (.ralph/skills.jsonl).

const std = @import("std");
const cfg = @import("config.zig");

fn nowEpochSeconds() i64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.REALTIME, &ts);
    return @intCast(ts.sec);
}

const SkillRecord = struct {
    content: []const u8,
    created_at: i64,
    source_run_id: ?[]const u8,
    score: f32,
};

pub fn appendSkill(
    allocator: std.mem.Allocator,
    io: std.Io,
    content: []const u8,
    source_run_id: ?[]const u8,
    score: f32,
) !void {
    const trimmed = std.mem.trim(u8, content, " \t\r\n");
    if (trimmed.len == 0) return;

    cfg.ensureDir(io, cfg.WORKSPACE_DIR);

    const previous = std.Io.Dir.cwd().readFileAlloc(
        io,
        cfg.SKILLS_FILE,
        allocator,
        .limited(2 * 1024 * 1024),
    ) catch try allocator.dupe(u8, "");
    defer allocator.free(previous);

    var line_buffer: std.ArrayListUnmanaged(u8) = .empty;
    defer line_buffer.deinit(allocator);
    var aw: std.Io.Writer.Allocating = .fromArrayList(allocator, &line_buffer);
    defer line_buffer = aw.toArrayList();
    try std.json.Stringify.value(
        SkillRecord{
            .content = trimmed,
            .created_at = nowEpochSeconds(),
            .source_run_id = source_run_id,
            .score = score,
        },
        .{},
        &aw.writer,
    );

    const next_contents = try std.fmt.allocPrint(
        allocator,
        "{s}{s}\n",
        .{ previous, line_buffer.items },
    );
    defer allocator.free(next_contents);

    try cfg.writeFile(allocator, io, cfg.SKILLS_FILE, next_contents);
}

pub fn clearSkills(allocator: std.mem.Allocator, io: std.Io) !void {
    cfg.ensureDir(io, cfg.WORKSPACE_DIR);
    try cfg.writeFile(allocator, io, cfg.SKILLS_FILE, "");
}

pub fn countSkills(allocator: std.mem.Allocator, io: std.Io) usize {
    const contents = std.Io.Dir.cwd().readFileAlloc(
        io,
        cfg.SKILLS_FILE,
        allocator,
        .limited(2 * 1024 * 1024),
    ) catch return 0;
    defer allocator.free(contents);

    var count: usize = 0;
    var lines = std.mem.splitScalar(u8, contents, '\n');
    while (lines.next()) |line| {
        if (std.mem.trim(u8, line, " \t\r").len > 0) count += 1;
    }
    return count;
}

pub fn loadSkillsContext(
    allocator: std.mem.Allocator,
    io: std.Io,
    max_chars: usize,
) ![]u8 {
    const contents = std.Io.Dir.cwd().readFileAlloc(
        io,
        cfg.SKILLS_FILE,
        allocator,
        .limited(2 * 1024 * 1024),
    ) catch return allocator.dupe(u8, "");
    defer allocator.free(contents);

    var all_skills = std.ArrayListUnmanaged([]u8).empty;
    defer {
        for (all_skills.items) |skill| allocator.free(skill);
        all_skills.deinit(allocator);
    }

    var lines = std.mem.splitScalar(u8, contents, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (trimmed.len == 0) continue;

        const parsed = std.json.parseFromSlice(std.json.Value, allocator, trimmed, .{}) catch continue;
        defer parsed.deinit();
        const obj = switch (parsed.value) {
            .object => |o| o,
            else => continue,
        };
        const skill_text = if (obj.get("content")) |value| switch (value) {
            .string => |s| s,
            else => "",
        } else "";
        if (skill_text.len == 0) continue;

        try all_skills.append(allocator, try allocator.dupe(u8, skill_text));
    }

    if (all_skills.items.len == 0) return allocator.dupe(u8, "");

    var out = std.ArrayListUnmanaged(u8).empty;
    defer out.deinit(allocator);
    try out.appendSlice(allocator, "\n\n[Persisted Ralph Skills]\n");

    var used: usize = out.items.len;
    var idx = all_skills.items.len;
    while (idx > 0) {
        idx -= 1;
        const skill = all_skills.items[idx];
        const line_len = skill.len + 3;
        if (used + line_len > max_chars) break;
        try out.appendSlice(allocator, "- ");
        try out.appendSlice(allocator, skill);
        try out.appendSlice(allocator, "\n");
        used += line_len;
    }

    return out.toOwnedSlice(allocator);
}

pub fn listSkills(allocator: std.mem.Allocator, io: std.Io) ![]u8 {
    const contents = std.Io.Dir.cwd().readFileAlloc(
        io,
        cfg.SKILLS_FILE,
        allocator,
        .limited(2 * 1024 * 1024),
    ) catch return allocator.dupe(u8, "");
    defer allocator.free(contents);

    var out = std.ArrayListUnmanaged(u8).empty;
    defer out.deinit(allocator);

    var lines = std.mem.splitScalar(u8, contents, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (trimmed.len == 0) continue;

        const parsed = std.json.parseFromSlice(std.json.Value, allocator, trimmed, .{}) catch continue;
        defer parsed.deinit();
        const obj = switch (parsed.value) {
            .object => |o| o,
            else => continue,
        };
        const skill_text = if (obj.get("content")) |value| switch (value) {
            .string => |s| s,
            else => "",
        } else "";
        if (skill_text.len == 0) continue;
        try out.appendSlice(allocator, "- ");
        try out.appendSlice(allocator, skill_text);
        try out.appendSlice(allocator, "\n");
    }

    return out.toOwnedSlice(allocator);
}
