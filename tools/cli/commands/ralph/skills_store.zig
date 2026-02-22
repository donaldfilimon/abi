//! Persistent Ralph skills store (.ralph/skills.jsonl).
//!
//! Skills are domain-tagged lessons learned from Ralph iterations.
//! Domain inference + deduplication prevent unbounded growth.

const std = @import("std");
const cfg = @import("config.zig");

fn nowEpochSeconds() i64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.REALTIME, &ts);
    return @intCast(ts.sec);
}

pub const SkillDomain = enum {
    general,
    zig,
    ai,
    gpu,
    cli,
    tui,
    build,
    test_infra,
    memory,
    network,

    pub fn label(self: SkillDomain) []const u8 {
        return switch (self) {
            .general => "general",
            .zig => "zig",
            .ai => "ai",
            .gpu => "gpu",
            .cli => "cli",
            .tui => "tui",
            .build => "build",
            .test_infra => "test",
            .memory => "memory",
            .network => "network",
        };
    }
};

const SkillRecord = struct {
    content: []const u8,
    created_at: i64,
    source_run_id: ?[]const u8,
    score: f32,
    domain: []const u8 = "general",
};

/// Infer domain from skill text by keyword matching.
pub fn inferDomain(text: []const u8) SkillDomain {
    const domain_keywords = [_]struct { domain: SkillDomain, keywords: []const []const u8 }{
        .{ .domain = .tui, .keywords = &.{ "tui", "dashboard", "terminal", "panel", "render", "unicode" } },
        .{ .domain = .gpu, .keywords = &.{ "gpu", "metal", "vulkan", "cuda", "backend", "shader" } },
        .{ .domain = .ai, .keywords = &.{ "abbey", "llm", "agent", "training", "reasoning", "inference", "model" } },
        .{ .domain = .cli, .keywords = &.{ "cli", "command", "subcommand", "arg", "flag", "descriptor" } },
        .{ .domain = .build, .keywords = &.{ "build", "compile", "link", "flag", "validate-flags", "feature-tests" } },
        .{ .domain = .test_infra, .keywords = &.{ "test", "baseline", "parity", "stub", "verify" } },
        .{ .domain = .zig, .keywords = &.{ "zig", "0.16", "std.Io", "allocator", "comptime", "errdefer" } },
        .{ .domain = .memory, .keywords = &.{ "memory", "allocat", "leak", "arena", "free", "deinit" } },
        .{ .domain = .network, .keywords = &.{ "network", "socket", "http", "tcp", "server", "stream" } },
    };

    var best_domain: SkillDomain = .general;
    var best_hits: usize = 0;

    for (domain_keywords) |entry| {
        var hits: usize = 0;
        for (entry.keywords) |kw| {
            if (cfg.containsIgnoreCase(text, kw)) hits += 1;
        }
        if (hits > best_hits) {
            best_hits = hits;
            best_domain = entry.domain;
        }
    }

    return best_domain;
}

/// Check if a skill is a near-duplicate of existing skills.
fn isDuplicate(
    allocator: std.mem.Allocator,
    io: std.Io,
    new_content: []const u8,
) bool {
    const contents = std.Io.Dir.cwd().readFileAlloc(
        io,
        cfg.SKILLS_FILE,
        allocator,
        .limited(2 * 1024 * 1024),
    ) catch return false;
    defer allocator.free(contents);

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
        const existing = if (obj.get("content")) |value| switch (value) {
            .string => |s| s,
            else => continue,
        } else continue;

        // Exact match
        if (std.mem.eql(u8, existing, new_content)) return true;

        // Prefix overlap (one is substring of other at 80%+ length)
        const shorter = @min(existing.len, new_content.len);
        const longer = @max(existing.len, new_content.len);
        if (longer > 0 and shorter * 100 / longer >= 80) {
            const prefix_len = @min(shorter, longer);
            var matching: usize = 0;
            for (0..prefix_len) |i| {
                if (std.ascii.toLower(existing[@min(i, existing.len - 1)]) ==
                    std.ascii.toLower(new_content[@min(i, new_content.len - 1)]))
                    matching += 1;
            }
            if (prefix_len > 0 and matching * 100 / prefix_len >= 85) return true;
        }
    }
    return false;
}

pub fn appendSkill(
    allocator: std.mem.Allocator,
    io: std.Io,
    content: []const u8,
    source_run_id: ?[]const u8,
    score: f32,
) !void {
    const trimmed = std.mem.trim(u8, content, " \t\r\n");
    if (trimmed.len == 0) return;

    // Skip very short or generic skills
    if (trimmed.len < 10) return;

    cfg.ensureDir(io, cfg.WORKSPACE_DIR);

    // Deduplicate — skip if near-duplicate exists
    if (isDuplicate(allocator, io, trimmed)) return;

    const domain = inferDomain(trimmed);

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
            .domain = domain.label(),
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
    return loadSkillsContextFiltered(allocator, io, max_chars, null, 0.0);
}

/// Load skills with optional domain filter and minimum score threshold.
pub fn loadSkillsContextFiltered(
    allocator: std.mem.Allocator,
    io: std.Io,
    max_chars: usize,
    domain_filter: ?SkillDomain,
    min_score: f32,
) ![]u8 {
    const contents = std.Io.Dir.cwd().readFileAlloc(
        io,
        cfg.SKILLS_FILE,
        allocator,
        .limited(2 * 1024 * 1024),
    ) catch return allocator.dupe(u8, "");
    defer allocator.free(contents);

    const ParsedSkill = struct {
        text: []u8,
        domain: []u8,
        score: f32,
    };

    var all_skills = std.ArrayListUnmanaged(ParsedSkill).empty;
    defer {
        for (all_skills.items) |skill| {
            allocator.free(skill.text);
            allocator.free(skill.domain);
        }
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

        const skill_score: f32 = if (obj.get("score")) |value| switch (value) {
            .float => |f| @floatCast(f),
            .integer => |i| @floatFromInt(i),
            else => 0.5,
        } else 0.5;

        // Apply score filter
        if (skill_score < min_score) continue;

        const skill_domain = if (obj.get("domain")) |value| switch (value) {
            .string => |s| s,
            else => "general",
        } else "general";

        // Apply domain filter
        if (domain_filter) |df| {
            if (!std.mem.eql(u8, skill_domain, df.label())) continue;
        }

        try all_skills.append(allocator, .{
            .text = try allocator.dupe(u8, skill_text),
            .domain = try allocator.dupe(u8, skill_domain),
            .score = skill_score,
        });
    }

    if (all_skills.items.len == 0) return allocator.dupe(u8, "");

    var out = std.ArrayListUnmanaged(u8).empty;
    defer out.deinit(allocator);

    if (domain_filter) |df| {
        const header = try std.fmt.allocPrint(allocator, "\n\n[Persisted Ralph Skills ({s})]\n", .{df.label()});
        defer allocator.free(header);
        try out.appendSlice(allocator, header);
    } else {
        try out.appendSlice(allocator, "\n\n[Persisted Ralph Skills]\n");
    }

    var used: usize = out.items.len;
    // Most recent first (LIFO), high-score priority
    var idx = all_skills.items.len;
    while (idx > 0) {
        idx -= 1;
        const skill = all_skills.items[idx];
        const tag = if (!std.mem.eql(u8, skill.domain, "general"))
            skill.domain
        else
            "";
        const prefix_len: usize = if (tag.len > 0) tag.len + 5 else 2; // "- [tag] " or "- "
        const line_len = prefix_len + skill.text.len + 1;
        if (used + line_len > max_chars) break;

        if (tag.len > 0) {
            try out.appendSlice(allocator, "- [");
            try out.appendSlice(allocator, tag);
            try out.appendSlice(allocator, "] ");
        } else {
            try out.appendSlice(allocator, "- ");
        }
        try out.appendSlice(allocator, skill.text);
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

        const domain = if (obj.get("domain")) |value| switch (value) {
            .string => |s| s,
            else => "general",
        } else "general";

        const score: f32 = if (obj.get("score")) |value| switch (value) {
            .float => |f| @floatCast(f),
            .integer => |i| @floatFromInt(i),
            else => 0.5,
        } else 0.5;

        var score_buf: [16]u8 = undefined;
        const score_str = std.fmt.bufPrint(&score_buf, "{d:.1}", .{score}) catch "?";

        try out.appendSlice(allocator, "- [");
        try out.appendSlice(allocator, domain);
        try out.appendSlice(allocator, "|");
        try out.appendSlice(allocator, score_str);
        try out.appendSlice(allocator, "] ");
        try out.appendSlice(allocator, skill_text);
        try out.appendSlice(allocator, "\n");
    }

    return out.toOwnedSlice(allocator);
}
