const std = @import("std");
const json = std.json;

pub const PatternType = enum {
    literal,
    glob,
    regex,
    fuzzy,
    compound,
};

pub const SearchPattern = struct {
    raw: []const u8,
    pattern_type: PatternType,
    literal: ?[]const u8 = null,
    lowercase_literal: ?[]const u8 = null,
    glob_pattern: ?[]const u8 = null,
    regex_pattern: ?[]const u8 = null,
    fuzzy_chars: ?[]const u8 = null,
    case_sensitive: bool = false,
    inverted: bool = false,

    pub fn deinit(self: *SearchPattern, allocator: std.mem.Allocator) void {
        if (self.literal) |lit| allocator.free(lit);
        if (self.lowercase_literal) |lit| allocator.free(lit);
        if (self.glob_pattern) |pat| allocator.free(pat);
        if (self.regex_pattern) |pat| allocator.free(pat);
        if (self.fuzzy_chars) |chars| allocator.free(chars);
    }
};

pub const MatchPosition = struct {
    start: usize,
    end: usize,
};

pub const SearchMatch = struct {
    text: []const u8,
    positions: std.ArrayListUnmanaged(MatchPosition),
    score: f32,

    pub fn deinit(self: *SearchMatch, allocator: std.mem.Allocator) void {
        self.positions.deinit(allocator);
    }
};

pub const PatternCompiler = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) PatternCompiler {
        return PatternCompiler{ .allocator = allocator };
    }

    pub fn compile(self: *PatternCompiler, pattern: []const u8, pattern_type: PatternType, case_sensitive: bool) !SearchPattern {
        return switch (pattern_type) {
            .literal => self.compileLiteral(pattern, case_sensitive),
            .glob => self.compileGlob(pattern, case_sensitive),
            .regex => self.compileRegex(pattern, case_sensitive),
            .fuzzy => self.compileFuzzy(pattern, case_sensitive),
            .compound => self.compileCompound(pattern, case_sensitive),
        };
    }

    fn compileLiteral(self: *PatternCompiler, pattern: []const u8, case_sensitive: bool) !SearchPattern {
        const lit = try self.allocator.dupe(u8, pattern);
        errdefer self.allocator.free(lit);

        // Pre-compute lowercase version for case-insensitive searches
        const lowercase_lit = if (!case_sensitive) blk: {
            const lower = try self.allocator.alloc(u8, pattern.len);
            _ = std.ascii.lowerString(lower, pattern);
            break :blk lower;
        } else null;

        return SearchPattern{
            .raw = pattern,
            .pattern_type = .literal,
            .literal = lit,
            .lowercase_literal = lowercase_lit,
            .case_sensitive = case_sensitive,
        };
    }

    fn compileGlob(self: *PatternCompiler, pattern: []const u8, case_sensitive: bool) !SearchPattern {
        const glob_pat = try self.allocator.dupe(u8, pattern);
        return SearchPattern{
            .raw = pattern,
            .pattern_type = .glob,
            .glob_pattern = glob_pat,
            .case_sensitive = case_sensitive,
        };
    }

    fn compileRegex(self: *PatternCompiler, pattern: []const u8, case_sensitive: bool) !SearchPattern {
        const regex_pat = try self.allocator.dupe(u8, pattern);
        return SearchPattern{
            .raw = pattern,
            .pattern_type = .regex,
            .regex_pattern = regex_pat,
            .case_sensitive = case_sensitive,
        };
    }

    fn compileFuzzy(self: *PatternCompiler, pattern: []const u8, case_sensitive: bool) !SearchPattern {
        const chars = try self.allocator.dupe(u8, pattern);
        return SearchPattern{
            .raw = pattern,
            .pattern_type = .fuzzy,
            .fuzzy_chars = chars,
            .case_sensitive = case_sensitive,
        };
    }

    fn compileCompound(_: *PatternCompiler, pattern: []const u8, case_sensitive: bool) !SearchPattern {
        return SearchPattern{
            .raw = pattern,
            .pattern_type = .compound,
            .case_sensitive = case_sensitive,
        };
    }
};

pub fn matchLiteral(pattern: SearchPattern, text: []const u8) bool {
    const search_lit = if (pattern.case_sensitive)
        pattern.literal.?
    else
        pattern.lowercase_literal.?;

    // For case-insensitive search, use stack buffer for temporary lowercase text
    var text_buf: [4096]u8 = undefined;
    const search_text = if (pattern.case_sensitive)
        text
    else blk: {
        if (text.len > text_buf.len) return false; // Skip extremely large strings
        _ = std.ascii.lowerString(&text_buf, text);
        break :blk text_buf[0..text.len];
    };

    return std.mem.indexOf(u8, search_text, search_lit) != null;
}

pub fn matchGlob(pattern: SearchPattern, text: []const u8) bool {
    const glob_pat = pattern.glob_pattern orelse return false;

    return matchesGlob(glob_pat, text);
}

pub fn matchesGlob(pattern: []const u8, name: []const u8) bool {
    if (pattern.len == 0) return name.len == 0;

    var p_idx: usize = 0;
    var n_idx: usize = 0;
    var star_idx: ?usize = null;
    var match_idx: usize = 0;

    while (n_idx < name.len) : (n_idx += 1) {
        if (p_idx < pattern.len) {
            const pc = pattern[p_idx];
            const nc = name[n_idx];

            if (pc == '*') {
                star_idx = p_idx;
                match_idx = n_idx;
                p_idx += 1;
            } else if (pc == '?' or pc == nc) {
                p_idx += 1;
            } else if (star_idx != null) {
                p_idx = star_idx.? + 1;
                match_idx += 1;
                n_idx = match_idx;
            } else {
                return false;
            }
        }
    }

    while (p_idx < pattern.len and pattern[p_idx] == '*') {
        p_idx += 1;
    }

    return p_idx == pattern.len;
}

pub fn matchRegex(pattern: SearchPattern, text: []const u8) bool {
    const regex_pat = pattern.regex_pattern orelse return false;

    const result = std.regex.compile(pattern.allocator, regex_pat) catch {
        return false;
    };
    defer result.deinit();

    return result.match(text) != null;
}

pub fn matchFuzzy(pattern: SearchPattern, text: []const u8) f32 {
    const fuzzy_chars = pattern.fuzzy_chars orelse return 0.0;

    if (fuzzy_chars.len == 0) return 1.0;

    var score: f32 = 0.0;
    var bonus: f32 = 0.0;
    var text_idx: usize = 0;

    for (fuzzy_chars, 0..) |fc, ci| {
        const search_char = if (pattern.case_sensitive) fc else std.ascii.toUpper(fc);

        while (text_idx < text.len) {
            const tc = text[text_idx];
            const search_tc = if (pattern.case_sensitive) tc else std.ascii.toUpper(tc);

            if (search_char == search_tc) {
                score += 1.0;
                if (ci == text_idx) {
                    bonus += 0.5;
                }
                text_idx += 1;
                break;
            }
            text_idx += 1;
        }
    }

    if (fuzzy_chars.len == text_idx) {
        return 0.0;
    }

    return (score + bonus) / @as(f32, @floatFromInt(fuzzy_chars.len + 1));
}

pub fn match(pattern: SearchPattern, text: []const u8) bool {
    const result = switch (pattern.pattern_type) {
        .literal => matchLiteral(pattern, text),
        .glob => matchGlob(pattern, text),
        .regex => matchRegex(pattern, text),
        .fuzzy => matchFuzzy(pattern, text) > 0.5,
        .compound => false,
    };

    return if (pattern.inverted) !result else result;
}

pub fn findAllLiteral(pattern: SearchPattern, text: []const u8, allocator: std.mem.Allocator) !std.ArrayListUnmanaged(MatchPosition) {
    var matches = std.ArrayListUnmanaged(MatchPosition){};
    var start: usize = 0;

    const search_text = if (pattern.case_sensitive) text else blk: {
        const lowered = try allocator.dupe(u8, text);
        defer allocator.free(lowered);
        break :blk std.ascii.lowerString(lowered, lowered.len);
    };

    const search_pat = if (pattern.case_sensitive and pattern.literal != null) pattern.literal.? else blk: {
        const pat = pattern.literal orelse return matches;
        break :blk if (pattern.case_sensitive) pat else blk2: {
            const lowered = try allocator.dupe(u8, pat);
            defer allocator.free(lowered);
            break :blk2 std.ascii.lowerString(lowered, lowered.len);
        };
    };

    while (std.mem.indexOfPos(u8, search_text, start, search_pat)) |idx| {
        try matches.append(allocator, .{ .start = idx, .end = idx + search_pat.len });
        start = idx + 1;
    }

    return matches;
}

pub fn splitIntoTokens(text: []const u8, allocator: std.mem.Allocator) !std.ArrayListUnmanaged([]const u8) {
    var tokens = std.ArrayListUnmanaged([]const u8){};
    var start: usize = 0;
    var in_token = false;

    for (text, 0..) |c, i| {
        if (std.ascii.isAlphanumeric(c) or c == '_') {
            if (!in_token) {
                start = i;
                in_token = true;
            }
        } else {
            if (in_token) {
                const token = try allocator.dupe(u8, text[start..i]);
                try tokens.append(allocator, token);
                in_token = false;
            }
        }
    }

    if (in_token) {
        const token = try allocator.dupe(u8, text[start..]);
        try tokens.append(allocator, token);
    }

    return tokens;
}

pub fn calculateRelevanceScore(matches: []const MatchPosition, total_length: usize, pattern_length: usize) f32 {
    if (matches.len == 0) return 0.0;

    var score: f32 = 1.0;

    const coverage = @as(f32, @floatFromInt(matches[0].end - matches[0].start)) / @as(f32, @floatFromInt(pattern_length));
    score += coverage * 2.0;

    if (matches[0].start < 50) {
        score += 0.5;
    }

    const density = @as(f32, @floatFromInt(matches.len)) / @as(f32, @floatFromInt(total_length / 100));
    score += density * 0.5;

    return @min(score, 1.0);
}
