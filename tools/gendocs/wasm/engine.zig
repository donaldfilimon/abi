const std = @import("std");

pub fn score(query: []const u8, target: []const u8) i32 {
    if (query.len == 0) return 1;
    if (target.len == 0) return 0;

    if (std.mem.eql(u8, query, target)) return 1400;
    if (std.mem.startsWith(u8, target, query)) {
        return @as(i32, @intCast(1000 - @min(target.len - query.len, 800)));
    }

    if (std.mem.indexOf(u8, target, query)) |idx| {
        return @as(i32, @intCast(700 - @min(idx, 650)));
    }

    // Fuzzy subsequence scoring
    var qi: usize = 0;
    var consecutive: i32 = 0;
    var best_consecutive: i32 = 0;
    var hits: i32 = 0;
    var penalty: i32 = 0;

    for (target) |tc| {
        if (qi < query.len and tc == query[qi]) {
            qi += 1;
            hits += 1;
            consecutive += 1;
            if (consecutive > best_consecutive) best_consecutive = consecutive;
        } else {
            if (consecutive == 0) penalty += 1;
            consecutive = 0;
        }
    }

    if (hits == 0) return 0;
    return hits * 70 + best_consecutive * 55 - penalty;
}

pub fn findMatchStart(query: []const u8, target: []const u8) i32 {
    if (query.len == 0) return 0;
    return @intCast(std.mem.indexOf(u8, target, query) orelse return -1);
}

test "score exact and prefix ordering" {
    const exact = score("gpu", "gpu");
    const prefix = score("gpu", "gpu-backend");
    const partial = score("gpu", "runtime-gpu");
    try std.testing.expect(exact > prefix);
    try std.testing.expect(prefix > partial);
}

test "find match start" {
    try std.testing.expectEqual(@as(i32, 4), findMatchStart("net", "api-network"));
    try std.testing.expectEqual(@as(i32, -1), findMatchStart("abc", "xyz"));
}
