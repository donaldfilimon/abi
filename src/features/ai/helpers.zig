const std = @import("std");

pub fn countNonEmptyLines(data: []const u8) usize {
    var records: usize = 0;
    var lines = std.mem.splitScalar(u8, data, '\n');
    while (lines.next()) |line| {
        if (std.mem.trim(u8, line, " \t\r").len > 0) records += 1;
    }
    return records;
}

pub fn textEmbedding(input: []const u8) [4]f32 {
    var out = [_]f32{ 0.01, 0.01, 0.01, 0.01 };
    for (input, 0..) |byte, i| {
        const lowered = std.ascii.toLower(byte);
        out[i % out.len] += @as(f32, @floatFromInt(lowered % 31)) / 31.0;
    }
    var norm: f32 = 0;
    for (out) |v| norm += v * v;
    if (norm == 0) return out;
    const scale = @sqrt(norm);
    for (&out) |*v| v.* /= scale;
    return out;
}

pub fn responseEmbedding(query: [4]f32) [4]f32 {
    return .{ query[0] * 0.97, query[1] * 1.01, query[2] * 1.03, query[3] * 0.99 };
}

test {
    std.testing.refAllDecls(@This());
}

test "countNonEmptyLines" {
    try std.testing.expectEqual(@as(usize, 2), countNonEmptyLines("hello\nworld\n"));
    try std.testing.expectEqual(@as(usize, 0), countNonEmptyLines(""));
}

test "textEmbedding returns normalized vector" {
    const v = textEmbedding("test");
    try std.testing.expect(v.len == 4);
    var norm: f32 = 0;
    for (v) |val| norm += val * val;
    try std.testing.expect(@abs(norm - 1.0) < 0.001);
}

test "responseEmbedding" {
    const q = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const r = responseEmbedding(q);
    try std.testing.expect(r[0] < q[0]);
    try std.testing.expect(r[1] > q[1]);
}
