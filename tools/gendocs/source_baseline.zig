const std = @import("std");
const model = @import("model.zig");

pub fn loadBuildMeta(allocator: std.mem.Allocator, io: std.Io, cwd: std.Io.Dir) !model.BuildMeta {
    const path = "tools/scripts/baseline.zig";
    const source = try cwd.readFileAlloc(io, path, allocator, .limited(256 * 1024));
    defer allocator.free(source);

    return .{
        .zig_version = try parseStringConst(allocator, source, "zig_version"),
        .test_main_pass = try parseUsizeConst(source, "test_main_pass"),
        .test_main_skip = try parseUsizeConst(source, "test_main_skip"),
        .test_main_total = try parseUsizeConst(source, "test_main_total"),
        .test_feature_pass = try parseUsizeConst(source, "test_feature_pass"),
        .test_feature_total = try parseUsizeConst(source, "test_feature_total"),
    };
}

fn parseStringConst(allocator: std.mem.Allocator, source: []const u8, name: []const u8) ![]const u8 {
    const needle = try std.fmt.allocPrint(allocator, "pub const {s} = \"", .{name});
    defer allocator.free(needle);

    const start = std.mem.indexOf(u8, source, needle) orelse return error.MissingBaselineField;
    const tail = source[start + needle.len ..];
    const end = std.mem.indexOfScalar(u8, tail, '"') orelse return error.InvalidBaselineField;
    return allocator.dupe(u8, tail[0..end]);
}

fn parseUsizeConst(source: []const u8, name: []const u8) !usize {
    var buf: [128]u8 = undefined;
    const needle = try std.fmt.bufPrint(&buf, "pub const {s}: usize = ", .{name});

    const start = std.mem.indexOf(u8, source, needle) orelse return error.MissingBaselineField;
    const tail = source[start + needle.len ..];
    const end = std.mem.indexOfScalar(u8, tail, ';') orelse return error.InvalidBaselineField;
    const value_text = std.mem.trim(u8, tail[0..end], " \t\r\n");
    return std.fmt.parseInt(usize, value_text, 10);
}

test "baseline parser reads constants" {
    const sample =
        \\pub const zig_version = "0.16.0-dev.9999+abcd";
        \\pub const test_main_pass: usize = 1;
        \\pub const test_main_skip: usize = 2;
        \\pub const test_main_total: usize = 3;
        \\pub const test_feature_pass: usize = 4;
        \\pub const test_feature_total: usize = 5;
    ;

    const zig_version = try parseStringConst(std.testing.allocator, sample, "zig_version");
    defer std.testing.allocator.free(zig_version);

    try std.testing.expectEqualStrings("0.16.0-dev.9999+abcd", zig_version);
    try std.testing.expectEqual(@as(usize, 5), try parseUsizeConst(sample, "test_feature_total"));
}
