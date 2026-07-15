const std = @import("std");
const sanitize = @import("sanitize.zig");

const sanitizeControlBytes = sanitize.sanitizeControlBytes;

pub const DIAG_WIDTH: usize = 68;
pub const LABEL_WIDTH: usize = 25;
pub const VALUE_WIDTH: usize = 40;

pub fn utf8PrefixLen(input: []const u8, max_len: usize) usize {
    var end = @min(input.len, max_len);
    while (end > 0 and !std.unicode.utf8ValidateSlice(input[0..end])) : (end -= 1) {}
    return end;
}

pub fn appendRepeated(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, byte: u8, count: usize) !void {
    var i: usize = 0;
    while (i < count) : (i += 1) try out.append(allocator, byte);
}

pub fn appendRule(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, count: usize) !void {
    var i: usize = 0;
    while (i < count) : (i += 1) try out.appendSlice(allocator, "─");
}

pub fn appendFitted(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, raw: []const u8, width: usize) !void {
    const safe = try sanitizeControlBytes(allocator, raw);
    defer allocator.free(safe);

    if (safe.len <= width) {
        try out.appendSlice(allocator, safe);
        try appendRepeated(out, allocator, ' ', width - safe.len);
        return;
    }

    if (width == 0) return;
    if (width == 1) {
        try out.append(allocator, '~');
        return;
    }

    const end = utf8PrefixLen(safe, width - 1);
    try out.appendSlice(allocator, safe[0..end]);
    try appendRepeated(out, allocator, ' ', (width - 1) - end);
    try out.append(allocator, '~');
}

pub fn appendBorder(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, left: []const u8, title: []const u8, right: []const u8) !void {
    try out.appendSlice(allocator, left);
    if (title.len > 0) {
        try out.appendSlice(allocator, " ");
        try appendFitted(out, allocator, title, @min(title.len, DIAG_WIDTH - 4));
        try out.appendSlice(allocator, " ");
        const used = @min(title.len, DIAG_WIDTH - 4) + 2;
        if (used < DIAG_WIDTH) try appendRule(out, allocator, DIAG_WIDTH - used);
    } else {
        try appendRule(out, allocator, DIAG_WIDTH);
    }
    try out.appendSlice(allocator, right);
    try out.append(allocator, '\n');
}

pub fn appendRow(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, label: []const u8, value: []const u8) !void {
    try out.appendSlice(allocator, "│ ");
    try appendFitted(out, allocator, label, LABEL_WIDTH);
    try out.appendSlice(allocator, " ");
    try appendFitted(out, allocator, value, VALUE_WIDTH);
    try out.appendSlice(allocator, " │\n");
}

pub fn appendMetricRow(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, label: []const u8, value: usize) !void {
    var buf: [32]u8 = undefined;
    const rendered = try std.fmt.bufPrint(&buf, "{d}", .{value});
    try appendRow(out, allocator, label, rendered);
}

pub fn appendPanelHeader(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, title: []const u8) !void {
    try appendBorder(out, allocator, "┌", title, "┐");
}

pub fn appendPanelFooter(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator) !void {
    try appendBorder(out, allocator, "└", "", "┘");
}

test {
    std.testing.refAllDecls(@This());
}
