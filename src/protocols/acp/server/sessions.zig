//! ACP Session management — groups related tasks together.

const std = @import("std");
const json_utils = @import("json_utils.zig");
const appendEscaped = json_utils.appendEscaped;

/// ACP Session — groups related tasks together
pub const Session = struct {
    id: []const u8,
    created_at: i64,
    metadata: ?[]const u8,
    task_ids: std.ArrayListUnmanaged([]const u8),

    pub fn deinit(self: *Session, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        if (self.metadata) |m| allocator.free(m);
        for (self.task_ids.items) |tid| {
            allocator.free(tid);
        }
        self.task_ids.deinit(allocator);
    }

    /// Serialize session to JSON
    pub fn toJson(self: *const Session, allocator: std.mem.Allocator) ![]u8 {
        var buf = std.ArrayListUnmanaged(u8).empty;
        errdefer buf.deinit(allocator);

        try buf.appendSlice(allocator, "{\"id\":\"");
        try appendEscaped(allocator, &buf, self.id);
        try buf.appendSlice(allocator, "\",\"created_at\":");
        var ts_buf: [32]u8 = undefined;
        const ts = std.fmt.bufPrint(&ts_buf, "{d}", .{self.created_at}) catch "0";
        try buf.appendSlice(allocator, ts);
        try buf.appendSlice(allocator, ",\"task_ids\":[");
        for (self.task_ids.items, 0..) |tid, i| {
            if (i > 0) try buf.append(allocator, ',');
            try buf.appendSlice(allocator, "\"");
            try appendEscaped(allocator, &buf, tid);
            try buf.appendSlice(allocator, "\"");
        }
        try buf.appendSlice(allocator, "]");

        if (self.metadata) |m| {
            try buf.appendSlice(allocator, ",\"metadata\":\"");
            try appendEscaped(allocator, &buf, m);
            try buf.appendSlice(allocator, "\"");
        }

        try buf.appendSlice(allocator, "}");
        return buf.toOwnedSlice(allocator);
    }
};

test "Session toJson" {
    const allocator = std.testing.allocator;
    const id = try allocator.dupe(u8, "session-1");
    const meta = try allocator.dupe(u8, "test-meta");
    var session = Session{
        .id = id,
        .created_at = 0,
        .metadata = meta,
        .task_ids = .empty,
    };
    defer session.deinit(allocator);

    const json = try session.toJson(allocator);
    defer allocator.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "session-1") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "test-meta") != null);
}

const parity_gate = @import("../../../common/parity_gate.zig");
test {
    if (!parity_gate.canRunTest()) return;
    std.testing.refAllDecls(@This());
}
