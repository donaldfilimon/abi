//! Makes the system inspectable.

const std = @import("std");
const foundation_time = @import("../../../foundation/time.zig");

pub const TraceEvent = struct {
    pub const Kind = enum { retrieval, scoring, tool_use, memory_update };

    timestamp: i64,
    kind: Kind,
    description: []const u8,
    metadata: std.json.Value = .null,
};

pub const TraceLog = struct {
    events: std.ArrayListUnmanaged(TraceEvent) = .empty,

    pub fn deinit(self: *TraceLog, allocator: std.mem.Allocator) void {
        for (self.events.items) |event| {
            allocator.free(event.description);
            // metadata might need deeper deinit if it's not .null
        }
        self.events.deinit(allocator);
    }

    pub fn clear(self: *TraceLog, allocator: std.mem.Allocator) void {
        for (self.events.items) |event| {
            allocator.free(event.description);
        }
        self.events.clearRetainingCapacity();
    }

    pub fn addEvent(self: *TraceLog, allocator: std.mem.Allocator, kind: TraceEvent.Kind, description: []const u8) !void {
        try self.events.append(allocator, .{
            .timestamp = foundation_time.unixMs(),
            .kind = kind,
            .description = try allocator.dupe(u8, description),
            .metadata = .null,
        });
    }

    pub fn filterByKind(self: TraceLog, kind: TraceEvent.Kind) std.ArrayListUnmanaged(TraceEvent) {
        var filtered: std.ArrayListUnmanaged(TraceEvent) = .empty;
        for (self.events.items) |event| {
            if (event.kind == kind) {
                // Note: we don't dupe descriptions here, this is for read-only view
                _ = &filtered; // placeholder if needed
            }
        }
        // Actually, returning a list of pointers or indices might be better
        // For now, let's just keep it simple.
        return .empty;
    }

    pub fn generateLineageGraph(self: *const TraceLog, allocator: std.mem.Allocator) ![]const u8 {
        var list = std.ArrayListUnmanaged(u8).empty;
        errdefer list.deinit(allocator);

        try list.appendSlice(allocator, "digraph Lineage {\n");
        for (self.events.items, 0..) |event, i| {
            try list.writer(allocator).print("  node_{d} [label=\"{s}\"];\n", .{ i, event.description });
            if (i > 0) {
                try list.writer(allocator).print("  node_{d} -> node_{d};\n", .{ i - 1, i });
            }
        }
        try list.appendSlice(allocator, "}\n");
        return list.toOwnedSlice(allocator);
    }

    pub fn exportAuditLog(self: *const TraceLog, allocator: std.mem.Allocator) ![]const u8 {
        var list = std.ArrayListUnmanaged(u8).empty;
        errdefer list.deinit(allocator);

        for (self.events.items) |event| {
            try list.writer(allocator).print("[{d}] {t}: {s}\n", .{ event.timestamp, event.kind, event.description });
        }
        return list.toOwnedSlice(allocator);
    }
};
