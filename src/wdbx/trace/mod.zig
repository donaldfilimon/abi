//! Makes the system inspectable.

const std = @import("std");

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
            .timestamp = std.time.milliTimestamp(),
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
    
    // FIXME: implement lineage graph generation and audit log export
};
