//! Developer and operator tooling.

const std = @import("std");

pub const CliHandler = struct {
    allocator: std.mem.Allocator,

    pub fn ingestDocument(self: *CliHandler, path: []const u8) !void {
        _ = self;
        std.debug.print("Ingesting document from {s}...\n", .{path});
    }

    pub fn queryMemory(self: *CliHandler, query: []const u8) !void {
        _ = self;
        std.debug.print("Querying WDBX for '{s}'...\n", .{query});
    }

    pub fn triggerCompaction(self: *CliHandler) !void {
        _ = self;
        std.debug.print("Triggering background log compaction...\n", .{});
    }

    // FIXME: implement inspectTrace(), createSnapshot()
};
