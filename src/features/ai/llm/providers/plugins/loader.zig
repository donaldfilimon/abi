const std = @import("std");
const manifest = @import("manifest.zig");

pub fn loadManifest(allocator: std.mem.Allocator) !manifest.Manifest {
    return manifest.loadDefault(allocator);
}

pub fn findEnabledByKind(
    allocator: std.mem.Allocator,
    kind: manifest.PluginKind,
    id: ?[]const u8,
) !?manifest.PluginEntry {
    var state = try manifest.loadDefault(allocator);
    defer state.deinit();

    if (id) |needle| {
        const entry = state.find(needle) orelse return null;
        if (!entry.enabled or entry.kind != kind) return null;
        return try manifest.cloneEntry(allocator, entry.*);
    }

    for (state.entries.items) |entry| {
        if (entry.enabled and entry.kind == kind) {
            return try manifest.cloneEntry(allocator, entry);
        }
    }

    return null;
}

pub fn hasAnyEnabled(kind: manifest.PluginKind) bool {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var state = manifest.loadDefault(allocator) catch return false;
    defer state.deinit();

    for (state.entries.items) |entry| {
        if (entry.enabled and entry.kind == kind) return true;
    }

    return false;
}
