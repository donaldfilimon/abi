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

/// Filter enabled entries of a given kind from a pre-loaded manifest.
/// This is a testable helper that does not depend on file I/O.
pub fn filterEnabledByKind(
    allocator: std.mem.Allocator,
    state: *const manifest.Manifest,
    kind: manifest.PluginKind,
    id: ?[]const u8,
) !?manifest.PluginEntry {
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

// ============================================================================
// Tests
// ============================================================================

test "filterEnabledByKind with empty manifest returns null" {
    const allocator = std.testing.allocator;
    var m = manifest.Manifest.init(allocator);
    defer m.deinit();

    const result = try filterEnabledByKind(allocator, &m, .http, null);
    try std.testing.expect(result == null);
}

test "filterEnabledByKind finds first enabled http plugin" {
    const allocator = std.testing.allocator;
    var m = manifest.Manifest.init(allocator);
    defer m.deinit();

    try m.addOrUpdateNative("native1", "/lib/a.dylib", null);
    try m.addOrUpdateHttp("http1", "http://host1", null, null);
    try m.addOrUpdateHttp("http2", "http://host2", null, null);

    var result = (try filterEnabledByKind(allocator, &m, .http, null)).?;
    defer result.deinit(allocator);

    try std.testing.expectEqualStrings("http1", result.id);
    try std.testing.expectEqual(manifest.PluginKind.http, result.kind);
}

test "filterEnabledByKind skips disabled entries" {
    const allocator = std.testing.allocator;
    var m = manifest.Manifest.init(allocator);
    defer m.deinit();

    try m.addOrUpdateHttp("disabled-http", "http://host1", null, null);
    _ = m.setEnabled("disabled-http", false);
    try m.addOrUpdateHttp("enabled-http", "http://host2", null, null);

    var result = (try filterEnabledByKind(allocator, &m, .http, null)).?;
    defer result.deinit(allocator);

    try std.testing.expectEqualStrings("enabled-http", result.id);
}

test "filterEnabledByKind filters by id" {
    const allocator = std.testing.allocator;
    var m = manifest.Manifest.init(allocator);
    defer m.deinit();

    try m.addOrUpdateHttp("http1", "http://host1", null, null);
    try m.addOrUpdateHttp("http2", "http://host2", null, null);

    var result = (try filterEnabledByKind(allocator, &m, .http, "http2")).?;
    defer result.deinit(allocator);
    try std.testing.expectEqualStrings("http2", result.id);

    // Wrong kind returns null
    const no_result = try filterEnabledByKind(allocator, &m, .native, "http1");
    try std.testing.expect(no_result == null);

    // Non-existent id returns null
    const missing = try filterEnabledByKind(allocator, &m, .http, "nonexistent");
    try std.testing.expect(missing == null);
}

test "filterEnabledByKind returns null when all disabled" {
    const allocator = std.testing.allocator;
    var m = manifest.Manifest.init(allocator);
    defer m.deinit();

    try m.addOrUpdateHttp("h1", "http://host1", null, null);
    _ = m.setEnabled("h1", false);

    const result = try filterEnabledByKind(allocator, &m, .http, null);
    try std.testing.expect(result == null);
}

test {
    std.testing.refAllDecls(@This());
}
