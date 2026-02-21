//! Roadmap Integration
//!
//! Roadmap items and import functionality for project tracking.

const std = @import("std");
const types = @import("types.zig");
const persistence = @import("persistence.zig");
const catalog = @import("roadmap_catalog.zig");
const time_utils = @import("../shared/utils.zig");

pub const RoadmapItem = struct {
    title: []const u8,
    category: []const u8,
    timeline: []const u8,
    description: ?[]const u8 = null,
};

/// Import all roadmap items that don't already exist as tasks.
/// Returns the number of items imported.
pub fn importAll(
    allocator: std.mem.Allocator,
    tasks: *std.AutoHashMapUnmanaged(u64, types.Task),
    next_id: *u64,
    strings: *std.ArrayListUnmanaged([]u8),
) !usize {
    const now = time_utils.unixSeconds();
    var count: usize = 0;
    for (catalog.roadmap_entries) |roadmap_entry| {
        if (roadmap_entry.status == .done) continue;

        var exists = false;
        var it = tasks.iterator();
        while (it.next()) |task_entry| {
            if (std.mem.eql(u8, task_entry.value_ptr.title, roadmap_entry.title)) {
                exists = true;
                break;
            }
        }
        if (!exists) {
            const id = next_id.*;
            next_id.* += 1;

            const gate_text = try catalog.formatValidationGate(allocator, roadmap_entry.validation_gate);
            defer allocator.free(gate_text);

            const description = try std.fmt.allocPrint(allocator,
                \\{s}
                \\
                \\Owner: {s}
                \\Track: {s}
                \\Horizon: {s}
                \\Status: {s}
                \\Validation Gate: {s}
                \\Plan: {s}
            , .{
                roadmap_entry.summary,
                roadmap_entry.owner,
                roadmap_entry.track.label(),
                roadmap_entry.horizon.label(),
                roadmap_entry.status.label(),
                gate_text,
                roadmap_entry.plan_slug,
            });
            defer allocator.free(description);

            const owned_title = try persistence.dupeString(allocator, strings, roadmap_entry.title);
            const owned_desc = try persistence.dupeString(allocator, strings, description);

            const task = types.Task{
                .id = id,
                .title = owned_title,
                .description = owned_desc,
                .priority = switch (roadmap_entry.horizon) {
                    .now => .high,
                    .next => .normal,
                    .later => .low,
                },
                .category = .roadmap,
                .created_at = now,
                .updated_at = now,
            };

            try tasks.put(allocator, id, task);
            count += 1;
        }
    }
    return count;
}

// ============================================================================
// Tests
// ============================================================================

test "importAll imports canonical non-done entries" {
    var tasks_map = std.AutoHashMapUnmanaged(u64, types.Task){};
    defer tasks_map.deinit(std.testing.allocator);

    var strings = std.ArrayListUnmanaged([]u8).empty;
    defer {
        for (strings.items) |s| std.testing.allocator.free(s);
        strings.deinit(std.testing.allocator);
    }

    var next_id: u64 = 1;
    const count = try importAll(
        std.testing.allocator,
        &tasks_map,
        &next_id,
        &strings,
    );

    try std.testing.expectEqual(catalog.nonDoneEntryCount(), count);
    try std.testing.expectEqual(catalog.nonDoneEntryCount(), tasks_map.count());
}

test "importAll keeps dedupe-by-title behavior" {
    var tasks_map = std.AutoHashMapUnmanaged(u64, types.Task){};
    defer tasks_map.deinit(std.testing.allocator);

    var strings = std.ArrayListUnmanaged([]u8).empty;
    defer {
        for (strings.items) |s| std.testing.allocator.free(s);
        strings.deinit(std.testing.allocator);
    }

    var maybe_first_non_done: ?catalog.RoadmapEntry = null;
    for (catalog.roadmap_entries) |entry| {
        if (entry.status != .done) {
            maybe_first_non_done = entry;
            break;
        }
    }
    const first_non_done = maybe_first_non_done orelse return error.TestFailed;

    const now = time_utils.unixSeconds();
    const pre_title = try persistence.dupeString(std.testing.allocator, &strings, first_non_done.title);
    const pre_desc = try persistence.dupeString(std.testing.allocator, &strings, "pre-existing");
    try tasks_map.put(std.testing.allocator, 99, .{
        .id = 99,
        .title = pre_title,
        .description = pre_desc,
        .priority = .normal,
        .category = .roadmap,
        .created_at = now,
        .updated_at = now,
    });

    var next_id: u64 = 100;
    const imported = try importAll(
        std.testing.allocator,
        &tasks_map,
        &next_id,
        &strings,
    );

    try std.testing.expectEqual(catalog.nonDoneEntryCount() - 1, imported);
    try std.testing.expectEqual(catalog.nonDoneEntryCount(), tasks_map.count());
}
