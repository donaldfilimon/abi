const std = @import("std");
const builtin = @import("builtin");
const model = @import("model.zig");
const catalog = if (builtin.is_test)
    @import("source_roadmap_test_catalog.zig")
else
    @import("roadmap_catalog");

pub const RoadmapData = struct {
    roadmap_entries: []model.RoadmapDocEntry,
    plan_entries: []model.PlanDocEntry,

    pub fn deinit(self: RoadmapData, allocator: std.mem.Allocator) void {
        model.deinitRoadmapSlice(allocator, self.roadmap_entries);
        model.deinitPlanSlice(allocator, self.plan_entries);
    }
};

pub fn discover(allocator: std.mem.Allocator) !RoadmapData {
    const plan_entries = try discoverPlanEntries(allocator);
    errdefer model.deinitPlanSlice(allocator, plan_entries);

    const roadmap_entries = try discoverRoadmapEntries(allocator);
    errdefer model.deinitRoadmapSlice(allocator, roadmap_entries);

    return .{
        .roadmap_entries = roadmap_entries,
        .plan_entries = plan_entries,
    };
}

fn discoverPlanEntries(allocator: std.mem.Allocator) ![]model.PlanDocEntry {
    var plans = std.ArrayListUnmanaged(model.PlanDocEntry).empty;
    errdefer {
        for (plans.items) |item| item.deinit(allocator);
        plans.deinit(allocator);
    }

    for (catalog.plan_specs) |plan| {
        try plans.append(allocator, .{
            .slug = try allocator.dupe(u8, plan.slug),
            .title = try allocator.dupe(u8, plan.title),
            .status = try allocator.dupe(u8, plan.status.label()),
            .status_order = plan.status.order(),
            .owner = try allocator.dupe(u8, plan.owner),
            .scope = try allocator.dupe(u8, plan.scope),
            .success_criteria = try dupeStringSlice(allocator, plan.success_criteria),
            .gate_commands = try dupeStringSlice(allocator, plan.gate_commands),
            .milestones = try dupeStringSlice(allocator, plan.milestones),
        });
    }

    insertionSortPlans(plans.items);
    return plans.toOwnedSlice(allocator);
}

fn discoverRoadmapEntries(allocator: std.mem.Allocator) ![]model.RoadmapDocEntry {
    var entries = std.ArrayListUnmanaged(model.RoadmapDocEntry).empty;
    errdefer {
        for (entries.items) |entry| entry.deinit(allocator);
        entries.deinit(allocator);
    }

    for (catalog.roadmap_entries) |entry| {
        const gate = try catalog.formatValidationGate(allocator, entry.validation_gate);
        defer allocator.free(gate);

        const plan_title = if (catalog.findPlanBySlug(entry.plan_slug)) |plan| plan.title else entry.plan_slug;

        try entries.append(allocator, .{
            .id = try allocator.dupe(u8, entry.id),
            .title = try allocator.dupe(u8, entry.title),
            .summary = try allocator.dupe(u8, entry.summary),
            .track = try allocator.dupe(u8, entry.track.label()),
            .track_order = entry.track.order(),
            .horizon = try allocator.dupe(u8, entry.horizon.label()),
            .horizon_order = entry.horizon.order(),
            .status = try allocator.dupe(u8, entry.status.label()),
            .status_order = entry.status.order(),
            .owner = try allocator.dupe(u8, entry.owner),
            .validation_gate = try allocator.dupe(u8, gate),
            .plan_slug = try allocator.dupe(u8, entry.plan_slug),
            .plan_title = try allocator.dupe(u8, plan_title),
        });
    }

    insertionSortRoadmap(entries.items);
    return entries.toOwnedSlice(allocator);
}

fn dupeStringSlice(
    allocator: std.mem.Allocator,
    source: []const []const u8,
) ![]const []const u8 {
    const out = try allocator.alloc([]const u8, source.len);
    var copied: usize = 0;
    errdefer {
        var i: usize = 0;
        while (i < copied) : (i += 1) {
            allocator.free(out[i]);
        }
        allocator.free(out);
    }

    for (source, 0..) |item, idx| {
        out[idx] = try allocator.dupe(u8, item);
        copied = idx + 1;
    }
    return out;
}

fn insertionSortRoadmap(items: []model.RoadmapDocEntry) void {
    var i: usize = 1;
    while (i < items.len) : (i += 1) {
        const value = items[i];
        var j = i;
        while (j > 0 and model.compareRoadmapEntries({}, value, items[j - 1])) : (j -= 1) {
            items[j] = items[j - 1];
        }
        items[j] = value;
    }
}

fn insertionSortPlans(items: []model.PlanDocEntry) void {
    var i: usize = 1;
    while (i < items.len) : (i += 1) {
        const value = items[i];
        var j = i;
        while (j > 0 and model.comparePlanEntries({}, value, items[j - 1])) : (j -= 1) {
            items[j] = items[j - 1];
        }
        items[j] = value;
    }
}

test "discover returns deterministic sorted roadmap and plans" {
    const data = try discover(std.testing.allocator);
    defer data.deinit(std.testing.allocator);

    try std.testing.expect(data.roadmap_entries.len > 0);
    try std.testing.expect(data.plan_entries.len > 0);

    var i: usize = 1;
    while (i < data.roadmap_entries.len) : (i += 1) {
        try std.testing.expect(!model.compareRoadmapEntries({}, data.roadmap_entries[i], data.roadmap_entries[i - 1]));
    }

    i = 1;
    while (i < data.plan_entries.len) : (i += 1) {
        try std.testing.expect(!model.comparePlanEntries({}, data.plan_entries[i], data.plan_entries[i - 1]));
    }
}

test "discover includes now next later horizons" {
    const data = try discover(std.testing.allocator);
    defer data.deinit(std.testing.allocator);

    var saw_now = false;
    var saw_next = false;
    var saw_later = false;

    for (data.roadmap_entries) |entry| {
        if (std.mem.eql(u8, entry.horizon, "Now")) saw_now = true;
        if (std.mem.eql(u8, entry.horizon, "Next")) saw_next = true;
        if (std.mem.eql(u8, entry.horizon, "Later")) saw_later = true;
    }

    try std.testing.expect(saw_now);
    try std.testing.expect(saw_next);
    try std.testing.expect(saw_later);
}
