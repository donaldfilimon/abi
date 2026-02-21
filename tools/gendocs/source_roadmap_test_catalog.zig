const std = @import("std");

pub const Horizon = enum {
    now,
    next,
    later,

    pub fn label(self: Horizon) []const u8 {
        return switch (self) {
            .now => "Now",
            .next => "Next",
            .later => "Later",
        };
    }

    pub fn order(self: Horizon) u8 {
        return switch (self) {
            .now => 0,
            .next => 1,
            .later => 2,
        };
    }
};

pub const RoadmapStatus = enum {
    planned,
    in_progress,
    blocked,
    done,

    pub fn label(self: RoadmapStatus) []const u8 {
        return switch (self) {
            .planned => "Planned",
            .in_progress => "In Progress",
            .blocked => "Blocked",
            .done => "Done",
        };
    }

    pub fn order(self: RoadmapStatus) u8 {
        return switch (self) {
            .in_progress => 0,
            .planned => 1,
            .blocked => 2,
            .done => 3,
        };
    }
};

pub const RoadmapTrack = enum {
    docs,
    gpu,
    cli_tui,

    pub fn label(self: RoadmapTrack) []const u8 {
        return switch (self) {
            .docs => "Docs",
            .gpu => "GPU",
            .cli_tui => "CLI/TUI",
        };
    }

    pub fn order(self: RoadmapTrack) u8 {
        return switch (self) {
            .docs => 0,
            .gpu => 1,
            .cli_tui => 2,
        };
    }
};

pub const PlanSpec = struct {
    slug: []const u8,
    title: []const u8,
    status: RoadmapStatus,
    owner: []const u8,
    scope: []const u8,
    success_criteria: []const []const u8,
    gate_commands: []const []const u8,
    milestones: []const []const u8,
};

pub const RoadmapEntry = struct {
    id: []const u8,
    title: []const u8,
    summary: []const u8,
    track: RoadmapTrack,
    horizon: Horizon,
    status: RoadmapStatus,
    owner: []const u8,
    validation_gate: []const []const u8,
    plan_slug: []const u8,
};

pub const plan_specs = [_]PlanSpec{
    .{
        .slug = "docs-sync",
        .title = "Docs Sync",
        .status = .in_progress,
        .owner = "Abbey",
        .scope = "Docs scope",
        .success_criteria = &.{ "one", "two" },
        .gate_commands = &.{"zig build check-docs"},
        .milestones = &.{"m1"},
    },
    .{
        .slug = "gpu-work",
        .title = "GPU Work",
        .status = .planned,
        .owner = "Abbey",
        .scope = "GPU scope",
        .success_criteria = &.{"a"},
        .gate_commands = &.{"zig build typecheck"},
        .milestones = &.{"m2"},
    },
};

pub const roadmap_entries = [_]RoadmapEntry{
    .{
        .id = "RM-1",
        .title = "Docs now",
        .summary = "Now item",
        .track = .docs,
        .horizon = .now,
        .status = .in_progress,
        .owner = "Abbey",
        .validation_gate = &.{"zig build check-docs"},
        .plan_slug = "docs-sync",
    },
    .{
        .id = "RM-2",
        .title = "GPU next",
        .summary = "Next item",
        .track = .gpu,
        .horizon = .next,
        .status = .planned,
        .owner = "Abbey",
        .validation_gate = &.{"zig build typecheck"},
        .plan_slug = "gpu-work",
    },
    .{
        .id = "RM-3",
        .title = "CLI later",
        .summary = "Later item",
        .track = .cli_tui,
        .horizon = .later,
        .status = .blocked,
        .owner = "Abbey",
        .validation_gate = &.{"zig build cli-tests"},
        .plan_slug = "docs-sync",
    },
};

pub fn findPlanBySlug(slug: []const u8) ?*const PlanSpec {
    for (&plan_specs) |*plan| {
        if (std.mem.eql(u8, plan.slug, slug)) return plan;
    }
    return null;
}

pub fn formatValidationGate(
    allocator: std.mem.Allocator,
    commands: []const []const u8,
) ![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    for (commands, 0..) |command, idx| {
        if (idx > 0) try out.appendSlice(allocator, " ; ");
        try out.appendSlice(allocator, command);
    }
    return out.toOwnedSlice(allocator);
}
