const std = @import("std");
const model = @import("model.zig");

const generated_footer =
    \\
    \\
    \\---
    \\
    \\*Generated automatically by `zig build gendocs`*
    \\
    \\
    \\## Zig Skill
    \\Use the `$zig` Codex skill for ABI Zig 0.16-dev syntax updates, modular build graph guidance, and targeted validation workflows.
    \\
;

pub fn render(
    allocator: std.mem.Allocator,
    io: std.Io,
    cwd: std.Io.Dir,
    plans: []const model.PlanDocEntry,
    roadmap_entries: []const model.RoadmapDocEntry,
    outputs: *std.ArrayListUnmanaged(model.OutputFile),
) !void {
    const index_template = loadTemplate(
        allocator,
        io,
        cwd,
        "tools/gendocs/templates/plans/index.md.tpl",
        default_index_template,
    );
    defer allocator.free(index_template);

    const plan_template = loadTemplate(
        allocator,
        io,
        cwd,
        "tools/gendocs/templates/plans/plan.md.tpl",
        default_plan_template,
    );
    defer allocator.free(plan_template);

    var active_count: usize = 0;
    for (plans) |plan| {
        if (!std.mem.eql(u8, plan.status, "Done")) active_count += 1;
    }

    const plans_table = try buildPlansTable(allocator, plans);
    defer allocator.free(plans_table);

    const horizon_counts = try buildHorizonCounts(allocator, roadmap_entries);
    defer allocator.free(horizon_counts);

    const index_summary = try std.fmt.allocPrint(
        allocator,
        "Active generated plans: **{d}**. This index is generated from the canonical roadmap catalog and kept in sync with task import metadata.",
        .{active_count},
    );
    defer allocator.free(index_summary);

    const index_body = try applyTemplate(allocator, index_template, .{
        .title = "Plans Index",
        .status = "",
        .owner = "",
        .scope = "",
        .summary = index_summary,
        .success_criteria = "",
        .gate_commands = "",
        .milestones = "",
        .related_roadmap = horizon_counts,
        .plans_table = plans_table,
    });
    defer allocator.free(index_body);

    var index_page = std.ArrayListUnmanaged(u8).empty;
    defer index_page.deinit(allocator);
    try appendFmt(allocator, &index_page,
        \\---
        \\title: Plans
        \\description: Generated active execution plans
        \\---
        \\
        \\# Plans
        \\
    , .{});
    try index_page.appendSlice(allocator, index_body);
    try index_page.appendSlice(allocator, generated_footer);
    try model.pushOutput(allocator, outputs, "docs/plans/index.md", index_page.items);

    for (plans) |plan| {
        if (std.mem.eql(u8, plan.status, "Done")) continue;

        const success = try formatBullets(allocator, plan.success_criteria);
        defer allocator.free(success);
        const gates = try formatBullets(allocator, plan.gate_commands);
        defer allocator.free(gates);
        const milestones = try formatBullets(allocator, plan.milestones);
        defer allocator.free(milestones);
        const related = try relatedRoadmapForPlan(allocator, plan.slug, roadmap_entries);
        defer allocator.free(related);

        const plan_body = try applyTemplate(allocator, plan_template, .{
            .title = plan.title,
            .status = plan.status,
            .owner = plan.owner,
            .scope = plan.scope,
            .summary = "",
            .success_criteria = success,
            .gate_commands = gates,
            .milestones = milestones,
            .related_roadmap = related,
            .plans_table = "",
        });
        defer allocator.free(plan_body);

        var page = std.ArrayListUnmanaged(u8).empty;
        defer page.deinit(allocator);
        try appendFmt(allocator, &page,
            \\---
            \\title: {s}
            \\description: Generated implementation plan
            \\---
            \\
            \\# {s}
            \\
        , .{ plan.title, plan.title });
        try page.appendSlice(allocator, plan_body);
        try page.appendSlice(allocator, generated_footer);

        const out_path = try std.fmt.allocPrint(allocator, "docs/plans/{s}.md", .{plan.slug});
        defer allocator.free(out_path);
        try model.pushOutput(allocator, outputs, out_path, page.items);
    }
}

const TemplateArgs = struct {
    title: []const u8,
    status: []const u8,
    owner: []const u8,
    scope: []const u8,
    summary: []const u8,
    success_criteria: []const u8,
    gate_commands: []const u8,
    milestones: []const u8,
    related_roadmap: []const u8,
    plans_table: []const u8,
};

fn loadTemplate(
    allocator: std.mem.Allocator,
    io: std.Io,
    cwd: std.Io.Dir,
    path: []const u8,
    fallback: []const u8,
) []u8 {
    return cwd.readFileAlloc(io, path, allocator, .limited(512 * 1024)) catch allocator.dupe(u8, fallback) catch @panic("OOM");
}

fn applyTemplate(allocator: std.mem.Allocator, template: []const u8, args: TemplateArgs) ![]u8 {
    const r1 = try replaceAll(allocator, template, "{{TITLE}}", args.title);
    defer allocator.free(r1);
    const r2 = try replaceAll(allocator, r1, "{{STATUS}}", args.status);
    defer allocator.free(r2);
    const r3 = try replaceAll(allocator, r2, "{{OWNER}}", args.owner);
    defer allocator.free(r3);
    const r4 = try replaceAll(allocator, r3, "{{SCOPE}}", args.scope);
    defer allocator.free(r4);
    const r5 = try replaceAll(allocator, r4, "{{INDEX_SUMMARY}}", args.summary);
    defer allocator.free(r5);
    const r6 = try replaceAll(allocator, r5, "{{SUCCESS_CRITERIA}}", args.success_criteria);
    defer allocator.free(r6);
    const r7 = try replaceAll(allocator, r6, "{{GATE_COMMANDS}}", args.gate_commands);
    defer allocator.free(r7);
    const r8 = try replaceAll(allocator, r7, "{{MILESTONES}}", args.milestones);
    defer allocator.free(r8);
    const r9 = try replaceAll(allocator, r8, "{{RELATED_ROADMAP}}", args.related_roadmap);
    defer allocator.free(r9);
    const r10 = try replaceAll(allocator, r9, "{{PLANS_TABLE}}", args.plans_table);
    return r10;
}

fn replaceAll(allocator: std.mem.Allocator, input: []const u8, needle: []const u8, repl: []const u8) ![]u8 {
    if (needle.len == 0) return allocator.dupe(u8, input);

    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    var cursor: usize = 0;
    while (true) {
        const pos = std.mem.indexOfPos(u8, input, cursor, needle) orelse {
            try out.appendSlice(allocator, input[cursor..]);
            break;
        };
        try out.appendSlice(allocator, input[cursor..pos]);
        try out.appendSlice(allocator, repl);
        cursor = pos + needle.len;
    }

    return out.toOwnedSlice(allocator);
}

fn formatBullets(
    allocator: std.mem.Allocator,
    lines: []const []const u8,
) ![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    for (lines) |line| {
        try out.appendSlice(allocator, "- ");
        try out.appendSlice(allocator, line);
        try out.append(allocator, '\n');
    }
    return out.toOwnedSlice(allocator);
}

fn buildPlansTable(allocator: std.mem.Allocator, plans: []const model.PlanDocEntry) ![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    try model.appendTableHeader(allocator, &out, &.{ "Plan", "Status", "Owner", "Scope" });
    for (plans) |plan| {
        if (std.mem.eql(u8, plan.status, "Done")) continue;
        const row = [_][]const u8{
            try std.fmt.allocPrint(allocator, "[{s}](./{s}.md)", .{ plan.title, plan.slug }),
            plan.status,
            plan.owner,
            plan.scope,
        };
        defer allocator.free(row[0]);
        try model.appendTableRow(allocator, &out, &row);
    }
    return out.toOwnedSlice(allocator);
}

fn buildHorizonCounts(
    allocator: std.mem.Allocator,
    roadmap_entries: []const model.RoadmapDocEntry,
) ![]u8 {
    var now_count: usize = 0;
    var next_count: usize = 0;
    var later_count: usize = 0;

    for (roadmap_entries) |entry| {
        if (std.mem.eql(u8, entry.horizon, "Now")) now_count += 1;
        if (std.mem.eql(u8, entry.horizon, "Next")) next_count += 1;
        if (std.mem.eql(u8, entry.horizon, "Later")) later_count += 1;
    }

    return std.fmt.allocPrint(allocator,
        \\## Roadmap Horizons
        \\
        \\- Now: **{d}** item(s)
        \\- Next: **{d}** item(s)
        \\- Later: **{d}** item(s)
        \\
        \\Roadmap guide: [../roadmap/](../roadmap/)
    , .{
        now_count,
        next_count,
        later_count,
    });
}

fn relatedRoadmapForPlan(
    allocator: std.mem.Allocator,
    plan_slug: []const u8,
    roadmap_entries: []const model.RoadmapDocEntry,
) ![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    try out.appendSlice(allocator, "## Related Roadmap Items\n\n");
    try model.appendTableHeader(allocator, &out, &.{ "ID", "Item", "Track", "Horizon", "Status", "Gate" });

    var count: usize = 0;
    for (roadmap_entries) |entry| {
        if (!std.mem.eql(u8, entry.plan_slug, plan_slug)) continue;
        const row = [_][]const u8{
            entry.id,
            entry.title,
            entry.track,
            entry.horizon,
            entry.status,
            entry.validation_gate,
        };
        try model.appendTableRow(allocator, &out, &row);
        count += 1;
    }
    if (count == 0) {
        try out.appendSlice(allocator, "\n_No linked roadmap entries._\n");
    }

    try out.appendSlice(allocator, "\nRoadmap guide: [../roadmap/](../roadmap/)\n");
    return out.toOwnedSlice(allocator);
}

fn appendFmt(
    allocator: std.mem.Allocator,
    out: *std.ArrayListUnmanaged(u8),
    comptime fmt: []const u8,
    args: anytype,
) !void {
    const text = try std.fmt.allocPrint(allocator, fmt, args);
    defer allocator.free(text);
    try out.appendSlice(allocator, text);
}

const default_index_template =
    \\## Summary
    \\{{INDEX_SUMMARY}}
    \\
    \\## Active Plans
    \\{{PLANS_TABLE}}
    \\
    \\{{RELATED_ROADMAP}}
;

const default_plan_template =
    \\## Status
    \\- Status: **{{STATUS}}**
    \\- Owner: **{{OWNER}}**
    \\
    \\## Scope
    \\{{SCOPE}}
    \\
    \\## Success Criteria
    \\{{SUCCESS_CRITERIA}}
    \\
    \\## Validation Gates
    \\{{GATE_COMMANDS}}
    \\
    \\## Milestones
    \\{{MILESTONES}}
    \\
    \\{{RELATED_ROADMAP}}
;

test "formatBullets emits deterministic markdown list" {
    const output = try formatBullets(std.testing.allocator, &.{ "A", "B" });
    defer std.testing.allocator.free(output);

    try std.testing.expectEqualStrings("- A\n- B\n", output);
}

test "buildPlansTable links active plans only" {
    const plans: []const model.PlanDocEntry = &.{
        .{
            .slug = "a",
            .title = "Plan A",
            .status = "In Progress",
            .status_order = 0,
            .owner = "Abbey",
            .scope = "Scope A",
            .success_criteria = &.{},
            .gate_commands = &.{},
            .milestones = &.{},
        },
        .{
            .slug = "b",
            .title = "Plan B",
            .status = "Done",
            .status_order = 3,
            .owner = "Abbey",
            .scope = "Scope B",
            .success_criteria = &.{},
            .gate_commands = &.{},
            .milestones = &.{},
        },
    };

    const table = try buildPlansTable(std.testing.allocator, plans);
    defer std.testing.allocator.free(table);

    try std.testing.expect(std.mem.indexOf(u8, table, "./a.md") != null);
    try std.testing.expect(std.mem.indexOf(u8, table, "./b.md") == null);
}
