const std = @import("std");
const model = @import("model.zig");
const site_map = @import("site_map.zig");

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
    build_meta: model.BuildMeta,
    modules: []const model.ModuleDoc,
    commands: []const model.CliCommand,
    readmes: []const model.ReadmeSummary,
    roadmap_entries: []const model.RoadmapDocEntry,
    plan_entries: []const model.PlanDocEntry,
    outputs: *std.ArrayListUnmanaged(model.OutputFile),
) !void {
    for (site_map.guides) |guide| {
        const template = loadTemplate(allocator, io, cwd, guide) catch defaultTemplate;
        defer allocator.free(template);

        const auto_content = try buildAutoContent(
            allocator,
            guide.slug,
            guide.section,
            build_meta,
            modules,
            commands,
            readmes,
            roadmap_entries,
            plan_entries,
        );
        defer allocator.free(auto_content);

        const replaced = try applyTemplate(allocator, template, .{
            .title = guide.title,
            .description = guide.description,
            .auto_content = auto_content,
        });
        defer allocator.free(replaced);

        var page = std.ArrayListUnmanaged(u8).empty;
        defer page.deinit(allocator);

        try appendFmt(allocator, &page,
            \\---
            \\title: {s}
            \\description: {s}
            \\section: {s}
            \\order: {d}
            \\permalink: {s}
            \\---
            \\
            \\# {s}
            \\
        , .{
            guide.title,
            guide.description,
            guide.section,
            guide.order,
            guide.permalink,
            guide.title,
        });

        try page.appendSlice(allocator, replaced);
        try page.appendSlice(allocator, generated_footer);

        const out_path = try std.fmt.allocPrint(allocator, "docs/_docs/{s}.md", .{guide.slug});
        defer allocator.free(out_path);
        try model.pushOutput(allocator, outputs, out_path, page.items);
    }
}

const TemplateArgs = struct {
    title: []const u8,
    description: []const u8,
    auto_content: []const u8,
};

fn loadTemplate(
    allocator: std.mem.Allocator,
    io: std.Io,
    cwd: std.Io.Dir,
    guide: model.GuideSpec,
) ![]u8 {
    return cwd.readFileAlloc(io, guide.template_path, allocator, .limited(512 * 1024));
}

const defaultTemplate =
    \\{{AUTO_CONTENT}}
;

fn applyTemplate(allocator: std.mem.Allocator, template: []const u8, args: TemplateArgs) ![]u8 {
    const t1 = try replaceAll(allocator, template, "{{TITLE}}", args.title);
    defer allocator.free(t1);
    const t2 = try replaceAll(allocator, t1, "{{DESCRIPTION}}", args.description);
    defer allocator.free(t2);
    const t3 = try replaceAll(allocator, t2, "{{AUTO_CONTENT}}", args.auto_content);
    return t3;
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

fn buildAutoContent(
    allocator: std.mem.Allocator,
    slug: []const u8,
    section: []const u8,
    build_meta: model.BuildMeta,
    modules: []const model.ModuleDoc,
    commands: []const model.CliCommand,
    readmes: []const model.ReadmeSummary,
    roadmap_entries: []const model.RoadmapDocEntry,
    plan_entries: []const model.PlanDocEntry,
) ![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    try out.appendSlice(allocator, "## Overview\n\n");
    try appendFmt(
        allocator,
        &out,
        "This guide is generated from repository metadata for **{s}** coverage and stays deterministic across runs.\n\n",
        .{section},
    );

    try out.appendSlice(allocator, "## Build Snapshot\n\n");
    try appendFmt(
        allocator,
        &out,
        "- Zig pin: `{s}`\n- Main tests: `{d}` pass / `{d}` skip / `{d}` total\n- Feature tests: `{d}` pass / `{d}` total\n\n",
        .{
            build_meta.zig_version,
            build_meta.test_main_pass,
            build_meta.test_main_skip,
            build_meta.test_main_total,
            build_meta.test_feature_pass,
            build_meta.test_feature_total,
        },
    );

    if (std.mem.eql(u8, slug, "api")) {
        try appendApiSummary(allocator, &out, modules);
    } else if (std.mem.eql(u8, slug, "cli")) {
        try appendCliSummary(allocator, &out, commands);
    } else if (std.mem.eql(u8, slug, "getting-started")) {
        try appendGettingStartedFlow(allocator, &out);
        try appendModuleCoverage(allocator, &out, section, modules);
        try appendGettingStartedEntryPoints(allocator, &out, commands);
    } else if (std.mem.eql(u8, slug, "gpu") or std.mem.eql(u8, slug, "gpu-backends")) {
        try appendGpuSummary(allocator, &out, modules);
    } else if (std.mem.eql(u8, slug, "connectors")) {
        try appendConnectorsSummary(allocator, &out, readmes);
    } else if (std.mem.eql(u8, slug, "roadmap")) {
        try appendRoadmapSummary(allocator, &out, roadmap_entries, plan_entries);
    } else if (std.mem.eql(u8, slug, "installation")) {
        try out.appendSlice(allocator,
            \\## Toolchain
            \\
            \\Use `zvm use master` for developer convenience and keep checks compatible with `.zigversion`.
            \\
            \\```bash
            \\zig build toolchain-doctor
            \\zig build typecheck
            \\```
            \\
            \\
        );
    } else {
        try appendModuleCoverage(allocator, &out, section, modules);
        try appendCommandEntryPoints(allocator, &out, section, commands);
    }

    try out.appendSlice(allocator,
        \\## Validation Commands
        \\
        \\- `zig build typecheck`
        \\- `zig build check-docs`
        \\- `zig build run -- gendocs --check`
        \\
        \\## Navigation
        \\
        \\- API Reference: [../api/](../api/)
        \\- API App: [../api-app/](../api-app/)
        \\- Plans Index: [../plans/index.md](../plans/index.md)
        \\- Source Root: [GitHub src tree](https://github.com/donaldfilimon/abi/tree/master/src)
    );

    return out.toOwnedSlice(allocator);
}

fn appendApiSummary(
    allocator: std.mem.Allocator,
    out: *std.ArrayListUnmanaged(u8),
    modules: []const model.ModuleDoc,
) !void {
    try out.appendSlice(allocator, "## API Surface Summary\n\n");
    try model.appendTableHeader(allocator, out, &.{ "Module", "Category", "Flag" });
    for (modules) |mod| {
        const row = [_][]const u8{
            try std.fmt.allocPrint(allocator, "[{s}](../api/{s}.html)", .{ mod.name, mod.name }),
            mod.category.name(),
            try std.fmt.allocPrint(allocator, "`{s}`", .{mod.build_flag}),
        };
        defer allocator.free(row[0]);
        defer allocator.free(row[2]);
        try model.appendTableRow(allocator, out, &row);
    }
    try out.appendSlice(allocator, "\nSee generated API app: [../api-app/](../api-app/)\n\n");
}

fn appendCliSummary(
    allocator: std.mem.Allocator,
    out: *std.ArrayListUnmanaged(u8),
    commands: []const model.CliCommand,
) !void {
    var alias_total: usize = 0;
    var nested_total: usize = 0;
    for (commands) |command| {
        alias_total += command.aliases.len;
        nested_total += command.subcommands.len;
    }

    try appendFmt(
        allocator,
        out,
        "## Command Tree\n\nTop-level commands: **{d}** | aliases: **{d}** | structural subcommands: **{d}**\n\n",
        .{ commands.len, alias_total, nested_total },
    );

    try model.appendTableHeader(allocator, out, &.{ "Command", "Aliases", "Description", "Subcommands" });
    for (commands) |command| {
        const alias_list = try joinListOrDash(allocator, command.aliases);
        defer allocator.free(alias_list);

        const structural = try formatStructuralSubcommands(allocator, command.subcommands);
        defer allocator.free(structural);

        const row = [_][]const u8{
            try std.fmt.allocPrint(allocator, "`{s}`", .{command.name}),
            alias_list,
            command.description,
            if (structural.len > 0) structural else "—",
        };
        defer allocator.free(row[0]);
        try model.appendTableRow(allocator, out, &row);
    }
    try out.append(allocator, '\n');
}

fn appendGpuSummary(
    allocator: std.mem.Allocator,
    out: *std.ArrayListUnmanaged(u8),
    modules: []const model.ModuleDoc,
) !void {
    try out.appendSlice(allocator,
        \\## GPU Module Surface
        \\
        \\The GPU subsystem is namespaced under `abi.gpu` with runtime API groups: `backends`, `devices`, `runtime`, `policy`, `multi`, `factory`.
        \\
        \\### Related API modules
        \\
    );
    for (modules) |mod| {
        if (std.mem.indexOf(u8, mod.path, "features/gpu") == null) continue;
        try appendFmt(allocator, out, "- `{s}` ([api](../api/{s}.html))\n", .{ mod.path, mod.name });
    }
    try out.append(allocator, '\n');
}

fn appendGettingStartedFlow(
    allocator: std.mem.Allocator,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    try out.appendSlice(allocator,
        \\## Quickstart Commands
        \\
        \\```bash
        \\zig build
        \\zig build run -- --help
        \\zig build cli-tests
        \\zig build check-docs
        \\```
        \\
        \\## First Interactive Flows
        \\
        \\- `abi ui launch` — command launcher TUI
        \\- `abi ui gpu` — GPU dashboard
        \\- `abi llm providers` — inspect local routing state
        \\
        \\
    );
}

fn joinListOrDash(
    allocator: std.mem.Allocator,
    items: []const []const u8,
) ![]u8 {
    if (items.len == 0) return allocator.dupe(u8, "—");

    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    for (items, 0..) |item, idx| {
        if (idx > 0) try out.appendSlice(allocator, ", ");
        try out.appendSlice(allocator, item);
    }
    return out.toOwnedSlice(allocator);
}

fn appendGettingStartedEntryPoints(
    allocator: std.mem.Allocator,
    out: *std.ArrayListUnmanaged(u8),
    commands: []const model.CliCommand,
) !void {
    try out.appendSlice(allocator, "## Command Entry Points\n\n");
    const starters = [_][]const u8{
        "status",
        "system-info",
        "db",
        "llm",
        "ui",
        "gpu",
        "task",
        "train",
    };

    var count: usize = 0;
    for (starters) |name| {
        const cmd = findCommand(commands, name) orelse continue;
        try appendFmt(allocator, out, "- `abi {s}` — {s}\n", .{ cmd.name, cmd.description });
        count += 1;
    }
    if (count == 0) {
        try out.appendSlice(allocator, "- No starter commands found.\n");
    }
    try out.append(allocator, '\n');
}

fn appendConnectorsSummary(
    allocator: std.mem.Allocator,
    out: *std.ArrayListUnmanaged(u8),
    readmes: []const model.ReadmeSummary,
) !void {
    try out.appendSlice(allocator, "## Connector Source Modules\n\n");
    var count: usize = 0;
    for (readmes) |summary| {
        if (std.mem.indexOf(u8, summary.path, "services/connectors") == null) continue;
        count += 1;
        try appendFmt(allocator, out, "- `{s}`: {s}\n", .{ summary.path, summary.summary });
    }
    if (count == 0) {
        try out.appendSlice(allocator, "- No connector README summaries were indexed.\n");
    }
    try out.append(allocator, '\n');
}

fn appendRoadmapSummary(
    allocator: std.mem.Allocator,
    out: *std.ArrayListUnmanaged(u8),
    roadmap_entries: []const model.RoadmapDocEntry,
    plan_entries: []const model.PlanDocEntry,
) !void {
    try out.appendSlice(allocator,
        \\## Now / Next / Later
        \\
        \\Canonical execution roadmap generated from `src/services/tasks/roadmap_catalog.zig`.
        \\
    );

    try appendRoadmapHorizonTable(allocator, out, roadmap_entries, "Now");
    try appendRoadmapHorizonTable(allocator, out, roadmap_entries, "Next");
    try appendRoadmapHorizonTable(allocator, out, roadmap_entries, "Later");

    try out.appendSlice(allocator, "## Active Plans\n\n");
    try model.appendTableHeader(allocator, out, &.{ "Plan", "Status", "Owner", "Scope" });

    var count: usize = 0;
    for (plan_entries) |plan| {
        if (std.mem.eql(u8, plan.status, "Done")) continue;
        const plan_link = try std.fmt.allocPrint(allocator, "[{s}](../plans/{s}.md)", .{ plan.title, plan.slug });
        defer allocator.free(plan_link);
        try model.appendTableRow(allocator, out, &.{ plan_link, plan.status, plan.owner, plan.scope });
        count += 1;
    }
    if (count == 0) {
        try out.appendSlice(allocator, "\n_No active generated plans._\n");
    }
    try out.append(allocator, '\n');
}

fn appendRoadmapHorizonTable(
    allocator: std.mem.Allocator,
    out: *std.ArrayListUnmanaged(u8),
    roadmap_entries: []const model.RoadmapDocEntry,
    horizon: []const u8,
) !void {
    try appendFmt(allocator, out, "### {s}\n\n", .{horizon});
    try model.appendTableHeader(allocator, out, &.{ "Track", "Item", "Owner", "Status", "Validation Gate", "Plan" });

    var count: usize = 0;
    for (roadmap_entries) |entry| {
        if (!std.mem.eql(u8, entry.horizon, horizon)) continue;

        const plan_link = try std.fmt.allocPrint(allocator, "[{s}](../plans/{s}.md)", .{ entry.plan_title, entry.plan_slug });
        defer allocator.free(plan_link);

        const item = try std.fmt.allocPrint(allocator, "`{s}` {s}", .{ entry.id, entry.title });
        defer allocator.free(item);

        try model.appendTableRow(allocator, out, &.{
            entry.track,
            item,
            entry.owner,
            entry.status,
            entry.validation_gate,
            plan_link,
        });
        count += 1;
    }
    if (count == 0) {
        try out.appendSlice(allocator, "| — | — | — | — | — | — |\n");
    }
    try out.append(allocator, '\n');
}

fn appendModuleCoverage(
    allocator: std.mem.Allocator,
    out: *std.ArrayListUnmanaged(u8),
    section: []const u8,
    modules: []const model.ModuleDoc,
) !void {
    try out.appendSlice(allocator, "## Module Coverage\n\n");
    var count: usize = 0;
    for (modules) |mod| {
        if (!moduleMatchesSection(section, mod)) continue;
        count += 1;
        if (count > 10) break;
        try appendFmt(allocator, out, "- `{s}` ([api](../api/{s}.html))\n", .{ mod.path, mod.name });
    }
    if (count == 0) {
        try out.appendSlice(allocator, "- No section-specific module mapping available.\n");
    }
    try out.append(allocator, '\n');
}

fn appendCommandEntryPoints(
    allocator: std.mem.Allocator,
    out: *std.ArrayListUnmanaged(u8),
    section: []const u8,
    commands: []const model.CliCommand,
) !void {
    try out.appendSlice(allocator, "## Command Entry Points\n\n");
    var count: usize = 0;
    for (commands) |command| {
        if (!commandMatchesSection(section, command.name)) continue;
        count += 1;
        if (count > 8) break;
        try appendFmt(allocator, out, "- `abi {s}` — {s}\n", .{ command.name, command.description });
    }
    if (count == 0) {
        try out.appendSlice(allocator, "- No section-specific command mapping available.\n");
    }
    try out.append(allocator, '\n');
}

fn findCommand(commands: []const model.CliCommand, name: []const u8) ?model.CliCommand {
    for (commands) |command| {
        if (std.mem.eql(u8, command.name, name)) return command;
    }
    return null;
}

fn moduleMatchesSection(section: []const u8, mod: model.ModuleDoc) bool {
    if (std.mem.eql(u8, section, "AI")) return mod.category == .ai;
    if (std.mem.eql(u8, section, "Data")) return mod.category == .data;
    if (std.mem.eql(u8, section, "Infrastructure")) return mod.category == .infrastructure;
    if (std.mem.eql(u8, section, "Core")) return mod.category == .core or mod.category == .compute;
    if (std.mem.eql(u8, section, "Services")) return std.mem.indexOf(u8, mod.path, "services") != null;
    if (std.mem.eql(u8, section, "GPU")) return std.mem.indexOf(u8, mod.path, "features/gpu") != null;
    return true;
}

fn commandMatchesSection(section: []const u8, name: []const u8) bool {
    if (std.mem.eql(u8, section, "AI")) {
        return std.mem.eql(u8, name, "llm") or
            std.mem.eql(u8, name, "agent") or
            std.mem.eql(u8, name, "train") or
            std.mem.eql(u8, name, "model") or
            std.mem.eql(u8, name, "embed") or
            std.mem.eql(u8, name, "multi-agent") or
            std.mem.eql(u8, name, "ralph");
    }
    if (std.mem.eql(u8, section, "GPU")) {
        return std.mem.eql(u8, name, "gpu") or
            std.mem.eql(u8, name, "ui") or
            std.mem.eql(u8, name, "gpu-dashboard") or
            std.mem.eql(u8, name, "bench");
    }
    if (std.mem.eql(u8, section, "Data")) {
        return std.mem.eql(u8, name, "db") or
            std.mem.eql(u8, name, "convert") or
            std.mem.eql(u8, name, "task");
    }
    if (std.mem.eql(u8, section, "Infrastructure")) {
        return std.mem.eql(u8, name, "network") or
            std.mem.eql(u8, name, "mcp") or
            std.mem.eql(u8, name, "acp");
    }
    if (std.mem.eql(u8, section, "Operations")) {
        return std.mem.eql(u8, name, "status") or
            std.mem.eql(u8, name, "toolchain") or
            std.mem.eql(u8, name, "config") or
            std.mem.eql(u8, name, "plugins") or
            std.mem.eql(u8, name, "profile");
    }
    return true;
}

fn formatStructuralSubcommands(
    allocator: std.mem.Allocator,
    subcommands: []const []const u8,
) ![]u8 {
    for (subcommands) |sub| {
        if (std.mem.startsWith(u8, sub, "-")) return allocator.dupe(u8, "");
    }

    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    for (subcommands, 0..) |sub, idx| {
        if (idx > 0) try out.appendSlice(allocator, ", ");
        try out.appendSlice(allocator, sub);
    }
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

test "applyTemplate replaces known placeholders deterministically" {
    const template =
        \\# {{TITLE}}
        \\
        \\{{DESCRIPTION}}
        \\
        \\{{AUTO_CONTENT}}
    ;
    const rendered = try applyTemplate(std.testing.allocator, template, .{
        .title = "CLI",
        .description = "Command reference",
        .auto_content = "Auto body",
    });
    defer std.testing.allocator.free(rendered);

    try std.testing.expect(std.mem.indexOf(u8, rendered, "{{TITLE}}") == null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "Auto body") != null);
}

test "formatStructuralSubcommands hides option-token lists" {
    const list = [_][]const u8{ "--provider", "openai", "mistral" };
    const got = try formatStructuralSubcommands(std.testing.allocator, &list);
    defer std.testing.allocator.free(got);
    try std.testing.expectEqual(@as(usize, 0), got.len);
}

test "roadmap summary includes owner status gate and plan links" {
    var out = std.ArrayListUnmanaged(u8).empty;
    defer out.deinit(std.testing.allocator);

    const roadmap_entries: []const model.RoadmapDocEntry = &.{
        .{
            .id = "RM-001",
            .title = "Canonical sync",
            .summary = "sync",
            .track = "Docs",
            .track_order = 0,
            .horizon = "Now",
            .horizon_order = 0,
            .status = "In Progress",
            .status_order = 0,
            .owner = "Abbey",
            .validation_gate = "zig build check-docs",
            .plan_slug = "docs-roadmap-sync-v2",
            .plan_title = "Docs + Roadmap Canonical Sync",
        },
    };
    const plan_entries: []const model.PlanDocEntry = &.{
        .{
            .slug = "docs-roadmap-sync-v2",
            .title = "Docs + Roadmap Canonical Sync",
            .status = "In Progress",
            .status_order = 0,
            .owner = "Abbey",
            .scope = "scope",
            .success_criteria = &.{},
            .gate_commands = &.{},
            .milestones = &.{},
        },
    };

    try appendRoadmapSummary(std.testing.allocator, &out, roadmap_entries, plan_entries);

    try std.testing.expect(std.mem.indexOf(u8, out.items, "Owner") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.items, "Status") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.items, "Validation Gate") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.items, "../plans/docs-roadmap-sync-v2.md") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.items, "||") == null);
}
