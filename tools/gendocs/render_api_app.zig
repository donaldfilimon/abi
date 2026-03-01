const std = @import("std");
const model = @import("model.zig");
const site_map = @import("site_map.zig");

pub fn render(
    allocator: std.mem.Allocator,
    modules: []const model.ModuleDoc,
    commands: []const model.CliCommand,
    features: []const model.FeatureDoc,
    roadmap_entries: []const model.RoadmapDocEntry,
    plan_entries: []const model.PlanDocEntry,
    outputs: *std.ArrayListUnmanaged(model.OutputFile),
) !void {
    try renderZonData(allocator, modules, commands, features, roadmap_entries, plan_entries, outputs);
    try model.pushOutput(allocator, outputs, "docs/index.html", html_template);
    try model.pushOutput(allocator, outputs, "docs/index.css", css_template);
    try model.pushOutput(allocator, outputs, "docs/index.js", js_template);
}

fn renderZonData(
    allocator: std.mem.Allocator,
    modules: []const model.ModuleDoc,
    commands: []const model.CliCommand,
    features: []const model.FeatureDoc,
    roadmap_entries: []const model.RoadmapDocEntry,
    plan_entries: []const model.PlanDocEntry,
    outputs: *std.ArrayListUnmanaged(model.OutputFile),
) !void {
    const ZonSymbol = struct {
        anchor: []const u8,
        signature: []const u8,
        doc: []const u8,
        kind: []const u8,
        line: usize,
    };
    const ZonModule = struct {
        name: []const u8,
        path: []const u8,
        description: []const u8,
        category: []const u8,
        build_flag: []const u8,
        symbols: []ZonSymbol,
    };

    var module_zon = try allocator.alloc(ZonModule, modules.len);
    defer {
        for (module_zon) |item| allocator.free(item.symbols);
        allocator.free(module_zon);
    }

    for (modules, 0..) |mod, idx| {
        const symbols = try allocator.alloc(ZonSymbol, mod.symbols.len);
        for (mod.symbols, 0..) |symbol, sidx| {
            symbols[sidx] = .{
                .anchor = symbol.anchor,
                .signature = symbol.signature,
                .doc = symbol.doc,
                .kind = symbol.kind.badge(),
                .line = symbol.line,
            };
        }

        module_zon[idx] = .{
            .name = mod.name,
            .path = mod.path,
            .description = mod.description,
            .category = mod.category.name(),
            .build_flag = mod.build_flag,
            .symbols = symbols,
        };
    }

    const ZonCommand = struct {
        name: []const u8,
        description: []const u8,
        aliases: []const []const u8,
        subcommands: []const []const u8,
    };

    var command_zon = try allocator.alloc(ZonCommand, commands.len);
    defer allocator.free(command_zon);
    for (commands, 0..) |command, idx| {
        command_zon[idx] = .{
            .name = command.name,
            .description = command.description,
            .aliases = command.aliases,
            .subcommands = command.subcommands,
        };
    }

    const ZonGuide = struct {
        slug: []const u8,
        title: []const u8,
        section: []const u8,
        permalink: []const u8,
        description: []const u8,
    };

    var guides_zon = try allocator.alloc(ZonGuide, site_map.guides.len);
    defer allocator.free(guides_zon);
    for (site_map.guides, 0..) |guide, idx| {
        guides_zon[idx] = .{
            .slug = guide.slug,
            .title = guide.title,
            .section = guide.section,
            .permalink = guide.permalink,
            .description = guide.description,
        };
    }

    const modules_zon_text = try stringifyAlloc(allocator, module_zon);
    defer allocator.free(modules_zon_text);
    try model.pushOutput(allocator, outputs, "docs/data/modules.zon", modules_zon_text);

    const commands_zon_text = try stringifyAlloc(allocator, command_zon);
    defer allocator.free(commands_zon_text);
    try model.pushOutput(allocator, outputs, "docs/data/commands.zon", commands_zon_text);

    const ZonFeature = struct {
        name: []const u8,
        description: []const u8,
        compile_flag: []const u8,
        parent: []const u8,
        real_module_path: []const u8,
        stub_module_path: []const u8,
    };

    var features_zon = try allocator.alloc(ZonFeature, features.len);
    defer allocator.free(features_zon);
    for (features, 0..) |feat, idx| {
        features_zon[idx] = .{
            .name = feat.name,
            .description = feat.description,
            .compile_flag = feat.compile_flag,
            .parent = feat.parent,
            .real_module_path = feat.real_module_path,
            .stub_module_path = feat.stub_module_path,
        };
    }

    const features_zon_text = try stringifyAlloc(allocator, features_zon);
    defer allocator.free(features_zon_text);
    try model.pushOutput(allocator, outputs, "docs/data/features.zon", features_zon_text);

    const guides_zon_text = try stringifyAlloc(allocator, guides_zon);
    defer allocator.free(guides_zon_text);
    try model.pushOutput(allocator, outputs, "docs/data/guides.zon", guides_zon_text);

    const ZonPlan = struct {
        slug: []const u8,
        title: []const u8,
        status: []const u8,
        owner: []const u8,
        scope: []const u8,
        gate_commands: []const []const u8,
    };

    var plans_zon = try allocator.alloc(ZonPlan, plan_entries.len);
    defer allocator.free(plans_zon);
    for (plan_entries, 0..) |plan, idx| {
        plans_zon[idx] = .{
            .slug = plan.slug,
            .title = plan.title,
            .status = plan.status,
            .owner = plan.owner,
            .scope = plan.scope,
            .gate_commands = plan.gate_commands,
        };
    }

    const plans_zon_text = try stringifyAlloc(allocator, plans_zon);
    defer allocator.free(plans_zon_text);
    try model.pushOutput(allocator, outputs, "docs/data/plans.zon", plans_zon_text);

    const ZonRoadmap = struct {
        id: []const u8,
        title: []const u8,
        summary: []const u8,
        track: []const u8,
        horizon: []const u8,
        status: []const u8,
        owner: []const u8,
        validation_gate: []const u8,
        plan_slug: []const u8,
        plan_title: []const u8,
    };

    var roadmap_zon = try allocator.alloc(ZonRoadmap, roadmap_entries.len);
    defer allocator.free(roadmap_zon);
    for (roadmap_entries, 0..) |entry, idx| {
        roadmap_zon[idx] = .{
            .id = entry.id,
            .title = entry.title,
            .summary = entry.summary,
            .track = entry.track,
            .horizon = entry.horizon,
            .status = entry.status,
            .owner = entry.owner,
            .validation_gate = entry.validation_gate,
            .plan_slug = entry.plan_slug,
            .plan_title = entry.plan_title,
        };
    }

    const roadmap_zon_text = try stringifyAlloc(allocator, roadmap_zon);
    defer allocator.free(roadmap_zon_text);
    try model.pushOutput(allocator, outputs, "docs/data/roadmap.zon", roadmap_zon_text);
}

fn stringifyAlloc(allocator: std.mem.Allocator, value: anytype) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();

    // In Zig 0.16, we use std.zon.stringify.serialize
    try std.zon.stringify.serialize(value, .{}, &out.writer);
    try out.writer.writeByte('\n');
    return out.toOwnedSlice();
}

const html_template = @embedFile("assets/index.html");
const css_template = @embedFile("assets/index.css");
const js_template = @embedFile("assets/index.js");
