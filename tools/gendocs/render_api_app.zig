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
    try renderJsonData(allocator, modules, commands, features, roadmap_entries, plan_entries, outputs);
    try model.pushOutput(allocator, outputs, "docs/index.html", html_template);
    try model.pushOutput(allocator, outputs, "docs/index.css", css_template);
    try model.pushOutput(allocator, outputs, "docs/index.js", js_template);
}

fn renderJsonData(
    allocator: std.mem.Allocator,
    modules: []const model.ModuleDoc,
    commands: []const model.CliCommand,
    features: []const model.FeatureDoc,
    roadmap_entries: []const model.RoadmapDocEntry,
    plan_entries: []const model.PlanDocEntry,
    outputs: *std.ArrayListUnmanaged(model.OutputFile),
) !void {
    const JsonSymbol = struct {
        anchor: []const u8,
        signature: []const u8,
        doc: []const u8,
        kind: []const u8,
        line: usize,
    };
    const JsonModule = struct {
        name: []const u8,
        path: []const u8,
        description: []const u8,
        category: []const u8,
        build_flag: []const u8,
        symbols: []JsonSymbol,
    };

    var module_json = try allocator.alloc(JsonModule, modules.len);
    defer {
        for (module_json) |item| allocator.free(item.symbols);
        allocator.free(module_json);
    }

    for (modules, 0..) |mod, idx| {
        const symbols = try allocator.alloc(JsonSymbol, mod.symbols.len);
        for (mod.symbols, 0..) |symbol, sidx| {
            symbols[sidx] = .{
                .anchor = symbol.anchor,
                .signature = symbol.signature,
                .doc = symbol.doc,
                .kind = symbol.kind.badge(),
                .line = symbol.line,
            };
        }

        module_json[idx] = .{
            .name = mod.name,
            .path = mod.path,
            .description = mod.description,
            .category = mod.category.name(),
            .build_flag = mod.build_flag,
            .symbols = symbols,
        };
    }

    const JsonCommand = struct {
        name: []const u8,
        description: []const u8,
        aliases: []const []const u8,
        subcommands: []const []const u8,
    };

    var command_json = try allocator.alloc(JsonCommand, commands.len);
    defer allocator.free(command_json);
    for (commands, 0..) |command, idx| {
        command_json[idx] = .{
            .name = command.name,
            .description = command.description,
            .aliases = command.aliases,
            .subcommands = command.subcommands,
        };
    }

    const JsonGuide = struct {
        slug: []const u8,
        title: []const u8,
        section: []const u8,
        permalink: []const u8,
        description: []const u8,
    };

    var guides_json = try allocator.alloc(JsonGuide, site_map.guides.len);
    defer allocator.free(guides_json);
    for (site_map.guides, 0..) |guide, idx| {
        guides_json[idx] = .{
            .slug = guide.slug,
            .title = guide.title,
            .section = guide.section,
            .permalink = guide.permalink,
            .description = guide.description,
        };
    }

    const modules_json_text = try stringifyAlloc(allocator, module_json);
    defer allocator.free(modules_json_text);
    try model.pushOutput(allocator, outputs, "docs/data/modules.json", modules_json_text);

    const commands_json_text = try stringifyAlloc(allocator, command_json);
    defer allocator.free(commands_json_text);
    try model.pushOutput(allocator, outputs, "docs/data/commands.json", commands_json_text);

    const JsonFeature = struct {
        name: []const u8,
        description: []const u8,
        compile_flag: []const u8,
        parent: []const u8,
        real_module_path: []const u8,
        stub_module_path: []const u8,
    };

    var features_json = try allocator.alloc(JsonFeature, features.len);
    defer allocator.free(features_json);
    for (features, 0..) |feat, idx| {
        features_json[idx] = .{
            .name = feat.name,
            .description = feat.description,
            .compile_flag = feat.compile_flag,
            .parent = feat.parent,
            .real_module_path = feat.real_module_path,
            .stub_module_path = feat.stub_module_path,
        };
    }

    const features_json_text = try stringifyAlloc(allocator, features_json);
    defer allocator.free(features_json_text);
    try model.pushOutput(allocator, outputs, "docs/data/features.json", features_json_text);

    const guides_json_text = try stringifyAlloc(allocator, guides_json);
    defer allocator.free(guides_json_text);
    try model.pushOutput(allocator, outputs, "docs/data/guides.json", guides_json_text);

    const JsonPlan = struct {
        slug: []const u8,
        title: []const u8,
        status: []const u8,
        owner: []const u8,
        scope: []const u8,
        gate_commands: []const []const u8,
    };

    var plans_json = try allocator.alloc(JsonPlan, plan_entries.len);
    defer allocator.free(plans_json);
    for (plan_entries, 0..) |plan, idx| {
        plans_json[idx] = .{
            .slug = plan.slug,
            .title = plan.title,
            .status = plan.status,
            .owner = plan.owner,
            .scope = plan.scope,
            .gate_commands = plan.gate_commands,
        };
    }

    const plans_json_text = try stringifyAlloc(allocator, plans_json);
    defer allocator.free(plans_json_text);
    try model.pushOutput(allocator, outputs, "docs/data/plans.json", plans_json_text);

    const JsonRoadmap = struct {
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

    var roadmap_json = try allocator.alloc(JsonRoadmap, roadmap_entries.len);
    defer allocator.free(roadmap_json);
    for (roadmap_entries, 0..) |entry, idx| {
        roadmap_json[idx] = .{
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

    const roadmap_json_text = try stringifyAlloc(allocator, roadmap_json);
    defer allocator.free(roadmap_json_text);
    try model.pushOutput(allocator, outputs, "docs/data/roadmap.json", roadmap_json_text);
}

fn stringifyAlloc(allocator: std.mem.Allocator, value: anytype) ![]u8 {
    var json_writer: std.Io.Writer.Allocating = .init(allocator);
    defer json_writer.deinit();

    try std.json.Stringify.value(value, .{ .whitespace = .indent_2 }, &json_writer.writer);
    try json_writer.writer.writeByte('\n');
    return json_writer.toOwnedSlice();
}

const html_template = @embedFile("assets/index.html");
const css_template = @embedFile("assets/index.css");
const js_template = @embedFile("assets/index.js");
