const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");

const plugins = abi.ai.llm.providers.plugins;

pub fn runPlugins(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (args.len == 0) {
        return listPlugins(allocator);
    }

    const sub = std.mem.sliceTo(args[0], 0);
    if (std.mem.eql(u8, sub, "help") or std.mem.eql(u8, sub, "--help") or std.mem.eql(u8, sub, "-h")) {
        printPluginsHelp();
        return;
    }

    if (std.mem.eql(u8, sub, "list")) {
        return listPlugins(allocator);
    }

    if (std.mem.eql(u8, sub, "add-http")) {
        return addHttpPlugin(ctx, args[1..]);
    }

    if (std.mem.eql(u8, sub, "add-native")) {
        return addNativePlugin(ctx, args[1..]);
    }

    if (std.mem.eql(u8, sub, "enable")) {
        return setEnabled(allocator, args[1..], true);
    }

    if (std.mem.eql(u8, sub, "disable")) {
        return setEnabled(allocator, args[1..], false);
    }

    if (std.mem.eql(u8, sub, "remove")) {
        return removePlugin(ctx, args[1..]);
    }

    utils.output.printError("Unknown llm plugins subcommand: {s}", .{sub});
    printPluginsHelp();
}

fn listPlugins(allocator: std.mem.Allocator) !void {
    var manifest = try plugins.manifest.loadDefault(allocator);
    defer manifest.deinit();

    utils.output.printHeader("LLM plugins");

    if (manifest.entries.items.len == 0) {
        utils.output.println("No plugins configured.", .{});
        return;
    }

    for (manifest.entries.items) |entry| {
        utils.output.println(
            "- {s} [{s}] {s}",
            .{ entry.id, entry.kind.label(), if (entry.enabled) "enabled" else "disabled" },
        );

        switch (entry.kind) {
            .http => {
                if (entry.base_url) |base_url| {
                    utils.output.printKeyValue("    base_url", base_url);
                }
                if (entry.model) |model| {
                    utils.output.printKeyValue("    model", model);
                }
                if (entry.api_key_env) |api_key_env| {
                    utils.output.printKeyValue("    api_key_env", api_key_env);
                }
            },
            .native => {
                if (entry.library_path) |library_path| {
                    utils.output.printKeyValue("    library_path", library_path);
                }
                if (entry.symbol) |symbol| {
                    utils.output.printKeyValue("    symbol", symbol);
                }
            },
        }
    }
}

fn addHttpPlugin(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (args.len == 0) {
        utils.output.println("Usage: abi llm plugins add-http <id> --url <base_url> [--model <model>] [--api-key-env <env>]", .{});
        return;
    }

    const id = std.mem.sliceTo(args[0], 0);
    var url: ?[]const u8 = null;
    var model: ?[]const u8 = null;
    var api_key_env: ?[]const u8 = null;

    var i: usize = 1;
    while (i < args.len) {
        const arg = std.mem.sliceTo(args[i], 0);
        i += 1;

        if (std.mem.eql(u8, arg, "--url") and i < args.len) {
            url = std.mem.sliceTo(args[i], 0);
            i += 1;
            continue;
        }

        if (std.mem.eql(u8, arg, "--model") and i < args.len) {
            model = std.mem.sliceTo(args[i], 0);
            i += 1;
            continue;
        }

        if (std.mem.eql(u8, arg, "--api-key-env") and i < args.len) {
            api_key_env = std.mem.sliceTo(args[i], 0);
            i += 1;
            continue;
        }
    }

    if (url == null) {
        utils.output.printError("--url is required", .{});
        return;
    }

    var manifest = try plugins.manifest.loadDefault(allocator);
    defer manifest.deinit();

    try manifest.addOrUpdateHttp(id, url.?, model, api_key_env);
    try plugins.manifest.saveDefault(&manifest);

    utils.output.printSuccess("Configured HTTP plugin '{s}'.", .{id});
}

fn addNativePlugin(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (args.len == 0) {
        utils.output.println("Usage: abi llm plugins add-native <id> --library <path> [--symbol <name>]", .{});
        return;
    }

    const id = std.mem.sliceTo(args[0], 0);
    var library_path: ?[]const u8 = null;
    var symbol: ?[]const u8 = null;

    var i: usize = 1;
    while (i < args.len) {
        const arg = std.mem.sliceTo(args[i], 0);
        i += 1;

        if (std.mem.eql(u8, arg, "--library") and i < args.len) {
            library_path = std.mem.sliceTo(args[i], 0);
            i += 1;
            continue;
        }

        if (std.mem.eql(u8, arg, "--symbol") and i < args.len) {
            symbol = std.mem.sliceTo(args[i], 0);
            i += 1;
            continue;
        }
    }

    if (library_path == null) {
        utils.output.printError("--library is required", .{});
        return;
    }

    var manifest = try plugins.manifest.loadDefault(allocator);
    defer manifest.deinit();

    try manifest.addOrUpdateNative(id, library_path.?, symbol);
    try plugins.manifest.saveDefault(&manifest);

    utils.output.printSuccess("Configured native plugin '{s}'.", .{id});
}

fn setEnabled(allocator: std.mem.Allocator, args: []const [:0]const u8, enabled: bool) !void {
    if (args.len == 0) {
        utils.output.println("Usage: abi llm plugins {s} <id>", .{if (enabled) "enable" else "disable"});
        return;
    }

    const id = std.mem.sliceTo(args[0], 0);
    var manifest = try plugins.manifest.loadDefault(allocator);
    defer manifest.deinit();

    if (!manifest.setEnabled(id, enabled)) {
        utils.output.printError("Plugin not found: {s}", .{id});
        return;
    }

    try plugins.manifest.saveDefault(&manifest);
    utils.output.printSuccess("Plugin '{s}' {s}.", .{ id, if (enabled) "enabled" else "disabled" });
}

fn removePlugin(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (args.len == 0) {
        utils.output.println("Usage: abi llm plugins remove <id>", .{});
        return;
    }

    const id = std.mem.sliceTo(args[0], 0);
    var manifest = try plugins.manifest.loadDefault(allocator);
    defer manifest.deinit();

    if (!manifest.remove(id)) {
        utils.output.printError("Plugin not found: {s}", .{id});
        return;
    }

    try plugins.manifest.saveDefault(&manifest);
    utils.output.printSuccess("Plugin '{s}' removed.", .{id});
}

pub fn printPluginsHelp() void {
    utils.output.print(
        "Usage: abi llm plugins <subcommand> [options]\\n\\n" ++
            "Subcommands:\\n" ++
            "  list\\n" ++
            "  add-http <id> --url <base_url> [--model <model>] [--api-key-env <env>]\\n" ++
            "  add-native <id> --library <path> [--symbol <name>]\\n" ++
            "  enable <id>\\n" ++
            "  disable <id>\\n" ++
            "  remove <id>\\n",
        .{},
    );
}

test {
    std.testing.refAllDecls(@This());
}
