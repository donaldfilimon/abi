const std = @import("std");
const abi = @import("abi");

const plugins = abi.ai.llm.providers.plugins;

pub fn runPlugins(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
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
        return addHttpPlugin(allocator, args[1..]);
    }

    if (std.mem.eql(u8, sub, "add-native")) {
        return addNativePlugin(allocator, args[1..]);
    }

    if (std.mem.eql(u8, sub, "enable")) {
        return setEnabled(allocator, args[1..], true);
    }

    if (std.mem.eql(u8, sub, "disable")) {
        return setEnabled(allocator, args[1..], false);
    }

    if (std.mem.eql(u8, sub, "remove")) {
        return removePlugin(allocator, args[1..]);
    }

    std.debug.print("Unknown llm plugins subcommand: {s}\n", .{sub});
    printPluginsHelp();
}

fn listPlugins(allocator: std.mem.Allocator) !void {
    var manifest = try plugins.manifest.loadDefault(allocator);
    defer manifest.deinit();

    std.debug.print("LLM plugins\n", .{});
    std.debug.print("===========\n", .{});

    if (manifest.entries.items.len == 0) {
        std.debug.print("No plugins configured.\n", .{});
        return;
    }

    for (manifest.entries.items) |entry| {
        std.debug.print(
            "- {s} [{s}] {s}\n",
            .{ entry.id, entry.kind.label(), if (entry.enabled) "enabled" else "disabled" },
        );

        switch (entry.kind) {
            .http => {
                if (entry.base_url) |base_url| {
                    std.debug.print("    base_url: {s}\n", .{base_url});
                }
                if (entry.model) |model| {
                    std.debug.print("    model: {s}\n", .{model});
                }
                if (entry.api_key_env) |api_key_env| {
                    std.debug.print("    api_key_env: {s}\n", .{api_key_env});
                }
            },
            .native => {
                if (entry.library_path) |library_path| {
                    std.debug.print("    library_path: {s}\n", .{library_path});
                }
                if (entry.symbol) |symbol| {
                    std.debug.print("    symbol: {s}\n", .{symbol});
                }
            },
        }
    }
}

fn addHttpPlugin(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (args.len == 0) {
        std.debug.print("Usage: abi llm plugins add-http <id> --url <base_url> [--model <model>] [--api-key-env <env>]\n", .{});
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
        std.debug.print("Error: --url is required\n", .{});
        return;
    }

    var manifest = try plugins.manifest.loadDefault(allocator);
    defer manifest.deinit();

    try manifest.addOrUpdateHttp(id, url.?, model, api_key_env);
    try plugins.manifest.saveDefault(&manifest);

    std.debug.print("Configured HTTP plugin '{s}'.\n", .{id});
}

fn addNativePlugin(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (args.len == 0) {
        std.debug.print("Usage: abi llm plugins add-native <id> --library <path> [--symbol <name>]\n", .{});
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
        std.debug.print("Error: --library is required\n", .{});
        return;
    }

    var manifest = try plugins.manifest.loadDefault(allocator);
    defer manifest.deinit();

    try manifest.addOrUpdateNative(id, library_path.?, symbol);
    try plugins.manifest.saveDefault(&manifest);

    std.debug.print("Configured native plugin '{s}'.\n", .{id});
}

fn setEnabled(allocator: std.mem.Allocator, args: []const [:0]const u8, enabled: bool) !void {
    if (args.len == 0) {
        std.debug.print("Usage: abi llm plugins {s} <id>\n", .{if (enabled) "enable" else "disable"});
        return;
    }

    const id = std.mem.sliceTo(args[0], 0);
    var manifest = try plugins.manifest.loadDefault(allocator);
    defer manifest.deinit();

    if (!manifest.setEnabled(id, enabled)) {
        std.debug.print("Plugin not found: {s}\n", .{id});
        return;
    }

    try plugins.manifest.saveDefault(&manifest);
    std.debug.print("Plugin '{s}' {s}.\n", .{ id, if (enabled) "enabled" else "disabled" });
}

fn removePlugin(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (args.len == 0) {
        std.debug.print("Usage: abi llm plugins remove <id>\n", .{});
        return;
    }

    const id = std.mem.sliceTo(args[0], 0);
    var manifest = try plugins.manifest.loadDefault(allocator);
    defer manifest.deinit();

    if (!manifest.remove(id)) {
        std.debug.print("Plugin not found: {s}\n", .{id});
        return;
    }

    try plugins.manifest.saveDefault(&manifest);
    std.debug.print("Plugin '{s}' removed.\n", .{id});
}

pub fn printPluginsHelp() void {
    std.debug.print(
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
