//! `abi agent tui` — interactive REPL with plugin slash-commands and context.

const std = @import("std");
const abi = @import("abi");

/// Wrapper that creates a PluginManager, loads bundled plugins, and dispatches
/// a plugin slash-command through its `run` method. Designed to be passed as
/// `ReplConfig.plugin_dispatch` from the TUI layer.
fn dispatchPluginCommand(allocator: std.mem.Allocator, plugin: []const u8, cmd_name: []const u8, arg: []const u8) ![]u8 {
    var pm = abi.plugins.PluginManager.init(allocator);
    defer pm.deinit();
    abi.plugins.loadBundled(&pm);
    // Mirror `__context__:<name>`: slash-commands reach plugins as `__cmd__:<name>`
    // with an optional newline-delimited argument payload.
    if (cmd_name.len == 0) return pm.run(allocator, plugin, arg);
    const input = if (arg.len == 0)
        try std.fmt.allocPrint(allocator, "__cmd__:{s}", .{cmd_name})
    else
        try std.fmt.allocPrint(allocator, "__cmd__:{s}\n{s}", .{ cmd_name, arg });
    defer allocator.free(input);
    return pm.run(allocator, plugin, input);
}

pub fn handleAgentTuiNoArgs(io: std.Io, allocator: std.mem.Allocator) !u8 {
    var session = try abi.features.wdbx.durable_store.Session.open(io, allocator);
    defer session.deinit();
    const store = session.storePtr();

    var sched = abi.scheduler.Scheduler.init(allocator);
    defer sched.deinit();

    // Build plugin command list from the Registry
    var reg = abi.registry.Registry.init(allocator);
    defer reg.deinit();
    try reg.loadPlugins();

    var plugin_cmds = std.ArrayListUnmanaged(abi.features.tui.PluginSlashCommand).empty;
    defer {
        for (plugin_cmds.items) |pc| {
            allocator.free(pc.name);
            allocator.free(pc.summary);
            allocator.free(pc.plugin);
            for (pc.aliases) |a| allocator.free(a);
            allocator.free(pc.aliases);
        }
        plugin_cmds.deinit(allocator);
    }

    {
        const plugins = try reg.snapshotPlugins(allocator);
        defer abi.registry.Registry.freePluginSnapshot(allocator, plugins);
        for (plugins) |plugin| {
            for (plugin.commands) |cmd| {
                const name = try allocator.dupe(u8, cmd.name);
                errdefer allocator.free(name);
                const summary = try allocator.dupe(u8, cmd.summary);
                errdefer allocator.free(summary);
                const plugin_name = try allocator.dupe(u8, plugin.name);
                errdefer allocator.free(plugin_name);
                const aliases = try allocator.alloc([]const u8, cmd.aliases.len);
                errdefer allocator.free(aliases);
                for (cmd.aliases, 0..) |a, j| {
                    aliases[j] = try allocator.dupe(u8, a);
                }
                try plugin_cmds.append(allocator, .{
                    .name = name,
                    .summary = summary,
                    .plugin = plugin_name,
                    .aliases = aliases,
                });
            }
        }
    }

    // Collect context snippets from plugin context providers
    var pm = abi.plugins.PluginManager.init(allocator);
    defer pm.deinit();
    abi.plugins.loadBundled(&pm);
    const context_snippets = try pm.collectContextSnippets(allocator);
    errdefer allocator.free(context_snippets);

    var repl = abi.features.tui.ReplLoop.init(allocator, store, &sched, .{
        .plugin_commands = plugin_cmds.items,
        .plugin_dispatch = dispatchPluginCommand,
        .context_snippets = context_snippets,
    });
    defer {
        allocator.free(context_snippets);
        repl.deinit();
    }
    repl.run(io) catch |err| {
        if (err == error.FeatureDisabled) {
            std.debug.print("error: TUI feature is disabled in this build; rebuild without -Dfeat-tui=false to use `abi agent tui`\n", .{});
            return 1;
        }
        std.debug.print("error: interactive REPL failed: {s}\n", .{@errorName(err)});
        return 1;
    };
    return 0;
}

test {
    std.testing.refAllDecls(@This());
}
