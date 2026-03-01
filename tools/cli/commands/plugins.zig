//! Plugin management commands for ABI CLI.
//!
//! Manage framework plugins: list, enable, disable, info.
//! Plugin state is persisted to the platform-specific ABI app root.

const std = @import("std");
const abi = @import("abi");
const command_mod = @import("../command.zig");
const context_mod = @import("../framework/context.zig");
const utils = @import("../utils/mod.zig");
const cli_io = utils.io_backend;
const app_paths = abi.services.shared.app_paths;

const PluginInfo = struct {
    name: []const u8,
    version: []const u8,
    description: []const u8,
    enabled: bool = false,
};

const builtin_plugins = [_]PluginInfo{
    .{ .name = "openai-connector", .version = "1.0.0", .description = "OpenAI LLM service connector" },
    .{ .name = "anthropic-connector", .version = "1.0.0", .description = "Anthropic LLM service connector" },
    .{ .name = "ollama-connector", .version = "1.0.0", .description = "Ollama local LLM service connector" },
    .{ .name = "discord-bot", .version = "0.5.0", .description = "Discord integration for ABI agents" },
    .{ .name = "vector-search", .version = "1.2.0", .description = "Advanced HNSW vector search engine" },
    .{ .name = "benchmarks", .version = "1.0.0", .description = "Performance benchmarking tools" },
};

const PluginState = struct {
    enabled_plugins: std.StringHashMapUnmanaged(bool),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) PluginState {
        return .{
            .enabled_plugins = .{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *PluginState) void {
        self.enabled_plugins.deinit(self.allocator);
    }

    pub fn isEnabled(self: *const PluginState, name: []const u8) bool {
        return self.enabled_plugins.get(name) orelse false;
    }

    pub fn setEnabled(self: *PluginState, name: []const u8, enabled: bool) !void {
        try self.enabled_plugins.put(self.allocator, name, enabled);
    }
};

const PluginLoadState = enum {
    loaded,
    missing,
};

const PluginEntry = struct {
    name: []const u8,
    enabled: bool,
};

const PersistedPluginState = struct {
    plugins: []const PluginEntry = &[_]PluginEntry{},
};

fn getPluginsConfigPath(allocator: std.mem.Allocator) ![]u8 {
    return try app_paths.resolvePath(allocator, "plugins.zon");
}

fn tryLoadPluginStateFromPath(
    allocator: std.mem.Allocator,
    path: []const u8,
    state: *PluginState,
) !PluginLoadState {
    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    const content = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1024 * 1024)) catch |err| switch (err) {
        error.FileNotFound => return .missing,
        else => return err,
    };
    defer allocator.free(content);

    const content_z = try allocator.dupeZ(u8, content);
    defer allocator.free(content_z);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    const data = std.zon.parse.fromSliceAlloc(PersistedPluginState, arena_allocator, content_z, null, .{}) catch {
        return .loaded;
    };

    for (data.plugins) |entry| {
        try state.setEnabled(entry.name, entry.enabled);
    }

    return .loaded;
}

fn loadPluginState(allocator: std.mem.Allocator) !PluginState {
    var state = PluginState.init(allocator);
    errdefer state.deinit();

    const config_path = getPluginsConfigPath(allocator) catch return state;
    defer allocator.free(config_path);
    _ = try tryLoadPluginStateFromPath(allocator, config_path, &state);

    return state;
}

fn savePluginState(allocator: std.mem.Allocator, state: *const PluginState) !void {
    const config_path = try getPluginsConfigPath(allocator);
    defer allocator.free(config_path);

    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();

    var entries = std.ArrayListUnmanaged(PluginEntry).empty;
    defer entries.deinit(allocator);

    var iter = state.enabled_plugins.iterator();
    while (iter.next()) |entry| {
        try entries.append(allocator, .{ .name = entry.key_ptr.*, .enabled = entry.value_ptr.* });
    }

    const persisted = PersistedPluginState{ .plugins = entries.items };
    try std.zon.stringify.serialize(persisted, .{}, &out.writer);

    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    const dir_path = std.fs.path.dirname(config_path) orelse ".";
    try std.Io.Dir.cwd().createDirPath(io, dir_path);

    var file = try std.Io.Dir.cwd().createFile(io, config_path, .{ .truncate = true });
    defer file.close(io);
    try file.writeStreamingAll(io, try out.toOwnedSlice());
}

fn getPlugins(allocator: std.mem.Allocator) ![]PluginInfo {
    var state = try loadPluginState(allocator);
    defer state.deinit();

    var plugins = try allocator.alloc(PluginInfo, builtin_plugins.len);
    for (builtin_plugins, 0..) |plugin, i| {
        plugins[i] = plugin;
        plugins[i].enabled = state.isEnabled(plugin.name);
    }
    return plugins;
}

pub const meta: command_mod.Meta = .{
    .name = "plugins",
    .description = "Manage framework plugins (list, enable, disable, info)",
    .subcommands = &.{ "list", "enable", "disable", "info", "search", "help" },
};

pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    if (args.len == 0) {
        printHelp();
        return;
    }

    const cmd = std.mem.sliceTo(args[0], 0);
    if (utils.args.matchesAny(cmd, &.{ "help", "--help", "-h" })) {
        printHelp();
        return;
    }

    if (std.mem.eql(u8, cmd, "list")) {
        try runList(ctx);
    } else if (std.mem.eql(u8, cmd, "enable")) {
        try runEnable(ctx, args[1..]);
    } else if (std.mem.eql(u8, cmd, "disable")) {
        try runDisable(ctx, args[1..]);
    } else if (std.mem.eql(u8, cmd, "info")) {
        try runInfo(ctx, args[1..]);
    } else if (std.mem.eql(u8, cmd, "search")) {
        try runSearch(ctx, args[1..]);
    } else {
        utils.output.printError("Unknown plugins command: {s}", .{cmd});
        printHelp();
    }
}

fn runList(ctx: *const context_mod.CommandContext) !void {
    const plugins = try getPlugins(ctx.allocator);
    defer ctx.allocator.free(plugins);

    utils.output.printHeader("ABI Framework Plugins");
    utils.output.println("{s:<25} {s:<10} {s:<10} {s}", .{ "NAME", "VERSION", "STATUS", "DESCRIPTION" });
    utils.output.println("{s}", .{"-" ** 80});

    for (plugins) |p| {
        const status = if (p.enabled)
            try std.fmt.allocPrint(ctx.allocator, "{s}Enabled{s}", .{ utils.output.Color.green(), utils.output.Color.reset() })
        else
            try std.fmt.allocPrint(ctx.allocator, "{s}Disabled{s}", .{ utils.output.Color.dim(), utils.output.Color.reset() });
        defer ctx.allocator.free(status);

        utils.output.println("{s:<25} {s:<10} {s:<10} {s}", .{
            p.name, p.version, status, p.description,
        });
    }
}

fn runEnable(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    if (args.len == 0) {
        utils.output.printError("Plugin name required", .{});
        return;
    }

    const name = std.mem.sliceTo(args[0], 0);

    var state = try loadPluginState(ctx.allocator);
    defer state.deinit();

    var found = false;
    for (builtin_plugins) |p| {
        if (std.mem.eql(u8, p.name, name)) {
            found = true;
            break;
        }
    }

    if (!found) {
        utils.output.printError("Plugin not found: {s}", .{name});
        return;
    }

    if (state.isEnabled(name)) {
        utils.output.printInfo("Plugin '{s}' is already enabled.", .{name});
        return;
    }

    try state.setEnabled(name, true);
    try savePluginState(ctx.allocator, &state);
    utils.output.printSuccess("Enabled plugin: {s}", .{name});
}

fn runDisable(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    if (args.len == 0) {
        utils.output.printError("Plugin name required", .{});
        return;
    }

    const name = std.mem.sliceTo(args[0], 0);

    var state = try loadPluginState(ctx.allocator);
    defer state.deinit();

    if (!state.isEnabled(name)) {
        utils.output.printInfo("Plugin '{s}' is not enabled.", .{name});
        return;
    }

    try state.setEnabled(name, false);
    try savePluginState(ctx.allocator, &state);
    utils.output.printSuccess("Disabled plugin: {s}", .{name});
}

fn runInfo(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    if (args.len == 0) {
        utils.output.printError("Plugin name required", .{});
        return;
    }

    const name = std.mem.sliceTo(args[0], 0);

    var found_plugin: ?PluginInfo = null;
    for (builtin_plugins) |p| {
        if (std.mem.eql(u8, p.name, name)) {
            found_plugin = p;
            break;
        }
    }

    if (found_plugin) |p| {
        var state = try loadPluginState(ctx.allocator);
        defer state.deinit();

        utils.output.printHeader("Plugin Information");
        utils.output.printKeyValue("Name", p.name);
        utils.output.printKeyValue("Version", p.version);
        utils.output.printKeyValue("Status", if (state.isEnabled(p.name)) "Enabled" else "Disabled");
        utils.output.printKeyValue("Description", p.description);
    } else {
        utils.output.printError("Plugin not found: {s}", .{name});
    }
}

fn runSearch(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    if (args.len == 0) {
        utils.output.printInfo("Showing all plugins...", .{});
        return try runList(ctx);
    }

    const query = std.mem.sliceTo(args[0], 0);

    var state = try loadPluginState(ctx.allocator);
    defer state.deinit();

    utils.output.printHeader("Search Results");
    var found = false;
    for (builtin_plugins) |p| {
        if (std.ascii.indexOfIgnoreCase(p.name, query) != null or
            std.ascii.indexOfIgnoreCase(p.description, query) != null)
        {
            const status = if (state.isEnabled(p.name)) "[Enabled]" else "[Disabled]";
            utils.output.println("{s:<25} {s:<10} {s}", .{ p.name, status, p.description });
            found = true;
        }
    }

    if (!found) {
        utils.output.printInfo("No plugins matching '{s}' found.", .{query});
    }
}

fn printHelp() void {
    const help =
        "Usage: abi plugins <command> [args]\n\n" ++
        "Commands:\n" ++
        "  list                 List all available plugins\n" ++
        "  enable <name>        Enable a plugin\n" ++
        "  disable <name>       Disable a plugin\n" ++
        "  info <name>          Show detailed plugin information\n" ++
        "  search <query>       Search for plugins\n" ++
        "  help                 Show this help message\n\n" ++
        "Examples:\n" ++
        "  abi plugins list                     # Show all plugins\n" ++
        "  abi plugins enable discord-bot       # Enable Discord bot plugin\n" ++
        "  abi plugins search llm               # Search for LLM plugins\n";
    utils.output.print("{s}", .{help});
}

test {
    std.testing.refAllDecls(@This());
}
