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
            .enabled_plugins = . {},
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

const PersistedPluginState = struct {
    plugins: std.StringArrayHashMapUnmanaged(bool) = . {},
};

fn getPluginsConfigPath(allocator: std.mem.Allocator) ![]u8 {
    return app_paths.resolvePath(allocator, "plugins.zon");
}

fn tryLoadPluginStateFromPath(
    allocator: std.mem.Allocator,
    path: []const u8,
    state: *PluginState,
) !PluginLoadState {
    // Initialize I/O backend for Zig 0.16
    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    const content = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1024 * 1024)) catch |err| switch (err) {
        error.FileNotFound => return .missing,
        else => return err,
    };
    defer allocator.free(content);

    // Ensure null-terminated for ZON
    const content_z = try allocator.dupeZ(u8, content);
    defer allocator.free(content_z);

    const parsed = std.zon.parse.fromSlice(PersistedPluginState, allocator, content_z, null, .{}) catch {
        return .loaded;
    };
    defer parsed.deinit();

    var iter = parsed.value.plugins.iterator();
    while (iter.next()) |entry| {
        try state.setEnabled(entry.key_ptr.*, entry.value_ptr.*);
    }

    return .loaded;
}

/// Load plugin state from disk
fn loadPluginState(allocator: std.mem.Allocator) !PluginState {
    var state = PluginState.init(allocator);
    errdefer state.deinit();

    const config_path = getPluginsConfigPath(allocator) catch return state;
    defer allocator.free(config_path);
    _ = try tryLoadPluginStateFromPath(allocator, config_path, &state);

    return state;
}

/// Save plugin state to disk
fn savePluginState(allocator: std.mem.Allocator, state: *const PluginState) !void {
    const config_path = try getPluginsConfigPath(allocator);
    defer allocator.free(config_path);

    // Build ZON content
    var zon_buf = std.ArrayList(u8).init(allocator);
    defer zon_buf.deinit();

    var persisted = PersistedPluginState{};
    var iter = state.enabled_plugins.iterator();
    while (iter.next()) |entry| {
        try persisted.plugins.put(allocator, entry.key_ptr.*, entry.value_ptr.*);
    }
    defer persisted.plugins.deinit(allocator);

    var writer = zon_buf.writer();
    try std.zon.stringify.serialize(persisted, .{}, &writer);

    // Initialize I/O backend for Zig 0.16
    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    const dir_path = std.fs.path.dirname(config_path) orelse ".";
    try std.Io.Dir.cwd().createDirPath(io, dir_path);

    // Write file
    var file = try std.Io.Dir.cwd().createFile(io, config_path, .{ .truncate = true });
    defer file.close(io);
    try file.writeStreamingAll(io, zon_buf.items);
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
    const allocator = ctx.allocator;
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
    const allocator = ctx.allocator;
    const plugins = try getPlugins(allocator);
    defer allocator.free(plugins);

    utils.output.printHeader("ABI Framework Plugins");
    utils.output.println("{s:<25} {s:<10} {s:<10} {s}", .{ "NAME", "VERSION", "STATUS", "DESCRIPTION" });
    utils.output.println("{s}", .{"-" ** 80});

    for (plugins) |p| {
        const status = if (p.enabled)
            utils.output.colorText("Enabled", .green)
        else
            utils.output.colorText("Disabled", .dim);

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

    const allocator = ctx.allocator;
    const name = std.mem.sliceTo(args[0], 0);

    var state = try loadPluginState(allocator);
    defer state.deinit();

    // Verify plugin exists
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
    try savePluginState(allocator, &state);
    utils.output.printSuccess("Enabled plugin: {s}", .{name});
}

fn runDisable(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    if (args.len == 0) {
        utils.output.printError("Plugin name required", .{});
        return;
    }

    const allocator = ctx.allocator;
    const name = std.mem.sliceTo(args[0], 0);

    var state = try loadPluginState(allocator);
    defer state.deinit();

    if (!state.isEnabled(name)) {
        utils.output.printInfo("Plugin '{s}' is not enabled.", .{name});
        return;
    }

    try state.setEnabled(name, false);
    try savePluginState(allocator, &state);
    utils.output.printSuccess("Disabled plugin: {s}", .{name});
}

fn runInfo(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    if (args.len == 0) {
        utils.output.printError("Plugin name required", .{});
        return;
    }

    const allocator = ctx.allocator;
    const name = std.mem.sliceTo(args[0], 0);

    var found_plugin: ?PluginInfo = null;
    for (builtin_plugins) |p| {
        if (std.mem.eql(u8, p.name, name)) {
            found_plugin = p;
            break;
        }
    }

    if (found_plugin) |p| {
        var state = try loadPluginState(allocator);
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

    const allocator = ctx.allocator;
    const query = std.mem.sliceTo(args[0], 0);

    const state = try loadPluginState(allocator);
    // Note: state is only for isEnabled check, doesn't need deinit here if we don't use it much
    // but better safe.
    var state_local = state;
    defer state_local.deinit();

    utils.output.printHeader("Search Results");
    var found = false;
    for (builtin_plugins) |p| {
        if (std.ascii.indexOfIgnoreCase(p.name, query) != null or
            std.ascii.indexOfIgnoreCase(p.description, query) != null) {
            const status = if (state_local.isEnabled(p.name)) "[Enabled]" else "[Disabled]";
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
        \\Usage: abi plugins <command> [args]
        \\
        \\Commands:
        \\  list                 List all available plugins
        \\  enable <name>        Enable a plugin
        \\  disable <name>       Disable a plugin
        \\  info <name>          Show detailed plugin information
        \\  search <query>       Search for plugins
        \\  help                 Show this help message
        \\
        \\Examples:
        \\  abi plugins list                     # Show all plugins
        \\  abi plugins enable discord-bot       # Enable Discord bot plugin
        \\  abi plugins search llm               # Search for LLM plugins
        \\
    ;
    utils.output.print("{s}", .{help});
}

test {
    std.testing.refAllDecls(@This());
}
