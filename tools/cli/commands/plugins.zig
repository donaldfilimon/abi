//! Plugin management commands for ABI CLI.
//!
//! Manage framework plugins: list, enable, disable, info.
//! Plugin state is persisted to ~/.abi/plugins.json

const std = @import("std");
const utils = @import("../utils/mod.zig");
const cli_io = utils.io_backend;

// libc import for environment access - required for Zig 0.16
const c = @cImport(@cInclude("stdlib.h"));

// Plugin system types
const PluginInfo = struct {
    name: []const u8,
    description: []const u8,
    version: []const u8 = "1.0.0",
    author: []const u8 = "ABI Team",
    enabled: bool,
    dependencies: []const []const u8 = &.{},
    config_schema: ?[]const u8 = null,
};

// Built-in plugin definitions (static metadata)
const builtin_plugins = [_]PluginInfo{
    .{ .name = "openai-connector", .description = "OpenAI API integration", .enabled = true },
    .{ .name = "ollama-connector", .description = "Ollama local LLM integration", .enabled = true },
    .{ .name = "anthropic-connector", .description = "Anthropic Claude API integration", .enabled = true },
    .{ .name = "huggingface-connector", .description = "HuggingFace model hub", .enabled = true },
    .{ .name = "discord-bot", .description = "Discord bot framework", .enabled = false },
    .{ .name = "mistral-connector", .description = "Mistral AI API integration", .enabled = true },
    .{ .name = "cohere-connector", .description = "Cohere AI API integration", .enabled = true },
};

/// Plugin state persisted to disk
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
        var iter = self.enabled_plugins.keyIterator();
        while (iter.next()) |key| {
            self.allocator.free(key.*);
        }
        self.enabled_plugins.deinit(self.allocator);
    }

    pub fn isEnabled(self: *const PluginState, name: []const u8) bool {
        if (self.enabled_plugins.get(name)) |enabled| {
            return enabled;
        }
        // Default to built-in state
        for (builtin_plugins) |plugin| {
            if (std.mem.eql(u8, plugin.name, name)) return plugin.enabled;
        }
        return false;
    }

    pub fn setEnabled(self: *PluginState, name: []const u8, enabled: bool) !void {
        // Check if key already exists
        if (self.enabled_plugins.contains(name)) {
            self.enabled_plugins.getPtr(name).?.* = enabled;
        } else {
            const name_copy = try self.allocator.dupe(u8, name);
            errdefer self.allocator.free(name_copy);
            try self.enabled_plugins.put(self.allocator, name_copy, enabled);
        }
    }
};

/// Get path to plugins config file
fn getPluginsConfigPath(allocator: std.mem.Allocator) ![]const u8 {
    const home = getEnvOwned(allocator, "HOME") orelse
        getEnvOwned(allocator, "USERPROFILE") orelse
        return error.NoHomeDirectory;
    defer allocator.free(home);

    return std.fmt.allocPrint(allocator, "{s}/.abi/plugins.json", .{home});
}

/// Get environment variable (owned memory) - Zig 0.16 compatible.
fn getEnvOwned(allocator: std.mem.Allocator, name: []const u8) ?[]u8 {
    const name_z = allocator.dupeZ(u8, name) catch return null;
    defer allocator.free(name_z);

    const value_ptr = c.getenv(name_z.ptr);
    if (value_ptr) |ptr| {
        const value = std.mem.span(ptr);
        return allocator.dupe(u8, value) catch null;
    }
    return null;
}

/// Load plugin state from disk
fn loadPluginState(allocator: std.mem.Allocator) !PluginState {
    var state = PluginState.init(allocator);
    errdefer state.deinit();

    const config_path = getPluginsConfigPath(allocator) catch return state;
    defer allocator.free(config_path);

    // Initialize I/O backend for Zig 0.16
    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    const content = std.Io.Dir.cwd().readFileAlloc(io, config_path, allocator, .limited(1024 * 1024)) catch {
        return state; // File doesn't exist, return empty state
    };
    defer allocator.free(content);

    // Parse JSON
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, content, .{}) catch {
        return state; // Invalid JSON, return empty state
    };
    defer parsed.deinit();

    if (parsed.value == .object) {
        if (parsed.value.object.get("plugins")) |plugins_val| {
            if (plugins_val == .object) {
                var iter = plugins_val.object.iterator();
                while (iter.next()) |entry| {
                    if (entry.value_ptr.* == .bool) {
                        try state.setEnabled(entry.key_ptr.*, entry.value_ptr.*.bool);
                    }
                }
            }
        }
    }

    return state;
}

/// Save plugin state to disk
fn savePluginState(allocator: std.mem.Allocator, state: *const PluginState) !void {
    const config_path = try getPluginsConfigPath(allocator);
    defer allocator.free(config_path);

    // Build JSON content
    var json_buf = std.ArrayListUnmanaged(u8).empty;
    defer json_buf.deinit(allocator);

    try json_buf.appendSlice(allocator, "{\"plugins\":{");

    var first = true;
    var iter = state.enabled_plugins.iterator();
    while (iter.next()) |entry| {
        if (!first) try json_buf.appendSlice(allocator, ",");
        first = false;
        try json_buf.appendSlice(allocator, "\"");
        try json_buf.appendSlice(allocator, entry.key_ptr.*);
        try json_buf.appendSlice(allocator, "\":");
        try json_buf.appendSlice(allocator, if (entry.value_ptr.*) "true" else "false");
    }

    try json_buf.appendSlice(allocator, "}}\n");

    // Initialize I/O backend for Zig 0.16
    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    // Ensure directory exists by trying to open/create it
    const home = getEnvOwned(allocator, "HOME") orelse
        getEnvOwned(allocator, "USERPROFILE") orelse
        return error.NoHomeDirectory;
    defer allocator.free(home);

    const dir_path = try std.fmt.allocPrint(allocator, "{s}/.abi", .{home});
    defer allocator.free(dir_path);

    // Try to create the directory (using openDir to check if it exists first)
    _ = std.Io.Dir.cwd().openDir(io, dir_path, .{}) catch {
        // Directory doesn't exist, try to create parent file to trigger error
        // or use platform-specific mkdir
        const builtin = @import("builtin");
        if (comptime builtin.os.tag == .windows) {
            const kernel32 = struct {
                extern "kernel32" fn CreateDirectoryA(
                    lpPathName: [*:0]const u8,
                    lpSecurityAttributes: ?*anyopaque,
                ) callconv(.winapi) i32;
            };
            const dir_z = allocator.dupeZ(u8, dir_path) catch return error.OutOfMemory;
            defer allocator.free(dir_z);
            _ = kernel32.CreateDirectoryA(dir_z.ptr, null);
        } else {
            const posix = struct {
                extern "c" fn mkdir(path: [*:0]const u8, mode: u32) c_int;
            };
            const dir_z = allocator.dupeZ(u8, dir_path) catch return error.OutOfMemory;
            defer allocator.free(dir_z);
            _ = posix.mkdir(dir_z.ptr, 0o755);
        }
    };

    // Write file
    var file = try std.Io.Dir.cwd().createFile(io, config_path, .{ .truncate = true });
    defer file.close(io);
    try file.writeStreamingAll(io, json_buf.items);
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

fn getPlugin(allocator: std.mem.Allocator, name: []const u8) !?PluginInfo {
    var state = try loadPluginState(allocator);
    defer state.deinit();

    for (builtin_plugins) |plugin| {
        if (std.mem.eql(u8, plugin.name, name)) {
            var result = plugin;
            result.enabled = state.isEnabled(plugin.name);
            return result;
        }
    }
    return null;
}

/// Entry point for the plugins command.
fn plList(alloc: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    _ = parser;
    try listPlugins(alloc);
}
fn plInfo(alloc: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    const name = parser.next() orelse {
        utils.output.printError("Usage: abi plugins info <name>", .{});
        return;
    };
    try showPluginInfo(alloc, name);
}
fn plEnable(alloc: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    const name = parser.next() orelse {
        utils.output.printError("Usage: abi plugins enable <name>", .{});
        return;
    };
    try enablePlugin(alloc, name);
}
fn plDisable(alloc: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    const name = parser.next() orelse {
        utils.output.printError("Usage: abi plugins disable <name>", .{});
        return;
    };
    try disablePlugin(alloc, name);
}
fn plSearch(alloc: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    const query = parser.next() orelse "";
    try searchPlugins(alloc, query);
}
fn plUnknown(cmd: []const u8) void {
    utils.output.printError("Unknown subcommand: {s}", .{cmd});
}
fn printHelpAlloc(_: std.mem.Allocator) void {
    printHelp();
}

const plugin_commands = [_]utils.subcommand.Command{
    .{ .names = &.{"list"}, .run = plList },
    .{ .names = &.{"info"}, .run = plInfo },
    .{ .names = &.{"enable"}, .run = plEnable },
    .{ .names = &.{"disable"}, .run = plDisable },
    .{ .names = &.{"search"}, .run = plSearch },
};

pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var parser = utils.args.ArgParser.init(allocator, args);
    try utils.subcommand.runSubcommand(
        allocator,
        &parser,
        &plugin_commands,
        plList,
        printHelpAlloc,
        plUnknown,
    );
}

fn listPlugins(allocator: std.mem.Allocator) !void {
    utils.output.printHeader("Installed Plugins");

    // Get plugins from local registry with persisted state
    const plugins = try getPlugins(allocator);
    defer allocator.free(plugins);

    if (plugins.len == 0) {
        utils.output.printInfo("No plugins installed", .{});
        utils.output.printInfo("Use 'abi plugins search' to find plugins", .{});
        return;
    }

    std.debug.print("\n{s:<25} {s:<10} {s:<40}\n", .{ "NAME", "STATUS", "DESCRIPTION" });
    std.debug.print("{s}\n", .{"-" ** 75});

    var enabled_count: usize = 0;
    for (plugins) |plugin| {
        const status_str = if (plugin.enabled) "enabled" else "disabled";
        if (plugin.enabled) enabled_count += 1;
        std.debug.print("{s:<25} {s:<10} {s:<40}\n", .{
            plugin.name,
            status_str,
            truncate(plugin.description, 40),
        });
    }

    std.debug.print("\nTotal: {d} plugin(s), {d} enabled\n", .{ plugins.len, enabled_count });
    std.debug.print("Config: ~/.abi/plugins.json\n", .{});
}

fn showPluginInfo(allocator: std.mem.Allocator, name: []const u8) !void {
    const plugin_opt = try getPlugin(allocator, name);

    if (plugin_opt) |plugin| {
        utils.output.printHeader("Plugin Information");
        std.debug.print("\n", .{});
        std.debug.print("Name:        {s}\n", .{plugin.name});
        std.debug.print("Version:     {s}\n", .{plugin.version});
        std.debug.print("Description: {s}\n", .{plugin.description});
        std.debug.print("Author:      {s}\n", .{plugin.author});
        std.debug.print("Status:      {s}\n", .{if (plugin.enabled) "Enabled" else "Disabled"});

        if (plugin.dependencies.len > 0) {
            std.debug.print("Dependencies:\n", .{});
            for (plugin.dependencies) |dep| {
                std.debug.print("  - {s}\n", .{dep});
            }
        }

        if (plugin.config_schema) |schema| {
            std.debug.print("\nConfiguration Schema:\n", .{});
            std.debug.print("{s}\n", .{schema});
        }

        // Show environment variable requirements
        std.debug.print("\nEnvironment Variables:\n", .{});
        if (std.mem.eql(u8, plugin.name, "openai-connector")) {
            std.debug.print("  ABI_OPENAI_API_KEY - OpenAI API key\n", .{});
        } else if (std.mem.eql(u8, plugin.name, "anthropic-connector")) {
            std.debug.print("  ABI_ANTHROPIC_API_KEY - Anthropic API key\n", .{});
        } else if (std.mem.eql(u8, plugin.name, "huggingface-connector")) {
            std.debug.print("  ABI_HF_API_TOKEN - HuggingFace token\n", .{});
        } else if (std.mem.eql(u8, plugin.name, "ollama-connector")) {
            std.debug.print("  ABI_OLLAMA_HOST - Ollama host (default: http://127.0.0.1:11434)\n", .{});
        } else if (std.mem.eql(u8, plugin.name, "discord-bot")) {
            std.debug.print("  DISCORD_BOT_TOKEN - Discord bot token\n", .{});
        } else if (std.mem.eql(u8, plugin.name, "mistral-connector")) {
            std.debug.print("  MISTRAL_API_KEY - Mistral API key\n", .{});
        } else if (std.mem.eql(u8, plugin.name, "cohere-connector")) {
            std.debug.print("  COHERE_API_KEY - Cohere API key\n", .{});
        } else {
            std.debug.print("  (none required)\n", .{});
        }
    } else {
        utils.output.printError("Plugin not found: {s}", .{name});
        utils.output.printInfo("Use 'abi plugins list' to see installed plugins", .{});
    }
}

fn enablePlugin(allocator: std.mem.Allocator, name: []const u8) !void {
    // Check if plugin exists
    var plugin_exists = false;
    for (builtin_plugins) |plugin| {
        if (std.mem.eql(u8, plugin.name, name)) {
            plugin_exists = true;
            break;
        }
    }

    if (!plugin_exists) {
        utils.output.printError("Plugin not found: {s}", .{name});
        utils.output.printInfo("Use 'abi plugins search' to find available plugins", .{});
        return;
    }

    // Load current state
    var state = try loadPluginState(allocator);
    defer state.deinit();

    // Check if already enabled
    if (state.isEnabled(name)) {
        utils.output.printInfo("Plugin '{s}' is already enabled", .{name});
        return;
    }

    // Enable the plugin
    try state.setEnabled(name, true);

    // Persist to disk
    try savePluginState(allocator, &state);

    utils.output.printSuccess("Plugin '{s}' enabled", .{name});
    utils.output.printInfo("Changes saved to ~/.abi/plugins.json", .{});
}

fn disablePlugin(allocator: std.mem.Allocator, name: []const u8) !void {
    // Check if plugin exists
    var plugin_exists = false;
    for (builtin_plugins) |plugin| {
        if (std.mem.eql(u8, plugin.name, name)) {
            plugin_exists = true;
            break;
        }
    }

    if (!plugin_exists) {
        utils.output.printError("Plugin not found: {s}", .{name});
        utils.output.printInfo("Use 'abi plugins list' to see installed plugins", .{});
        return;
    }

    // Load current state
    var state = try loadPluginState(allocator);
    defer state.deinit();

    // Check if already disabled
    if (!state.isEnabled(name)) {
        utils.output.printInfo("Plugin '{s}' is already disabled", .{name});
        return;
    }

    // Disable the plugin
    try state.setEnabled(name, false);

    // Persist to disk
    try savePluginState(allocator, &state);

    utils.output.printSuccess("Plugin '{s}' disabled", .{name});
    utils.output.printInfo("Changes saved to ~/.abi/plugins.json", .{});
}

fn searchPlugins(allocator: std.mem.Allocator, query: []const u8) !void {
    utils.output.printHeader("Available Plugins");

    // Load current state to show enabled status
    var state = try loadPluginState(allocator);
    defer state.deinit();

    // Show all available plugins (installed and potential future plugins)
    const available = [_]struct {
        name: []const u8,
        description: []const u8,
        installed: bool,
    }{
        .{ .name = "openai-connector", .description = "OpenAI API integration", .installed = true },
        .{ .name = "ollama-connector", .description = "Ollama local LLM integration", .installed = true },
        .{ .name = "anthropic-connector", .description = "Anthropic Claude API integration", .installed = true },
        .{ .name = "huggingface-connector", .description = "HuggingFace model hub", .installed = true },
        .{ .name = "discord-bot", .description = "Discord bot framework", .installed = true },
        .{ .name = "mistral-connector", .description = "Mistral AI API integration", .installed = true },
        .{ .name = "cohere-connector", .description = "Cohere AI API integration", .installed = true },
        .{ .name = "metrics-prometheus", .description = "Prometheus metrics exporter", .installed = false },
        .{ .name = "storage-s3", .description = "AWS S3 storage backend", .installed = false },
        .{ .name = "storage-gcs", .description = "Google Cloud Storage backend", .installed = false },
        .{ .name = "cache-redis", .description = "Redis caching layer", .installed = false },
    };

    std.debug.print("\n{s:<25} {s:<12} {s:<10} {s:<30}\n", .{ "NAME", "STATUS", "ENABLED", "DESCRIPTION" });
    std.debug.print("{s}\n", .{"-" ** 77});

    var count: usize = 0;
    for (available) |plugin| {
        // Filter by query if provided
        if (query.len > 0) {
            if (std.mem.indexOf(u8, plugin.name, query) == null and
                std.mem.indexOf(u8, plugin.description, query) == null)
            {
                continue;
            }
        }

        const status = if (plugin.installed) "installed" else "available";
        const enabled = if (plugin.installed) (if (state.isEnabled(plugin.name)) "yes" else "no") else "-";
        std.debug.print("{s:<25} {s:<12} {s:<10} {s:<30}\n", .{
            plugin.name,
            status,
            enabled,
            truncate(plugin.description, 30),
        });
        count += 1;
    }

    if (count == 0) {
        utils.output.printInfo("No plugins match query: {s}", .{query});
    } else {
        std.debug.print("\nFound: {d} plugin(s)\n", .{count});
        std.debug.print("\nUse 'abi plugins enable <name>' to enable a plugin\n", .{});
        std.debug.print("Use 'abi plugins disable <name>' to disable a plugin\n", .{});
    }
}

fn truncate(s: []const u8, max_len: usize) []const u8 {
    if (s.len <= max_len) return s;
    return s[0..max_len];
}

fn printHelp() void {
    const help =
        \\Usage: abi plugins <subcommand> [options]
        \\
        \\Manage ABI framework plugins.
        \\
        \\Subcommands:
        \\  list                  List installed plugins (default)
        \\  info <name>           Show detailed plugin information
        \\  enable <name>         Enable a plugin
        \\  disable <name>        Disable a plugin
        \\  search [query]        Search available plugins
        \\  help                  Show this help
        \\
        \\Examples:
        \\  abi plugins                   # List installed plugins
        \\  abi plugins info openai       # Show OpenAI connector info
        \\  abi plugins enable redis      # Enable Redis plugin
        \\  abi plugins search storage    # Search for storage plugins
        \\
    ;
    std.debug.print("{s}", .{help});
}
