//! Plugin management commands for ABI CLI.
//!
//! Manage framework plugins: list, enable, disable, info.

const std = @import("std");
const utils = @import("../utils/mod.zig");

// Plugin system types (local stubs until full plugin registry is implemented)
const PluginInfo = struct {
    name: []const u8,
    description: []const u8,
    version: []const u8 = "1.0.0",
    author: []const u8 = "ABI Team",
    enabled: bool,
    dependencies: []const []const u8 = &.{},
    config_schema: ?[]const u8 = null,
};

// Built-in plugin registry (static for now)
const builtin_plugins = [_]PluginInfo{
    .{ .name = "openai-connector", .description = "OpenAI API integration", .enabled = true },
    .{ .name = "ollama-connector", .description = "Ollama local LLM integration", .enabled = true },
    .{ .name = "anthropic-connector", .description = "Anthropic Claude API integration", .enabled = true },
    .{ .name = "huggingface-connector", .description = "HuggingFace model hub", .enabled = true },
    .{ .name = "discord-bot", .description = "Discord bot framework", .enabled = false },
};

fn getPlugins() []const PluginInfo {
    return &builtin_plugins;
}

fn getPlugin(name: []const u8) ?PluginInfo {
    for (builtin_plugins) |plugin| {
        if (std.mem.eql(u8, plugin.name, name)) return plugin;
    }
    return null;
}

/// Entry point for the plugins command.
pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var parser = utils.args.ArgParser.init(allocator, args);

    if (parser.wantsHelp()) {
        printHelp();
        return;
    }

    const subcommand = parser.next() orelse {
        try listPlugins(allocator);
        return;
    };

    if (std.mem.eql(u8, subcommand, "list")) {
        try listPlugins(allocator);
    } else if (std.mem.eql(u8, subcommand, "info")) {
        const plugin_name = parser.next() orelse {
            utils.output.printError("Usage: abi plugins info <name>", .{});
            return;
        };
        try showPluginInfo(allocator, plugin_name);
    } else if (std.mem.eql(u8, subcommand, "enable")) {
        const plugin_name = parser.next() orelse {
            utils.output.printError("Usage: abi plugins enable <name>", .{});
            return;
        };
        try enablePlugin(allocator, plugin_name);
    } else if (std.mem.eql(u8, subcommand, "disable")) {
        const plugin_name = parser.next() orelse {
            utils.output.printError("Usage: abi plugins disable <name>", .{});
            return;
        };
        try disablePlugin(allocator, plugin_name);
    } else if (std.mem.eql(u8, subcommand, "search")) {
        const query = parser.next() orelse "";
        try searchPlugins(allocator, query);
    } else if (std.mem.eql(u8, subcommand, "help")) {
        printHelp();
    } else {
        utils.output.printError("Unknown subcommand: {s}", .{subcommand});
        printHelp();
    }
}

fn listPlugins(allocator: std.mem.Allocator) !void {
    _ = allocator;
    utils.output.printHeader("Installed Plugins");

    // Get plugins from local registry
    const plugins = getPlugins();

    if (plugins.len == 0) {
        utils.output.printInfo("No plugins installed", .{});
        utils.output.printInfo("Use 'abi plugins search' to find plugins", .{});
        return;
    }

    std.debug.print("\n{s:<20} {s:<10} {s:<40}\n", .{ "NAME", "STATUS", "DESCRIPTION" });
    std.debug.print("{s}\n", .{"-" ** 70});

    for (plugins) |plugin| {
        const status_str = if (plugin.enabled) "enabled" else "disabled";
        std.debug.print("{s:<20} {s:<10} {s:<40}\n", .{
            plugin.name,
            status_str,
            truncate(plugin.description, 40),
        });
    }

    std.debug.print("\nTotal: {d} plugin(s)\n", .{plugins.len});
}

fn showPluginInfo(allocator: std.mem.Allocator, name: []const u8) !void {
    _ = allocator;
    const plugin_opt = getPlugin(name);

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
    } else {
        utils.output.printError("Plugin not found: {s}", .{name});
        utils.output.printInfo("Use 'abi plugins list' to see installed plugins", .{});
    }
}

fn enablePlugin(allocator: std.mem.Allocator, name: []const u8) !void {
    _ = allocator;
    // Plugin enable/disable is not implemented yet (static registry)
    if (getPlugin(name) != null) {
        utils.output.printInfo("Plugin '{s}' found but runtime toggling not yet implemented", .{name});
        utils.output.printInfo("Plugins are configured at compile-time via build options", .{});
    } else {
        utils.output.printError("Plugin not found: {s}", .{name});
        utils.output.printInfo("Use 'abi plugins search' to find available plugins", .{});
    }
}

fn disablePlugin(allocator: std.mem.Allocator, name: []const u8) !void {
    _ = allocator;
    // Plugin enable/disable is not implemented yet (static registry)
    if (getPlugin(name) != null) {
        utils.output.printInfo("Plugin '{s}' found but runtime toggling not yet implemented", .{name});
        utils.output.printInfo("Plugins are configured at compile-time via build options", .{});
    } else {
        utils.output.printError("Plugin not found: {s}", .{name});
        utils.output.printInfo("Use 'abi plugins list' to see installed plugins", .{});
    }
}

fn searchPlugins(allocator: std.mem.Allocator, query: []const u8) !void {
    _ = allocator;
    utils.output.printHeader("Available Plugins");

    // Show built-in/available plugins
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
        .{ .name = "metrics-prometheus", .description = "Prometheus metrics exporter", .installed = false },
        .{ .name = "storage-s3", .description = "AWS S3 storage backend", .installed = false },
        .{ .name = "storage-gcs", .description = "Google Cloud Storage backend", .installed = false },
        .{ .name = "cache-redis", .description = "Redis caching layer", .installed = false },
    };

    std.debug.print("\n{s:<25} {s:<10} {s:<40}\n", .{ "NAME", "STATUS", "DESCRIPTION" });
    std.debug.print("{s}\n", .{"-" ** 75});

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
        std.debug.print("{s:<25} {s:<10} {s:<40}\n", .{
            plugin.name,
            status,
            truncate(plugin.description, 40),
        });
        count += 1;
    }

    if (count == 0) {
        utils.output.printInfo("No plugins match query: {s}", .{query});
    } else {
        std.debug.print("\nFound: {d} plugin(s)\n", .{count});
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
