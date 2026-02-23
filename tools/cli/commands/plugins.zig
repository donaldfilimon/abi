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
const app_paths = abi.shared.app_paths;

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

// Built-in plugin definitions (static metadata — all 15 LLM providers + discord + scheduler)
const builtin_plugins = [_]PluginInfo{
    .{ .name = "openai-connector", .description = "OpenAI API integration", .enabled = true },
    .{ .name = "anthropic-connector", .description = "Anthropic Claude API integration", .enabled = true },
    .{ .name = "claude-connector", .description = "Claude with Anthropic env fallbacks", .enabled = true },
    .{ .name = "ollama-connector", .description = "Ollama local LLM integration", .enabled = true },
    .{ .name = "ollama-passthrough", .description = "Ollama OpenAI-compatible passthrough", .enabled = true },
    .{ .name = "huggingface-connector", .description = "HuggingFace model hub", .enabled = true },
    .{ .name = "mistral-connector", .description = "Mistral AI API integration", .enabled = true },
    .{ .name = "cohere-connector", .description = "Cohere AI API integration", .enabled = true },
    .{ .name = "gemini-connector", .description = "Google Gemini API integration", .enabled = true },
    .{ .name = "codex-connector", .description = "Codex (OpenAI-compatible) integration", .enabled = true },
    .{ .name = "opencode-connector", .description = "OpenCode (OpenAI-compatible) integration", .enabled = true },
    .{ .name = "lm-studio-connector", .description = "LM Studio local server integration", .enabled = true },
    .{ .name = "vllm-connector", .description = "vLLM inference server integration", .enabled = true },
    .{ .name = "mlx-connector", .description = "MLX Apple Silicon inference", .enabled = true },
    .{ .name = "llama-cpp-connector", .description = "llama.cpp local server integration", .enabled = true },
    .{ .name = "discord-bot", .description = "Discord bot framework", .enabled = false },
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
fn getPluginsConfigPath(allocator: std.mem.Allocator) ![]u8 {
    return app_paths.resolvePath(allocator, "plugins.json");
}

fn printPluginsConfigLocation(allocator: std.mem.Allocator) void {
    const config_path = getPluginsConfigPath(allocator) catch {
        utils.output.printKeyValue("Config", "(unavailable)");
        return;
    };
    defer allocator.free(config_path);
    utils.output.printKeyValue("Config", config_path);
}

fn printPluginsSavedPath(allocator: std.mem.Allocator) void {
    const config_path = getPluginsConfigPath(allocator) catch {
        utils.output.printInfo("Changes saved to plugin settings", .{});
        return;
    };
    defer allocator.free(config_path);
    utils.output.printInfo("Changes saved to {s}", .{config_path});
}

const PluginLoadState = enum {
    loaded,
    missing,
};

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

    // Preserve historical behavior: invalid JSON yields an empty plugin state.
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, content, .{}) catch {
        return .loaded;
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

    return .loaded;
}

/// Load plugin state from disk
fn loadPluginState(allocator: std.mem.Allocator) !PluginState {
    var state = PluginState.init(allocator);
    errdefer state.deinit();

    const config_path = app_paths.resolvePath(allocator, "plugins.json") catch return state;
    defer allocator.free(config_path);
    _ = try tryLoadPluginStateFromPath(allocator, config_path, &state);

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

    const dir_path = std.fs.path.dirname(config_path) orelse ".";
    try std.Io.Dir.cwd().createDirPath(io, dir_path);

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

// Wrapper functions for comptime children dispatch
fn wrapPlList(ctx: *const context_mod.CommandContext, _: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    try listPlugins(allocator);
}
fn wrapPlInfo(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (args.len == 0) {
        utils.output.printError("Usage: abi plugins info <name>", .{});
        return;
    }
    try showPluginInfo(allocator, std.mem.sliceTo(args[0], 0));
}
fn wrapPlEnable(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (args.len == 0) {
        utils.output.printError("Usage: abi plugins enable <name>", .{});
        return;
    }
    try enablePlugin(allocator, std.mem.sliceTo(args[0], 0));
}
fn wrapPlDisable(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (args.len == 0) {
        utils.output.printError("Usage: abi plugins disable <name>", .{});
        return;
    }
    try disablePlugin(allocator, std.mem.sliceTo(args[0], 0));
}
fn wrapPlSearch(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    const query = if (args.len > 0) std.mem.sliceTo(args[0], 0) else "";
    try searchPlugins(allocator, query);
}

pub const meta: command_mod.Meta = .{
    .name = "plugins",
    .description = "Plugin management (list, enable, disable, info)",
    .subcommands = &.{ "list", "info", "enable", "disable", "search" },
    .children = &.{
        .{ .name = "list", .description = "List installed plugins", .handler = wrapPlList },
        .{ .name = "info", .description = "Show detailed plugin information", .handler = wrapPlInfo },
        .{ .name = "enable", .description = "Enable a plugin", .handler = wrapPlEnable },
        .{ .name = "disable", .description = "Disable a plugin", .handler = wrapPlDisable },
        .{ .name = "search", .description = "Search available plugins", .handler = wrapPlSearch },
    },
};

const plugin_subcommands = [_][]const u8{
    "list", "info", "enable", "disable", "search", "help",
};

pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (args.len == 0) {
        // Default action: list plugins
        try listPlugins(allocator);
        return;
    }
    const cmd = std.mem.sliceTo(args[0], 0);
    if (utils.args.matchesAny(cmd, &.{ "--help", "-h", "help" })) {
        printHelp();
        return;
    }
    // Unknown subcommand
    utils.output.printError("Unknown subcommand: {s}", .{cmd});
    if (utils.args.suggestCommand(cmd, &plugin_subcommands)) |suggestion| {
        utils.output.printInfo("Did you mean: {s}", .{suggestion});
    }
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

    utils.output.println("\n{s:<25} {s:<10} {s:<40}", .{ "NAME", "STATUS", "DESCRIPTION" });
    utils.output.printSeparator(75);

    var enabled_count: usize = 0;
    for (plugins) |plugin| {
        const status_str = if (plugin.enabled) "enabled" else "disabled";
        if (plugin.enabled) enabled_count += 1;
        utils.output.println("{s:<25} {s:<10} {s:<40}", .{
            plugin.name,
            status_str,
            truncate(plugin.description, 40),
        });
    }

    utils.output.println("\nTotal: {d} plugin(s), {d} enabled", .{ plugins.len, enabled_count });
    printPluginsConfigLocation(allocator);
}

fn showPluginInfo(allocator: std.mem.Allocator, name: []const u8) !void {
    const plugin_opt = try getPlugin(allocator, name);

    if (plugin_opt) |plugin| {
        utils.output.printHeader("Plugin Information");
        utils.output.println("", .{});
        utils.output.printKeyValue("Name", plugin.name);
        utils.output.printKeyValue("Version", plugin.version);
        utils.output.printKeyValue("Description", plugin.description);
        utils.output.printKeyValue("Author", plugin.author);
        utils.output.printKeyValue("Status", if (plugin.enabled) "Enabled" else "Disabled");

        if (plugin.dependencies.len > 0) {
            utils.output.println("  Dependencies:", .{});
            for (plugin.dependencies) |dep| {
                utils.output.println("    - {s}", .{dep});
            }
        }

        if (plugin.config_schema) |schema| {
            utils.output.println("\n  Configuration Schema:", .{});
            utils.output.println("  {s}", .{schema});
        }

        // Show environment variable requirements (ABI_* convention)
        utils.output.println("\n  Environment Variables:", .{});
        if (std.mem.eql(u8, plugin.name, "openai-connector")) {
            utils.output.println("    ABI_OPENAI_API_KEY - OpenAI API key", .{});
        } else if (std.mem.eql(u8, plugin.name, "anthropic-connector")) {
            utils.output.println("    ABI_ANTHROPIC_API_KEY - Anthropic API key", .{});
        } else if (std.mem.eql(u8, plugin.name, "claude-connector")) {
            utils.output.println("    ABI_CLAUDE_API_KEY (falls back to ABI_ANTHROPIC_API_KEY)", .{});
        } else if (std.mem.eql(u8, plugin.name, "huggingface-connector")) {
            utils.output.println("    ABI_HF_API_TOKEN - HuggingFace token", .{});
        } else if (std.mem.eql(u8, plugin.name, "ollama-connector")) {
            utils.output.println("    ABI_OLLAMA_HOST - Ollama host (default: http://127.0.0.1:11434)", .{});
        } else if (std.mem.eql(u8, plugin.name, "ollama-passthrough")) {
            utils.output.println("    ABI_OLLAMA_PASSTHROUGH_URL - Ollama passthrough URL", .{});
        } else if (std.mem.eql(u8, plugin.name, "discord-bot")) {
            utils.output.println("    DISCORD_BOT_TOKEN - Discord bot token", .{});
        } else if (std.mem.eql(u8, plugin.name, "mistral-connector")) {
            utils.output.println("    ABI_MISTRAL_API_KEY - Mistral API key", .{});
        } else if (std.mem.eql(u8, plugin.name, "cohere-connector")) {
            utils.output.println("    ABI_COHERE_API_KEY - Cohere API key", .{});
        } else if (std.mem.eql(u8, plugin.name, "gemini-connector")) {
            utils.output.println("    ABI_GEMINI_API_KEY - Google Gemini API key", .{});
        } else if (std.mem.eql(u8, plugin.name, "codex-connector")) {
            utils.output.println("    ABI_CODEX_API_KEY (falls back to ABI_OPENAI_API_KEY)", .{});
        } else if (std.mem.eql(u8, plugin.name, "opencode-connector")) {
            utils.output.println("    ABI_OPENCODE_API_KEY (falls back to ABI_OPENAI_API_KEY)", .{});
        } else if (std.mem.eql(u8, plugin.name, "lm-studio-connector")) {
            utils.output.println("    ABI_LM_STUDIO_HOST - LM Studio host (default: http://localhost:1234)", .{});
        } else if (std.mem.eql(u8, plugin.name, "vllm-connector")) {
            utils.output.println("    ABI_VLLM_HOST - vLLM server host", .{});
        } else if (std.mem.eql(u8, plugin.name, "mlx-connector")) {
            utils.output.println("    ABI_MLX_HOST - MLX server host", .{});
        } else if (std.mem.eql(u8, plugin.name, "llama-cpp-connector")) {
            utils.output.println("    ABI_LLAMA_CPP_HOST - llama.cpp host (default: http://localhost:8080)", .{});
        } else {
            utils.output.println("    (none required)", .{});
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
    printPluginsSavedPath(allocator);
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
    printPluginsSavedPath(allocator);
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
        .{ .name = "anthropic-connector", .description = "Anthropic Claude API integration", .installed = true },
        .{ .name = "claude-connector", .description = "Claude with Anthropic env fallbacks", .installed = true },
        .{ .name = "ollama-connector", .description = "Ollama local LLM integration", .installed = true },
        .{ .name = "ollama-passthrough", .description = "Ollama OpenAI-compatible passthrough", .installed = true },
        .{ .name = "huggingface-connector", .description = "HuggingFace model hub", .installed = true },
        .{ .name = "mistral-connector", .description = "Mistral AI API integration", .installed = true },
        .{ .name = "cohere-connector", .description = "Cohere AI API integration", .installed = true },
        .{ .name = "gemini-connector", .description = "Google Gemini API integration", .installed = true },
        .{ .name = "codex-connector", .description = "Codex (OpenAI-compatible)", .installed = true },
        .{ .name = "opencode-connector", .description = "OpenCode (OpenAI-compatible)", .installed = true },
        .{ .name = "lm-studio-connector", .description = "LM Studio local server", .installed = true },
        .{ .name = "vllm-connector", .description = "vLLM inference server", .installed = true },
        .{ .name = "mlx-connector", .description = "MLX Apple Silicon inference", .installed = true },
        .{ .name = "llama-cpp-connector", .description = "llama.cpp local server", .installed = true },
        .{ .name = "discord-bot", .description = "Discord bot framework", .installed = true },
        .{ .name = "metrics-prometheus", .description = "Prometheus metrics exporter", .installed = false },
        .{ .name = "storage-s3", .description = "AWS S3 storage backend", .installed = false },
        .{ .name = "storage-gcs", .description = "Google Cloud Storage backend", .installed = false },
        .{ .name = "cache-redis", .description = "Redis caching layer", .installed = false },
    };

    utils.output.println("\n{s:<25} {s:<12} {s:<10} {s:<30}", .{ "NAME", "STATUS", "ENABLED", "DESCRIPTION" });
    utils.output.printSeparator(77);

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
        utils.output.println("{s:<25} {s:<12} {s:<10} {s:<30}", .{
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
        utils.output.println("\nFound: {d} plugin(s)", .{count});
        utils.output.println("", .{});
        utils.output.printInfo("Use 'abi plugins enable <name>' to enable a plugin", .{});
        utils.output.printInfo("Use 'abi plugins disable <name>' to disable a plugin", .{});
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
        \\Manage ABI framework plugins (16 built-in connectors + extensions).
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
        \\  abi plugins                          # List installed plugins
        \\  abi plugins info openai-connector    # Show OpenAI connector info
        \\  abi plugins enable discord-bot       # Enable Discord bot plugin
        \\  abi plugins search llm               # Search for LLM plugins
        \\
    ;
    utils.output.print("{s}", .{help});
}
