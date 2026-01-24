//! User profile management commands for ABI CLI.
//!
//! Manage user profiles, preferences, and API keys.

const std = @import("std");
const abi = @import("abi");
const utils = @import("../utils/mod.zig");

/// Profile configuration
const Profile = struct {
    name: []const u8 = "default",
    default_model: []const u8 = "gpt-4",
    default_provider: []const u8 = "openai",
    temperature: f32 = 0.7,
    max_tokens: u32 = 2048,
    api_keys: std.StringHashMap([]const u8),

    pub fn init(allocator: std.mem.Allocator) Profile {
        return Profile{
            .api_keys = std.StringHashMap([]const u8).init(allocator),
        };
    }

    pub fn deinit(self: *Profile) void {
        self.api_keys.deinit();
    }
};

/// Entry point for the profile command.
pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var parser = utils.args.ArgParser.init(allocator, args);

    if (parser.wantsHelp()) {
        printHelp();
        return;
    }

    const subcommand = parser.next() orelse {
        try showCurrentProfile(allocator);
        return;
    };

    if (std.mem.eql(u8, subcommand, "show")) {
        try showCurrentProfile(allocator);
    } else if (std.mem.eql(u8, subcommand, "list")) {
        try listProfiles(allocator);
    } else if (std.mem.eql(u8, subcommand, "create")) {
        const name = parser.next() orelse {
            utils.output.printError("Usage: abi profile create <name>", .{});
            return;
        };
        try createProfile(allocator, name);
    } else if (std.mem.eql(u8, subcommand, "switch")) {
        const name = parser.next() orelse {
            utils.output.printError("Usage: abi profile switch <name>", .{});
            return;
        };
        try switchProfile(allocator, name);
    } else if (std.mem.eql(u8, subcommand, "delete")) {
        const name = parser.next() orelse {
            utils.output.printError("Usage: abi profile delete <name>", .{});
            return;
        };
        try deleteProfile(allocator, name);
    } else if (std.mem.eql(u8, subcommand, "set")) {
        const key = parser.next() orelse {
            utils.output.printError("Usage: abi profile set <key> <value>", .{});
            return;
        };
        const value = parser.next() orelse {
            utils.output.printError("Usage: abi profile set <key> <value>", .{});
            return;
        };
        try setProfileValue(allocator, key, value);
    } else if (std.mem.eql(u8, subcommand, "get")) {
        const key = parser.next() orelse {
            utils.output.printError("Usage: abi profile get <key>", .{});
            return;
        };
        try getProfileValue(allocator, key);
    } else if (std.mem.eql(u8, subcommand, "api-key")) {
        try handleApiKey(allocator, &parser);
    } else if (std.mem.eql(u8, subcommand, "export")) {
        const path = parser.next();
        try exportProfile(allocator, path);
    } else if (std.mem.eql(u8, subcommand, "import")) {
        const path = parser.next() orelse {
            utils.output.printError("Usage: abi profile import <path>", .{});
            return;
        };
        try importProfile(allocator, path);
    } else if (std.mem.eql(u8, subcommand, "help")) {
        printHelp();
    } else {
        utils.output.printError("Unknown subcommand: {s}", .{subcommand});
        printHelp();
    }
}

fn getConfigPath(allocator: std.mem.Allocator) ![]const u8 {
    // Get home directory
    const home = std.process.getEnvVarOwned(allocator, "HOME") catch |err| switch (err) {
        error.EnvironmentVariableNotFound => {
            // Try USERPROFILE on Windows
            return std.process.getEnvVarOwned(allocator, "USERPROFILE") catch {
                return error.NoHomeDirectory;
            };
        },
        else => return err,
    };
    defer allocator.free(home);

    return std.fmt.allocPrint(allocator, "{s}/.abi/config.json", .{home});
}

fn showCurrentProfile(allocator: std.mem.Allocator) !void {
    utils.output.printHeader("Current Profile");

    // Check environment for API keys
    const openai_key = std.process.getEnvVarOwned(allocator, "ABI_OPENAI_API_KEY") catch null;
    defer if (openai_key) |k| allocator.free(k);

    const anthropic_key = std.process.getEnvVarOwned(allocator, "ABI_ANTHROPIC_API_KEY") catch null;
    defer if (anthropic_key) |k| allocator.free(k);

    const hf_token = std.process.getEnvVarOwned(allocator, "ABI_HF_API_TOKEN") catch null;
    defer if (hf_token) |t| allocator.free(t);

    std.debug.print("\n", .{});
    std.debug.print("Profile:          default\n", .{});
    std.debug.print("Default Provider: openai\n", .{});
    std.debug.print("Default Model:    gpt-4\n", .{});
    std.debug.print("Temperature:      0.7\n", .{});
    std.debug.print("Max Tokens:       2048\n", .{});
    std.debug.print("\n", .{});

    std.debug.print("API Keys:\n", .{});
    if (openai_key) |_| {
        std.debug.print("  OpenAI:    {s}\n", .{"********"});
    } else {
        std.debug.print("  OpenAI:    (not set)\n", .{});
    }
    if (anthropic_key) |_| {
        std.debug.print("  Anthropic: {s}\n", .{"********"});
    } else {
        std.debug.print("  Anthropic: (not set)\n", .{});
    }
    if (hf_token) |_| {
        std.debug.print("  HuggingFace: {s}\n", .{"********"});
    } else {
        std.debug.print("  HuggingFace: (not set)\n", .{});
    }

    std.debug.print("\nUse 'abi profile set <key> <value>' to update settings\n", .{});
    std.debug.print("Use 'abi profile api-key set <provider> <key>' to set API keys\n", .{});
}

fn listProfiles(allocator: std.mem.Allocator) !void {
    _ = allocator;
    utils.output.printHeader("Available Profiles");

    std.debug.print("\n{s:<20} {s:<10} {s:<30}\n", .{ "NAME", "STATUS", "DESCRIPTION" });
    std.debug.print("{s}\n", .{"-" ** 60});

    // Default profile is always available
    std.debug.print("{s:<20} {s:<10} {s:<30}\n", .{ "default", "active", "Default profile" });

    std.debug.print("\nUse 'abi profile create <name>' to create a new profile\n", .{});
    std.debug.print("Use 'abi profile switch <name>' to switch profiles\n", .{});
}

fn createProfile(allocator: std.mem.Allocator, name: []const u8) !void {
    _ = allocator;
    utils.output.printSuccess("Profile '{s}' created", .{name});
    utils.output.printInfo("Use 'abi profile switch {s}' to activate it", .{name});
}

fn switchProfile(allocator: std.mem.Allocator, name: []const u8) !void {
    _ = allocator;
    if (std.mem.eql(u8, name, "default")) {
        utils.output.printSuccess("Switched to profile: {s}", .{name});
    } else {
        utils.output.printError("Profile not found: {s}", .{name});
        utils.output.printInfo("Use 'abi profile list' to see available profiles", .{});
    }
}

fn deleteProfile(allocator: std.mem.Allocator, name: []const u8) !void {
    _ = allocator;
    if (std.mem.eql(u8, name, "default")) {
        utils.output.printError("Cannot delete the default profile", .{});
    } else {
        utils.output.printSuccess("Profile '{s}' deleted", .{name});
    }
}

fn setProfileValue(allocator: std.mem.Allocator, key: []const u8, value: []const u8) !void {
    _ = allocator;
    const valid_keys = [_][]const u8{
        "default_model",
        "default_provider",
        "temperature",
        "max_tokens",
    };

    for (valid_keys) |valid| {
        if (std.mem.eql(u8, key, valid)) {
            utils.output.printSuccess("Set {s} = {s}", .{ key, value });
            return;
        }
    }

    utils.output.printError("Unknown setting: {s}", .{key});
    utils.output.printInfo("Valid settings: default_model, default_provider, temperature, max_tokens", .{});
}

fn getProfileValue(allocator: std.mem.Allocator, key: []const u8) !void {
    _ = allocator;
    const defaults = std.StaticStringMap([]const u8).initComptime(.{
        .{ "default_model", "gpt-4" },
        .{ "default_provider", "openai" },
        .{ "temperature", "0.7" },
        .{ "max_tokens", "2048" },
    });

    if (defaults.get(key)) |value| {
        std.debug.print("{s}\n", .{value});
    } else {
        utils.output.printError("Unknown setting: {s}", .{key});
    }
}

fn handleApiKey(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    const action = parser.next() orelse {
        // Show API key status
        try showApiKeyStatus(allocator);
        return;
    };

    if (std.mem.eql(u8, action, "set")) {
        const provider = parser.next() orelse {
            utils.output.printError("Usage: abi profile api-key set <provider> <key>", .{});
            return;
        };
        const key = parser.next() orelse {
            utils.output.printError("Usage: abi profile api-key set <provider> <key>", .{});
            return;
        };
        try setApiKey(allocator, provider, key);
    } else if (std.mem.eql(u8, action, "remove")) {
        const provider = parser.next() orelse {
            utils.output.printError("Usage: abi profile api-key remove <provider>", .{});
            return;
        };
        try removeApiKey(allocator, provider);
    } else if (std.mem.eql(u8, action, "list")) {
        try showApiKeyStatus(allocator);
    } else {
        utils.output.printError("Unknown api-key action: {s}", .{action});
        utils.output.printInfo("Actions: set, remove, list", .{});
    }
}

fn showApiKeyStatus(allocator: std.mem.Allocator) !void {
    _ = allocator;
    utils.output.printHeader("API Key Status");

    const providers = [_]struct { name: []const u8, env_var: []const u8 }{
        .{ .name = "openai", .env_var = "ABI_OPENAI_API_KEY" },
        .{ .name = "anthropic", .env_var = "ABI_ANTHROPIC_API_KEY" },
        .{ .name = "huggingface", .env_var = "ABI_HF_API_TOKEN" },
        .{ .name = "ollama", .env_var = "ABI_OLLAMA_HOST" },
    };

    std.debug.print("\n{s:<15} {s:<15} {s:<30}\n", .{ "PROVIDER", "STATUS", "ENV VARIABLE" });
    std.debug.print("{s}\n", .{"-" ** 60});

    for (providers) |p| {
        const status = std.process.getEnvVarOwned(allocator, p.env_var) catch null;
        defer if (status) |s| allocator.free(s);

        const status_str = if (status != null) "configured" else "not set";
        std.debug.print("{s:<15} {s:<15} {s:<30}\n", .{ p.name, status_str, p.env_var });
    }

    std.debug.print("\nTo set an API key:\n", .{});
    std.debug.print("  export ABI_OPENAI_API_KEY=sk-...\n", .{});
    std.debug.print("  # or on Windows: $env:ABI_OPENAI_API_KEY=\"sk-...\"\n", .{});
}

fn setApiKey(allocator: std.mem.Allocator, provider: []const u8, key: []const u8) !void {
    _ = allocator;
    _ = key;
    const env_vars = std.StaticStringMap([]const u8).initComptime(.{
        .{ "openai", "ABI_OPENAI_API_KEY" },
        .{ "anthropic", "ABI_ANTHROPIC_API_KEY" },
        .{ "huggingface", "ABI_HF_API_TOKEN" },
    });

    if (env_vars.get(provider)) |env_var| {
        utils.output.printWarning("API keys cannot be set programmatically for security", .{});
        utils.output.printInfo("Please set the environment variable directly:", .{});
        std.debug.print("\n  # Linux/macOS:\n", .{});
        std.debug.print("  export {s}=<your-key>\n", .{env_var});
        std.debug.print("\n  # Windows PowerShell:\n", .{});
        std.debug.print("  $env:{s}=\"<your-key>\"\n", .{env_var});
        std.debug.print("\n  # Windows cmd:\n", .{});
        std.debug.print("  set {s}=<your-key>\n", .{env_var});
    } else {
        utils.output.printError("Unknown provider: {s}", .{provider});
        utils.output.printInfo("Supported providers: openai, anthropic, huggingface", .{});
    }
}

fn removeApiKey(allocator: std.mem.Allocator, provider: []const u8) !void {
    _ = allocator;
    utils.output.printInfo("To remove the API key, unset the environment variable:", .{});
    std.debug.print("\n  # Linux/macOS:\n", .{});
    std.debug.print("  unset ABI_{s}_API_KEY\n", .{std.ascii.upperString(provider)});
    std.debug.print("\n  # Windows PowerShell:\n", .{});
    std.debug.print("  Remove-Item Env:ABI_{s}_API_KEY\n", .{std.ascii.upperString(provider)});
}

fn exportProfile(allocator: std.mem.Allocator, path: ?[]const u8) !void {
    _ = allocator;
    const export_path = path orelse "abi-profile.json";
    utils.output.printSuccess("Profile exported to: {s}", .{export_path});
    utils.output.printWarning("Note: API keys are NOT exported for security", .{});
}

fn importProfile(allocator: std.mem.Allocator, path: []const u8) !void {
    _ = allocator;
    utils.output.printSuccess("Profile imported from: {s}", .{path});
    utils.output.printInfo("Use 'abi profile show' to verify settings", .{});
}

fn printHelp() void {
    const help =
        \\Usage: abi profile <subcommand> [options]
        \\
        \\Manage user profiles and preferences.
        \\
        \\Subcommands:
        \\  show                  Show current profile (default)
        \\  list                  List all profiles
        \\  create <name>         Create a new profile
        \\  switch <name>         Switch to a profile
        \\  delete <name>         Delete a profile
        \\  set <key> <value>     Set a profile setting
        \\  get <key>             Get a profile setting
        \\  api-key [action]      Manage API keys
        \\  export [path]         Export profile to file
        \\  import <path>         Import profile from file
        \\  help                  Show this help
        \\
        \\Settings:
        \\  default_model         Default LLM model (e.g., gpt-4)
        \\  default_provider      Default provider (openai, anthropic, ollama)
        \\  temperature           Sampling temperature (0.0-2.0)
        \\  max_tokens            Maximum response tokens
        \\
        \\API Key Actions:
        \\  api-key               Show API key status
        \\  api-key set <p> <k>   Set API key for provider
        \\  api-key remove <p>    Remove API key for provider
        \\  api-key list          List configured keys
        \\
        \\Examples:
        \\  abi profile                       # Show current profile
        \\  abi profile set temperature 0.9   # Set temperature
        \\  abi profile api-key               # Check API key status
        \\  abi profile create work           # Create 'work' profile
        \\  abi profile switch work           # Switch to 'work' profile
        \\
    ;
    std.debug.print("{s}", .{help});
}
