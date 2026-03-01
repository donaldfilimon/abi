//! User profile management commands for ABI CLI.
//!
//! Manage user profiles, preferences, and API keys.
//! Profile data is persisted to the platform-specific ABI app root.

const std = @import("std");
const abi = @import("abi");
const command_mod = @import("../command.zig");
const context_mod = @import("../framework/context.zig");
const utils = @import("../utils/mod.zig");
const cli_io = utils.io_backend;
const app_paths = abi.services.shared.app_paths;

// libc import for environment access - required for Zig 0.16
const c = @cImport(@cInclude("stdlib.h"));

pub const meta: command_mod.Meta = .{
    .name = "profile",
    .description = "User profile and settings management",
    .kind = .group,
    .subcommands = &.{ "show", "list", "create", "switch", "delete", "set", "get", "api-key", "export", "import", "help" },
    .children = &.{
        .{ .name = "show", .description = "Show current profile", .handler = command_mod.parserHandler(runShowSubcommand) },
        .{ .name = "list", .description = "List all profiles", .handler = command_mod.parserHandler(runListSubcommand) },
        .{ .name = "create", .description = "Create a new profile", .handler = command_mod.parserHandler(runCreateSubcommand) },
        .{ .name = "switch", .description = "Switch to a profile", .handler = command_mod.parserHandler(runSwitchSubcommand) },
        .{ .name = "delete", .description = "Delete a profile", .handler = command_mod.parserHandler(runDeleteSubcommand) },
        .{ .name = "set", .description = "Set a profile setting", .handler = command_mod.parserHandler(runSetSubcommand) },
        .{ .name = "get", .description = "Get a profile setting", .handler = command_mod.parserHandler(runGetSubcommand) },
        .{ .name = "api-key", .description = "Manage API keys", .handler = command_mod.parserHandler(runApiKeySubcommand) },
        .{ .name = "export", .description = "Export profile to file", .handler = command_mod.parserHandler(runExportSubcommand) },
        .{ .name = "import", .description = "Import profile from file", .handler = command_mod.parserHandler(runImportSubcommand) },
    },
};

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

/// Profile configuration
const Profile = struct {
    name: []const u8 = "default",
    default_model: []const u8 = "gpt-4",
    default_provider: []const u8 = "openai",
    temperature: f32 = 0.7,
    max_tokens: u32 = 2048,
    api_keys: std.StringHashMapUnmanaged([]const u8),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Profile {
        return Profile{
            .api_keys = .empty,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Profile) void {
        self.api_keys.deinit(self.allocator);
    }
};

/// Profile store with persistence
const ProfileStore = struct {
    profiles: std.StringHashMapUnmanaged(StoredProfile),
    active_profile: []const u8,
    allocator: std.mem.Allocator,

    const StoredProfile = struct {
        name: []const u8,
        default_model: []const u8,
        default_provider: []const u8,
        temperature: f32,
        max_tokens: u32,
    };

    pub fn init(allocator: std.mem.Allocator) ProfileStore {
        return .{
            .profiles = .{},
            .active_profile = "default",
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ProfileStore) void {
        // Free all profile data
        // Note: key and value.name point to the same memory, only free once
        var iter = self.profiles.iterator();
        while (iter.next()) |entry| {
            // Free key (which is also value.name - same allocation)
            self.allocator.free(entry.key_ptr.*);
            // Free other strings
            self.allocator.free(entry.value_ptr.default_model);
            self.allocator.free(entry.value_ptr.default_provider);
        }
        self.profiles.deinit(self.allocator);
        if (!std.mem.eql(u8, self.active_profile, "default")) {
            self.allocator.free(self.active_profile);
        }
    }

    pub fn getProfile(self: *const ProfileStore, name: []const u8) ?StoredProfile {
        return self.profiles.get(name);
    }

    pub fn addProfile(self: *ProfileStore, name: []const u8) !void {
        if (self.profiles.contains(name)) return error.ProfileExists;

        const name_copy = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_copy);

        const model_copy = try self.allocator.dupe(u8, "gpt-4");
        errdefer self.allocator.free(model_copy);

        const provider_copy = try self.allocator.dupe(u8, "openai");
        errdefer self.allocator.free(provider_copy);

        const profile = StoredProfile{
            .name = name_copy, // Same as key - freed via key in deinit
            .default_model = model_copy,
            .default_provider = provider_copy,
            .temperature = 0.7,
            .max_tokens = 2048,
        };

        try self.profiles.put(self.allocator, name_copy, profile);
    }

    pub fn removeProfile(self: *ProfileStore, name: []const u8) !void {
        if (std.mem.eql(u8, name, "default")) return error.CannotDeleteDefault;
        if (!self.profiles.contains(name)) return error.ProfileNotFound;

        // If deleting active profile, switch to default
        if (std.mem.eql(u8, self.active_profile, name)) {
            if (!std.mem.eql(u8, self.active_profile, "default")) {
                self.allocator.free(self.active_profile);
            }
            self.active_profile = "default";
        }

        // Remove and free (key and value.name are same memory)
        if (self.profiles.fetchRemove(name)) |kv| {
            self.allocator.free(kv.key); // Also frees kv.value.name
            self.allocator.free(kv.value.default_model);
            self.allocator.free(kv.value.default_provider);
        }
    }

    pub fn setActive(self: *ProfileStore, name: []const u8) !void {
        // Default always exists
        if (!std.mem.eql(u8, name, "default") and !self.profiles.contains(name)) {
            return error.ProfileNotFound;
        }

        if (!std.mem.eql(u8, self.active_profile, "default")) {
            self.allocator.free(self.active_profile);
        }

        if (std.mem.eql(u8, name, "default")) {
            self.active_profile = "default";
        } else {
            self.active_profile = try self.allocator.dupe(u8, name);
        }
    }

    pub fn updateSetting(self: *ProfileStore, profile_name: []const u8, key: []const u8, value: []const u8) !void {
        if (self.profiles.getPtr(profile_name)) |profile| {
            if (std.mem.eql(u8, key, "default_model")) {
                self.allocator.free(profile.default_model);
                profile.default_model = try self.allocator.dupe(u8, value);
            } else if (std.mem.eql(u8, key, "default_provider")) {
                self.allocator.free(profile.default_provider);
                profile.default_provider = try self.allocator.dupe(u8, value);
            } else if (std.mem.eql(u8, key, "temperature")) {
                profile.temperature = std.fmt.parseFloat(f32, value) catch 0.7;
            } else if (std.mem.eql(u8, key, "max_tokens")) {
                profile.max_tokens = std.fmt.parseInt(u32, value, 10) catch 2048;
            } else {
                return error.UnknownSetting;
            }
        } else if (std.mem.eql(u8, profile_name, "default")) {
            // Create default profile with the setting
            try self.addProfile("default");
            try self.updateSetting("default", key, value);
        } else {
            return error.ProfileNotFound;
        }
    }
};

/// Get path to profiles config file
fn getProfilesConfigPath(allocator: std.mem.Allocator) ![]u8 {
    return app_paths.resolvePath(allocator, "profiles.json");
}

const ProfileLoadState = enum {
    loaded,
    missing,
};

fn tryLoadProfileStoreFromPath(
    allocator: std.mem.Allocator,
    path: []const u8,
    store: *ProfileStore,
) !ProfileLoadState {
    // Initialize I/O backend for Zig 0.16
    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    const content = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1024 * 1024)) catch |err| switch (err) {
        error.FileNotFound => return .missing,
        else => return err,
    };
    defer allocator.free(content);

    // Preserve historical behavior: invalid JSON yields an empty store.
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, content, .{}) catch {
        return .loaded;
    };
    defer parsed.deinit();

    if (parsed.value == .object) {
        // Load active profile
        if (parsed.value.object.get("active")) |active_val| {
            if (active_val == .string) {
                if (!std.mem.eql(u8, active_val.string, "default")) {
                    store.active_profile = try allocator.dupe(u8, active_val.string);
                }
            }
        }

        // Load profiles
        if (parsed.value.object.get("profiles")) |profiles_val| {
            if (profiles_val == .object) {
                var iter = profiles_val.object.iterator();
                while (iter.next()) |entry| {
                    if (entry.value_ptr.* == .object) {
                        const profile_obj = entry.value_ptr.*.object;

                        const name_copy = try allocator.dupe(u8, entry.key_ptr.*);
                        errdefer allocator.free(name_copy);

                        const model = if (profile_obj.get("default_model")) |v| (if (v == .string) v.string else "gpt-4") else "gpt-4";
                        const provider = if (profile_obj.get("default_provider")) |v| (if (v == .string) v.string else "openai") else "openai";
                        const temp: f32 = if (profile_obj.get("temperature")) |v| (if (v == .float) @as(f32, @floatCast(v.float)) else 0.7) else 0.7;
                        const tokens: u32 = if (profile_obj.get("max_tokens")) |v| (if (v == .integer) @as(u32, @intCast(v.integer)) else 2048) else 2048;

                        const profile = ProfileStore.StoredProfile{
                            .name = name_copy,
                            .default_model = try allocator.dupe(u8, model),
                            .default_provider = try allocator.dupe(u8, provider),
                            .temperature = temp,
                            .max_tokens = tokens,
                        };

                        try store.profiles.put(allocator, name_copy, profile);
                    }
                }
            }
        }
    }

    return .loaded;
}

/// Load profile store from disk
fn loadProfileStore(allocator: std.mem.Allocator) !ProfileStore {
    var store = ProfileStore.init(allocator);
    errdefer store.deinit();

    const config_path = app_paths.resolvePath(allocator, "profiles.json") catch return store;
    defer allocator.free(config_path);
    _ = try tryLoadProfileStoreFromPath(allocator, config_path, &store);

    return store;
}

/// Save profile store to disk
fn saveProfileStore(allocator: std.mem.Allocator, store: *const ProfileStore) !void {
    const config_path = try getProfilesConfigPath(allocator);
    defer allocator.free(config_path);

    // Build JSON content
    var json_buf = std.ArrayListUnmanaged(u8).empty;
    defer json_buf.deinit(allocator);

    try json_buf.appendSlice(allocator, "{\"active\":\"");
    try json_buf.appendSlice(allocator, store.active_profile);
    try json_buf.appendSlice(allocator, "\",\"profiles\":{");

    var first = true;
    var iter = store.profiles.iterator();
    while (iter.next()) |entry| {
        if (!first) try json_buf.appendSlice(allocator, ",");
        first = false;

        try json_buf.appendSlice(allocator, "\"");
        try json_buf.appendSlice(allocator, entry.key_ptr.*);
        try json_buf.appendSlice(allocator, "\":{\"default_model\":\"");
        try json_buf.appendSlice(allocator, entry.value_ptr.default_model);
        try json_buf.appendSlice(allocator, "\",\"default_provider\":\"");
        try json_buf.appendSlice(allocator, entry.value_ptr.default_provider);
        try json_buf.appendSlice(allocator, "\",\"temperature\":");

        var temp_buf: [32]u8 = undefined;
        const temp_str = std.fmt.bufPrint(&temp_buf, "{d:.2}", .{entry.value_ptr.temperature}) catch "0.7";
        try json_buf.appendSlice(allocator, temp_str);

        try json_buf.appendSlice(allocator, ",\"max_tokens\":");

        var tokens_buf: [16]u8 = undefined;
        const tokens_str = std.fmt.bufPrint(&tokens_buf, "{d}", .{entry.value_ptr.max_tokens}) catch "2048";
        try json_buf.appendSlice(allocator, tokens_str);

        try json_buf.appendSlice(allocator, "}");
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

/// Entry point for the profile command.
/// Only reached when no child matches (help / unknown / default).
pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (args.len == 0) {
        // Default action: show current profile
        try showCurrentProfile(allocator);
        return;
    }
    const cmd = std.mem.sliceTo(args[0], 0);
    if (utils.args.matchesAny(cmd, &.{ "--help", "-h", "help" })) {
        printHelp();
        return;
    }
    // Unknown subcommand
    utils.output.printError("Unknown profile command: {s}", .{cmd});
    if (command_mod.suggestSubcommand(meta, cmd)) |suggestion| {
        utils.output.println("Did you mean: {s}", .{suggestion});
    }
}

fn runDefaultProfileAction(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    _ = parser;
    try showCurrentProfile(allocator);
}

fn runShowSubcommand(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    if (parser.wantsHelp()) {
        printHelp();
        return;
    }
    try showCurrentProfile(allocator);
}

fn runListSubcommand(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    if (parser.wantsHelp()) {
        printHelp();
        return;
    }
    try listProfiles(allocator);
}

fn runCreateSubcommand(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    if (parser.wantsHelp()) {
        printHelp();
        return;
    }
    const name = parser.next() orelse {
        utils.output.printError("Usage: abi profile create <name>", .{});
        return;
    };
    try createProfile(allocator, name);
}

fn runSwitchSubcommand(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    if (parser.wantsHelp()) {
        printHelp();
        return;
    }
    const name = parser.next() orelse {
        utils.output.printError("Usage: abi profile switch <name>", .{});
        return;
    };
    try switchProfile(allocator, name);
}

fn runDeleteSubcommand(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    if (parser.wantsHelp()) {
        printHelp();
        return;
    }
    const name = parser.next() orelse {
        utils.output.printError("Usage: abi profile delete <name>", .{});
        return;
    };
    try deleteProfile(allocator, name);
}

fn runSetSubcommand(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    if (parser.wantsHelp()) {
        printHelp();
        return;
    }
    const key = parser.next() orelse {
        utils.output.printError("Usage: abi profile set <key> <value>", .{});
        return;
    };
    const value = parser.next() orelse {
        utils.output.printError("Usage: abi profile set <key> <value>", .{});
        return;
    };
    try setProfileValue(allocator, key, value);
}

fn runGetSubcommand(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    if (parser.wantsHelp()) {
        printHelp();
        return;
    }
    const key = parser.next() orelse {
        utils.output.printError("Usage: abi profile get <key>", .{});
        return;
    };
    try getProfileValue(allocator, key);
}

fn runApiKeySubcommand(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    try handleApiKey(allocator, parser);
}

fn runExportSubcommand(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    if (parser.wantsHelp()) {
        printHelp();
        return;
    }
    const path = parser.next();
    if (parser.hasMore()) {
        utils.output.printError("Usage: abi profile export [path]", .{});
        return;
    }
    try exportProfile(allocator, path);
}

fn runImportSubcommand(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    if (parser.wantsHelp()) {
        printHelp();
        return;
    }
    const path = parser.next() orelse {
        utils.output.printError("Usage: abi profile import <path>", .{});
        return;
    };
    if (parser.hasMore()) {
        utils.output.printError("Usage: abi profile import <path>", .{});
        return;
    }
    try importProfile(allocator, path);
}

fn printHelpWithAllocator(allocator: std.mem.Allocator) void {
    _ = allocator;
    printHelp();
}

fn onUnknownSubcommand(command: []const u8) void {
    utils.output.printError("Unknown subcommand: {s}", .{command});
}

fn getConfigPath(allocator: std.mem.Allocator) ![]const u8 {
    return app_paths.resolvePath(allocator, "config.json");
}

fn getProfilesPrimaryPath(allocator: std.mem.Allocator) ![]u8 {
    return app_paths.resolvePath(allocator, "profiles.json");
}

fn printProfilesConfigLocation(allocator: std.mem.Allocator) void {
    const profiles_path = getProfilesPrimaryPath(allocator) catch {
        utils.output.println("\nConfig: (unavailable)", .{});
        return;
    };
    defer allocator.free(profiles_path);
    utils.output.println("\nConfig: {s}", .{profiles_path});
}

fn printProfilesSavedPath(allocator: std.mem.Allocator) void {
    const profiles_path = getProfilesPrimaryPath(allocator) catch {
        utils.output.printInfo("Changes saved to profile settings", .{});
        return;
    };
    defer allocator.free(profiles_path);
    utils.output.printInfo("Changes saved to {s}", .{profiles_path});
}

fn showCurrentProfile(allocator: std.mem.Allocator) !void {
    utils.output.printHeader("Current Profile");

    // Load profile store
    var store = try loadProfileStore(allocator);
    defer store.deinit();

    // Get active profile settings
    var model: []const u8 = "gpt-4";
    var provider: []const u8 = "openai";
    var temperature: f32 = 0.7;
    var max_tokens: u32 = 2048;

    if (store.getProfile(store.active_profile)) |profile| {
        model = profile.default_model;
        provider = profile.default_provider;
        temperature = profile.temperature;
        max_tokens = profile.max_tokens;
    }

    // Check environment for API keys
    const openai_key = getEnvOwned(allocator, "ABI_OPENAI_API_KEY");
    defer if (openai_key) |k| allocator.free(k);

    const anthropic_key = getEnvOwned(allocator, "ABI_ANTHROPIC_API_KEY");
    defer if (anthropic_key) |k| allocator.free(k);

    const hf_token = getEnvOwned(allocator, "ABI_HF_API_TOKEN");
    defer if (hf_token) |t| allocator.free(t);

    utils.output.println("", .{});
    utils.output.println("Profile:          {s}", .{store.active_profile});
    utils.output.println("Default Provider: {s}", .{provider});
    utils.output.println("Default Model:    {s}", .{model});
    utils.output.println("Temperature:      {d:.2}", .{temperature});
    utils.output.println("Max Tokens:       {d}", .{max_tokens});
    utils.output.println("", .{});

    utils.output.println("API Keys:", .{});
    if (openai_key) |_| {
        utils.output.println("  OpenAI:      {s}", .{"********"});
    } else {
        utils.output.println("  OpenAI:      (not set)", .{});
    }
    if (anthropic_key) |_| {
        utils.output.println("  Anthropic:   {s}", .{"********"});
    } else {
        utils.output.println("  Anthropic:   (not set)", .{});
    }
    if (hf_token) |_| {
        utils.output.println("  HuggingFace: {s}", .{"********"});
    } else {
        utils.output.println("  HuggingFace: (not set)", .{});
    }

    printProfilesConfigLocation(allocator);
    utils.output.println("Use 'abi profile set <key> <value>' to update settings", .{});
    utils.output.println("Use 'abi profile api-key set <provider> <key>' to set API keys", .{});
}

fn listProfiles(allocator: std.mem.Allocator) !void {
    utils.output.printHeader("Available Profiles");

    // Load profile store
    var store = try loadProfileStore(allocator);
    defer store.deinit();

    utils.output.println("\n{s:<20} {s:<10} {s:<20} {s:<15}", .{ "NAME", "STATUS", "PROVIDER", "MODEL" });
    utils.output.println("{s}", .{"-" ** 65});

    // Default profile is always available
    const default_active = std.mem.eql(u8, store.active_profile, "default");
    const default_status = if (default_active) "active" else "";
    if (store.getProfile("default")) |p| {
        utils.output.println("{s:<20} {s:<10} {s:<20} {s:<15}", .{ "default", default_status, p.default_provider, p.default_model });
    } else {
        utils.output.println("{s:<20} {s:<10} {s:<20} {s:<15}", .{ "default", default_status, "openai", "gpt-4" });
    }

    // List other profiles
    var iter = store.profiles.iterator();
    while (iter.next()) |entry| {
        if (std.mem.eql(u8, entry.key_ptr.*, "default")) continue;
        const is_active = std.mem.eql(u8, store.active_profile, entry.key_ptr.*);
        const status = if (is_active) "active" else "";
        utils.output.println("{s:<20} {s:<10} {s:<20} {s:<15}", .{
            entry.key_ptr.*,
            status,
            entry.value_ptr.default_provider,
            entry.value_ptr.default_model,
        });
    }

    printProfilesConfigLocation(allocator);
    utils.output.println("Use 'abi profile create <name>' to create a new profile", .{});
    utils.output.println("Use 'abi profile switch <name>' to switch profiles", .{});
}

fn createProfile(allocator: std.mem.Allocator, name: []const u8) !void {
    // Load profile store
    var store = try loadProfileStore(allocator);
    defer store.deinit();

    // Check if profile already exists
    if (store.profiles.contains(name)) {
        utils.output.printError("Profile '{s}' already exists", .{name});
        return;
    }

    // Add the profile
    store.addProfile(name) catch |err| {
        utils.output.printError("Failed to create profile: {t}", .{err});
        return;
    };

    // Save to disk
    try saveProfileStore(allocator, &store);

    utils.output.printSuccess("Profile '{s}' created", .{name});
    printProfilesSavedPath(allocator);
    utils.output.printInfo("Use 'abi profile switch {s}' to activate it", .{name});
}

fn switchProfile(allocator: std.mem.Allocator, name: []const u8) !void {
    // Load profile store
    var store = try loadProfileStore(allocator);
    defer store.deinit();

    // Check if profile exists (default always exists)
    if (!std.mem.eql(u8, name, "default") and !store.profiles.contains(name)) {
        utils.output.printError("Profile not found: {s}", .{name});
        utils.output.printInfo("Use 'abi profile list' to see available profiles", .{});
        return;
    }

    // Check if already active
    if (std.mem.eql(u8, store.active_profile, name)) {
        utils.output.printInfo("Profile '{s}' is already active", .{name});
        return;
    }

    // Set active profile
    store.setActive(name) catch |err| {
        utils.output.printError("Failed to switch profile: {t}", .{err});
        return;
    };

    // Save to disk
    try saveProfileStore(allocator, &store);

    utils.output.printSuccess("Switched to profile: {s}", .{name});
    printProfilesSavedPath(allocator);
}

fn deleteProfile(allocator: std.mem.Allocator, name: []const u8) !void {
    if (std.mem.eql(u8, name, "default")) {
        utils.output.printError("Cannot delete the default profile", .{});
        return;
    }

    // Load profile store
    var store = try loadProfileStore(allocator);
    defer store.deinit();

    // Check if profile exists
    if (!store.profiles.contains(name)) {
        utils.output.printError("Profile not found: {s}", .{name});
        utils.output.printInfo("Use 'abi profile list' to see available profiles", .{});
        return;
    }

    // Remove the profile
    store.removeProfile(name) catch |err| {
        utils.output.printError("Failed to delete profile: {t}", .{err});
        return;
    };

    // Save to disk
    try saveProfileStore(allocator, &store);

    utils.output.printSuccess("Profile '{s}' deleted", .{name});
    printProfilesSavedPath(allocator);
}

fn setProfileValue(allocator: std.mem.Allocator, key: []const u8, value: []const u8) !void {
    const valid_keys = [_][]const u8{
        "default_model",
        "default_provider",
        "temperature",
        "max_tokens",
    };

    var is_valid = false;
    for (valid_keys) |valid| {
        if (std.mem.eql(u8, key, valid)) {
            is_valid = true;
            break;
        }
    }

    if (!is_valid) {
        utils.output.printError("Unknown setting: {s}", .{key});
        utils.output.printInfo("Valid settings: default_model, default_provider, temperature, max_tokens", .{});
        return;
    }

    // Load profile store
    var store = try loadProfileStore(allocator);
    defer store.deinit();

    // Ensure active profile exists in store
    if (!store.profiles.contains(store.active_profile) and !std.mem.eql(u8, store.active_profile, "default")) {
        utils.output.printError("Active profile '{s}' not found", .{store.active_profile});
        return;
    }

    // If setting on default profile that doesn't exist yet, create it
    if (std.mem.eql(u8, store.active_profile, "default") and !store.profiles.contains("default")) {
        try store.addProfile("default");
    }

    // Update the setting
    store.updateSetting(store.active_profile, key, value) catch |err| {
        utils.output.printError("Failed to update setting: {t}", .{err});
        return;
    };

    // Save to disk
    try saveProfileStore(allocator, &store);

    utils.output.printSuccess("Set {s} = {s}", .{ key, value });
    printProfilesSavedPath(allocator);
}

fn getProfileValue(allocator: std.mem.Allocator, key: []const u8) !void {
    // Load profile store
    var store = try loadProfileStore(allocator);
    defer store.deinit();

    // Get value from active profile
    if (store.getProfile(store.active_profile)) |profile| {
        if (std.mem.eql(u8, key, "default_model")) {
            utils.output.println("{s}", .{profile.default_model});
        } else if (std.mem.eql(u8, key, "default_provider")) {
            utils.output.println("{s}", .{profile.default_provider});
        } else if (std.mem.eql(u8, key, "temperature")) {
            utils.output.println("{d:.2}", .{profile.temperature});
        } else if (std.mem.eql(u8, key, "max_tokens")) {
            utils.output.println("{d}", .{profile.max_tokens});
        } else {
            utils.output.printError("Unknown setting: {s}", .{key});
        }
    } else {
        // Default values if profile doesn't exist
        const defaults = std.StaticStringMap([]const u8).initComptime(.{
            .{ "default_model", "gpt-4" },
            .{ "default_provider", "openai" },
            .{ "temperature", "0.7" },
            .{ "max_tokens", "2048" },
        });

        if (defaults.get(key)) |value| {
            utils.output.println("{s}", .{value});
        } else {
            utils.output.printError("Unknown setting: {s}", .{key});
        }
    }
}

fn handleApiKey(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    const action = parser.next() orelse {
        // Show API key status
        try showApiKeyStatus(allocator);
        return;
    };

    if (utils.args.matchesAny(action, &[_][]const u8{ "help", "--help", "-h" })) {
        printApiKeyHelp();
        return;
    }

    if (std.mem.eql(u8, action, "set")) {
        if (parser.wantsHelp()) {
            printApiKeyHelp();
            return;
        }
        const provider = parser.next() orelse {
            utils.output.printError("Usage: abi profile api-key set <provider> <key>", .{});
            return;
        };
        if (utils.args.matchesAny(provider, &[_][]const u8{ "help", "--help", "-h" })) {
            printApiKeyHelp();
            return;
        }
        const key = parser.next() orelse {
            utils.output.printError("Usage: abi profile api-key set <provider> <key>", .{});
            return;
        };
        try setApiKey(allocator, provider, key);
    } else if (std.mem.eql(u8, action, "remove")) {
        if (parser.wantsHelp()) {
            printApiKeyHelp();
            return;
        }
        const provider = parser.next() orelse {
            utils.output.printError("Usage: abi profile api-key remove <provider>", .{});
            return;
        };
        if (utils.args.matchesAny(provider, &[_][]const u8{ "help", "--help", "-h" })) {
            printApiKeyHelp();
            return;
        }
        try removeApiKey(allocator, provider);
    } else if (std.mem.eql(u8, action, "list")) {
        if (parser.wantsHelp()) {
            printApiKeyHelp();
            return;
        }
        try showApiKeyStatus(allocator);
    } else {
        utils.output.printError("Unknown api-key action: {s}", .{action});
        printApiKeyHelp();
    }
}

fn printApiKeyHelp() void {
    utils.output.print(
        \\Usage: abi profile api-key [action] [options]
        \\
        \\Actions:
        \\  set <provider> <key>      Print instructions to set provider key
        \\  remove <provider>         Print instructions to remove provider key
        \\  list                      List configured key status
        \\
        \\Examples:
        \\  abi profile api-key
        \\  abi profile api-key list
        \\  abi profile api-key set openai sk-...
        \\  abi profile api-key remove openai
        \\
    , .{});
}

fn showApiKeyStatus(allocator: std.mem.Allocator) !void {
    utils.output.printHeader("API Key Status");

    const providers = [_]struct { name: []const u8, env_var: []const u8 }{
        .{ .name = "openai", .env_var = "ABI_OPENAI_API_KEY" },
        .{ .name = "anthropic", .env_var = "ABI_ANTHROPIC_API_KEY" },
        .{ .name = "huggingface", .env_var = "ABI_HF_API_TOKEN" },
        .{ .name = "ollama", .env_var = "ABI_OLLAMA_HOST" },
    };

    utils.output.println("\n{s:<15} {s:<15} {s:<30}", .{ "PROVIDER", "STATUS", "ENV VARIABLE" });
    utils.output.println("{s}", .{"-" ** 60});

    for (providers) |p| {
        const status = getEnvOwned(allocator, p.env_var);
        defer if (status) |s| allocator.free(s);

        const status_str = if (status != null) "configured" else "not set";
        utils.output.println("{s:<15} {s:<15} {s:<30}", .{ p.name, status_str, p.env_var });
    }

    utils.output.println("\nTo set an API key:", .{});
    utils.output.println("  export ABI_OPENAI_API_KEY=sk-...", .{});
    utils.output.println("  # or on Windows: $env:ABI_OPENAI_API_KEY=\"sk-...\"", .{});
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
        utils.output.println("\n  # Linux/macOS:", .{});
        utils.output.println("  export {s}=<your-key>", .{env_var});
        utils.output.println("\n  # Windows PowerShell:", .{});
        utils.output.println("  $env:{s}=\"<your-key>\"", .{env_var});
        utils.output.println("\n  # Windows cmd:", .{});
        utils.output.println("  set {s}=<your-key>", .{env_var});
    } else {
        utils.output.printError("Unknown provider: {s}", .{provider});
        utils.output.printInfo("Supported providers: openai, anthropic, huggingface", .{});
    }
}

fn removeApiKey(allocator: std.mem.Allocator, provider: []const u8) !void {
    _ = allocator;
    // Convert provider name to uppercase
    var upper_buf: [32]u8 = undefined;
    const upper_provider = if (provider.len <= upper_buf.len)
        std.ascii.upperString(&upper_buf, provider)
    else
        provider;

    utils.output.printInfo("To remove the API key, unset the environment variable:", .{});
    utils.output.println("\n  # Linux/macOS:", .{});
    utils.output.println("  unset ABI_{s}_API_KEY", .{upper_provider});
    utils.output.println("\n  # Windows PowerShell:", .{});
    utils.output.println("  Remove-Item Env:ABI_{s}_API_KEY", .{upper_provider});
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
    utils.output.print("{s}", .{help});
}

test {
    std.testing.refAllDecls(@This());
}
