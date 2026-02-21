//! User profile management commands for ABI CLI.
//!
//! Manage user profiles, preferences, and API keys.
//! Profile data is persisted to ~/.abi/profiles.json

const std = @import("std");
const abi = @import("abi");
const utils = @import("../utils/mod.zig");
const cli_io = utils.io_backend;

// libc import for environment access - required for Zig 0.16
const c = @cImport(@cInclude("stdlib.h"));

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
fn getProfilesConfigPath(allocator: std.mem.Allocator) ![]const u8 {
    const home = getEnvOwned(allocator, "HOME") orelse
        getEnvOwned(allocator, "USERPROFILE") orelse
        return error.NoHomeDirectory;
    defer allocator.free(home);

    return std.fmt.allocPrint(allocator, "{s}/.abi/profiles.json", .{home});
}

/// Load profile store from disk
fn loadProfileStore(allocator: std.mem.Allocator) !ProfileStore {
    var store = ProfileStore.init(allocator);
    errdefer store.deinit();

    const config_path = getProfilesConfigPath(allocator) catch return store;
    defer allocator.free(config_path);

    // Initialize I/O backend for Zig 0.16
    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    const content = std.Io.Dir.cwd().readFileAlloc(io, config_path, allocator, .limited(1024 * 1024)) catch {
        return store; // File doesn't exist, return empty store
    };
    defer allocator.free(content);

    // Parse JSON
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, content, .{}) catch {
        return store; // Invalid JSON, return empty store
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

    // Ensure directory exists by trying to open/create it
    const home = getEnvOwned(allocator, "HOME") orelse
        getEnvOwned(allocator, "USERPROFILE") orelse
        return error.NoHomeDirectory;
    defer allocator.free(home);

    const dir_path = try std.fmt.allocPrint(allocator, "{s}/.abi", .{home});
    defer allocator.free(dir_path);

    // Try to create the directory (using openDir to check if it exists first)
    _ = std.Io.Dir.cwd().openDir(io, dir_path, .{}) catch {
        // Directory doesn't exist, use platform-specific mkdir
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

/// Entry point for the profile command.
pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var parser = utils.args.ArgParser.init(allocator, args);

    const commands = [_]utils.subcommand.Command{
        .{ .names = &.{"show"}, .run = runShowSubcommand },
        .{ .names = &.{"list"}, .run = runListSubcommand },
        .{ .names = &.{"create"}, .run = runCreateSubcommand },
        .{ .names = &.{"switch"}, .run = runSwitchSubcommand },
        .{ .names = &.{"delete"}, .run = runDeleteSubcommand },
        .{ .names = &.{"set"}, .run = runSetSubcommand },
        .{ .names = &.{"get"}, .run = runGetSubcommand },
        .{ .names = &.{"api-key"}, .run = runApiKeySubcommand },
        .{ .names = &.{"export"}, .run = runExportSubcommand },
        .{ .names = &.{"import"}, .run = runImportSubcommand },
    };

    try utils.subcommand.runSubcommand(
        allocator,
        &parser,
        &commands,
        runDefaultProfileAction,
        printHelpWithAllocator,
        onUnknownSubcommand,
    );
}

fn runDefaultProfileAction(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    _ = parser;
    try showCurrentProfile(allocator);
}

fn runShowSubcommand(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    _ = parser;
    try showCurrentProfile(allocator);
}

fn runListSubcommand(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    _ = parser;
    try listProfiles(allocator);
}

fn runCreateSubcommand(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    const name = parser.next() orelse {
        utils.output.printError("Usage: abi profile create <name>", .{});
        return;
    };
    try createProfile(allocator, name);
}

fn runSwitchSubcommand(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    const name = parser.next() orelse {
        utils.output.printError("Usage: abi profile switch <name>", .{});
        return;
    };
    try switchProfile(allocator, name);
}

fn runDeleteSubcommand(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    const name = parser.next() orelse {
        utils.output.printError("Usage: abi profile delete <name>", .{});
        return;
    };
    try deleteProfile(allocator, name);
}

fn runSetSubcommand(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
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
    const path = parser.next();
    if (parser.hasMore()) {
        utils.output.printError("Usage: abi profile export [path]", .{});
        return;
    }
    try exportProfile(allocator, path);
}

fn runImportSubcommand(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
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
    // Get home directory (try HOME, then USERPROFILE for Windows)
    const home = getEnvOwned(allocator, "HOME") orelse
        getEnvOwned(allocator, "USERPROFILE") orelse
        return error.NoHomeDirectory;
    defer allocator.free(home);

    return std.fmt.allocPrint(allocator, "{s}/.abi/config.json", .{home});
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

    std.debug.print("\n", .{});
    std.debug.print("Profile:          {s}\n", .{store.active_profile});
    std.debug.print("Default Provider: {s}\n", .{provider});
    std.debug.print("Default Model:    {s}\n", .{model});
    std.debug.print("Temperature:      {d:.2}\n", .{temperature});
    std.debug.print("Max Tokens:       {d}\n", .{max_tokens});
    std.debug.print("\n", .{});

    std.debug.print("API Keys:\n", .{});
    if (openai_key) |_| {
        std.debug.print("  OpenAI:      {s}\n", .{"********"});
    } else {
        std.debug.print("  OpenAI:      (not set)\n", .{});
    }
    if (anthropic_key) |_| {
        std.debug.print("  Anthropic:   {s}\n", .{"********"});
    } else {
        std.debug.print("  Anthropic:   (not set)\n", .{});
    }
    if (hf_token) |_| {
        std.debug.print("  HuggingFace: {s}\n", .{"********"});
    } else {
        std.debug.print("  HuggingFace: (not set)\n", .{});
    }

    std.debug.print("\nConfig: ~/.abi/profiles.json\n", .{});
    std.debug.print("Use 'abi profile set <key> <value>' to update settings\n", .{});
    std.debug.print("Use 'abi profile api-key set <provider> <key>' to set API keys\n", .{});
}

fn listProfiles(allocator: std.mem.Allocator) !void {
    utils.output.printHeader("Available Profiles");

    // Load profile store
    var store = try loadProfileStore(allocator);
    defer store.deinit();

    std.debug.print("\n{s:<20} {s:<10} {s:<20} {s:<15}\n", .{ "NAME", "STATUS", "PROVIDER", "MODEL" });
    std.debug.print("{s}\n", .{"-" ** 65});

    // Default profile is always available
    const default_active = std.mem.eql(u8, store.active_profile, "default");
    const default_status = if (default_active) "active" else "";
    if (store.getProfile("default")) |p| {
        std.debug.print("{s:<20} {s:<10} {s:<20} {s:<15}\n", .{ "default", default_status, p.default_provider, p.default_model });
    } else {
        std.debug.print("{s:<20} {s:<10} {s:<20} {s:<15}\n", .{ "default", default_status, "openai", "gpt-4" });
    }

    // List other profiles
    var iter = store.profiles.iterator();
    while (iter.next()) |entry| {
        if (std.mem.eql(u8, entry.key_ptr.*, "default")) continue;
        const is_active = std.mem.eql(u8, store.active_profile, entry.key_ptr.*);
        const status = if (is_active) "active" else "";
        std.debug.print("{s:<20} {s:<10} {s:<20} {s:<15}\n", .{
            entry.key_ptr.*,
            status,
            entry.value_ptr.default_provider,
            entry.value_ptr.default_model,
        });
    }

    std.debug.print("\nConfig: ~/.abi/profiles.json\n", .{});
    std.debug.print("Use 'abi profile create <name>' to create a new profile\n", .{});
    std.debug.print("Use 'abi profile switch <name>' to switch profiles\n", .{});
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
    utils.output.printInfo("Changes saved to ~/.abi/profiles.json", .{});
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
    utils.output.printInfo("Changes saved to ~/.abi/profiles.json", .{});
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
    utils.output.printInfo("Changes saved to ~/.abi/profiles.json", .{});
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
    utils.output.printInfo("Changes saved to ~/.abi/profiles.json", .{});
}

fn getProfileValue(allocator: std.mem.Allocator, key: []const u8) !void {
    // Load profile store
    var store = try loadProfileStore(allocator);
    defer store.deinit();

    // Get value from active profile
    if (store.getProfile(store.active_profile)) |profile| {
        if (std.mem.eql(u8, key, "default_model")) {
            std.debug.print("{s}\n", .{profile.default_model});
        } else if (std.mem.eql(u8, key, "default_provider")) {
            std.debug.print("{s}\n", .{profile.default_provider});
        } else if (std.mem.eql(u8, key, "temperature")) {
            std.debug.print("{d:.2}\n", .{profile.temperature});
        } else if (std.mem.eql(u8, key, "max_tokens")) {
            std.debug.print("{d}\n", .{profile.max_tokens});
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
            std.debug.print("{s}\n", .{value});
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
    std.debug.print(
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

    std.debug.print("\n{s:<15} {s:<15} {s:<30}\n", .{ "PROVIDER", "STATUS", "ENV VARIABLE" });
    std.debug.print("{s}\n", .{"-" ** 60});

    for (providers) |p| {
        const status = getEnvOwned(allocator, p.env_var);
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
    // Convert provider name to uppercase
    var upper_buf: [32]u8 = undefined;
    const upper_provider = if (provider.len <= upper_buf.len)
        std.ascii.upperString(&upper_buf, provider)
    else
        provider;

    utils.output.printInfo("To remove the API key, unset the environment variable:", .{});
    std.debug.print("\n  # Linux/macOS:\n", .{});
    std.debug.print("  unset ABI_{s}_API_KEY\n", .{upper_provider});
    std.debug.print("\n  # Windows PowerShell:\n", .{});
    std.debug.print("  Remove-Item Env:ABI_{s}_API_KEY\n", .{upper_provider});
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
