const std = @import("std");
const builtin = @import("builtin");
const io = @import("io/mod.zig");
const utils = @import("utils.zig");
const env = @import("env.zig");

pub const Credentials = struct {
    openai_api_key: ?[]const u8 = null,
    anthropic_api_key: ?[]const u8 = null,
    discord_token: ?[]const u8 = null,
    grok_api_key: ?[]const u8 = null,
    twilio_account_sid: ?[]const u8 = null,
    twilio_auth_token: ?[]const u8 = null,

    pub fn deinit(self: *Credentials, allocator: std.mem.Allocator) void {
        if (self.openai_api_key) |k| allocator.free(k);
        if (self.anthropic_api_key) |k| allocator.free(k);
        if (self.discord_token) |k| allocator.free(k);
        if (self.grok_api_key) |k| allocator.free(k);
        if (self.twilio_account_sid) |k| allocator.free(k);
        if (self.twilio_auth_token) |k| allocator.free(k);
    }
};

pub fn replaceOwnedString(allocator: std.mem.Allocator, field: *?[]const u8, value: []const u8) !void {
    const replacement = try allocator.dupe(u8, value);
    if (field.*) |old| allocator.free(old);
    field.* = replacement;
}

pub fn getCredentialsPath(allocator: std.mem.Allocator) ![]const u8 {
    // Portable home-dir resolution (no libc); borrowed from the captured
    // process environment. Windows exposes the profile dir as USERPROFILE;
    // POSIX uses HOME.
    const home_var = if (builtin.target.os.tag == .windows) "USERPROFILE" else "HOME";
    const home = env.get(home_var) orelse return error.HomeNotFound;
    return try utils.pathJoin(home, ".abi/credentials.json", allocator);
}

pub fn loadCredentials(allocator: std.mem.Allocator) !Credentials {
    const path = try getCredentialsPath(allocator);
    defer allocator.free(path);

    return try loadCredentialsFromPath(allocator, path);
}

fn loadCredentialsFromPath(allocator: std.mem.Allocator, path: []const u8) !Credentials {
    if (!io.fileExists(path)) return Credentials{};

    const content = try io.asyncReadFile(allocator, path);
    defer allocator.free(content);

    const tree = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    defer tree.deinit();

    const root = switch (tree.value) {
        .object => |obj| obj,
        else => return error.InvalidCredentialsJson,
    };

    return Credentials{
        .openai_api_key = try dupeStringField(allocator, root, "openai_api_key"),
        .anthropic_api_key = try dupeStringField(allocator, root, "anthropic_api_key"),
        .discord_token = try dupeStringField(allocator, root, "discord_token"),
        .grok_api_key = try dupeStringField(allocator, root, "grok_api_key"),
        .twilio_account_sid = try dupeStringField(allocator, root, "twilio_account_sid"),
        .twilio_auth_token = try dupeStringField(allocator, root, "twilio_auth_token"),
    };
}

pub fn saveCredentials(allocator: std.mem.Allocator, creds: Credentials) !void {
    const path = try getCredentialsPath(allocator);
    defer allocator.free(path);

    const dir = utils.pathDirname(path);
    try io.ensureDir(dir);

    try saveCredentialsToPath(allocator, path, creds);
}

fn saveCredentialsToPath(allocator: std.mem.Allocator, path: []const u8, creds: Credentials) !void {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();

    var writer = std.json.Stringify{
        .writer = &out.writer,
        .options = .{ .whitespace = .minified },
    };
    try writer.beginObject();
    if (creds.openai_api_key) |k| {
        try writer.objectField("openai_api_key");
        try writer.write(k);
    }
    if (creds.anthropic_api_key) |k| {
        try writer.objectField("anthropic_api_key");
        try writer.write(k);
    }
    if (creds.discord_token) |k| {
        try writer.objectField("discord_token");
        try writer.write(k);
    }
    if (creds.grok_api_key) |k| {
        try writer.objectField("grok_api_key");
        try writer.write(k);
    }
    if (creds.twilio_account_sid) |k| {
        try writer.objectField("twilio_account_sid");
        try writer.write(k);
    }
    if (creds.twilio_auth_token) |k| {
        try writer.objectField("twilio_auth_token");
        try writer.write(k);
    }
    try writer.endObject();

    try io.asyncWriteFile(path, out.written());

    // Set restrictive permissions (0600)
    const os_tag = builtin.target.os.tag;
    if (os_tag == .macos or os_tag == .linux) {
        var threaded: std.Io.Threaded = .init(std.heap.page_allocator, .{});
        defer threaded.deinit();
        const io_context = threaded.io();
        const file = try std.Io.Dir.openFileAbsolute(io_context, path, .{ .mode = .read_write });
        defer file.close(io_context);
        try file.setPermissions(io_context, std.Io.File.Permissions.fromMode(0o600));
    }
}

fn dupeStringField(allocator: std.mem.Allocator, root: std.json.ObjectMap, key: []const u8) !?[]const u8 {
    const value = root.get(key) orelse return null;
    if (value != .string) return null;
    return try allocator.dupe(u8, value.string);
}

fn testCredentialsPath(allocator: std.mem.Allocator, name: []const u8) ![]u8 {
    return try std.fmt.allocPrint(allocator, "/tmp/{s}_{d}.json", .{ name, std.c.getpid() });
}

test {
    std.testing.refAllDecls(@This());
}

test "Credentials missing file returns empty credentials" {
    const test_path = try testCredentialsPath(std.testing.allocator, "abi_credentials_missing");
    defer std.testing.allocator.free(test_path);

    const creds = try loadCredentialsFromPath(std.testing.allocator, test_path);
    try std.testing.expect(creds.openai_api_key == null);
    try std.testing.expect(creds.anthropic_api_key == null);
    try std.testing.expect(creds.discord_token == null);
    try std.testing.expect(creds.grok_api_key == null);
    try std.testing.expect(creds.twilio_account_sid == null);
    try std.testing.expect(creds.twilio_auth_token == null);
}

test "Credentials save and load from explicit path" {
    const allocator = std.testing.allocator;
    const test_path = try testCredentialsPath(allocator, "abi_credentials_roundtrip");
    defer allocator.free(test_path);
    defer {
        var threaded: std.Io.Threaded = .init(std.heap.page_allocator, .{});
        defer threaded.deinit();
        std.Io.Dir.deleteFileAbsolute(threaded.io(), test_path) catch |err| switch (err) {
            error.FileNotFound => {},
            else => std.log.warn("cleanup failed: {s}", .{@errorName(err)}),
        };
    }

    const creds = Credentials{
        .openai_api_key = try allocator.dupe(u8, "sk-test-key"),
        .anthropic_api_key = null,
        .discord_token = try allocator.dupe(u8, "token-123"),
        .grok_api_key = null,
        .twilio_account_sid = try allocator.dupe(u8, "AC123"),
        .twilio_auth_token = try allocator.dupe(u8, "twilio-secret"),
    };
    var mut_creds = creds;
    defer mut_creds.deinit(allocator);

    try saveCredentialsToPath(allocator, test_path, mut_creds);

    var loaded = try loadCredentialsFromPath(allocator, test_path);
    defer loaded.deinit(allocator);

    try std.testing.expectEqualStrings("sk-test-key", loaded.openai_api_key orelse return error.MissingOpenAiKey);
    try std.testing.expect(loaded.anthropic_api_key == null);
    try std.testing.expectEqualStrings("token-123", loaded.discord_token orelse return error.MissingDiscordToken);
    try std.testing.expect(loaded.grok_api_key == null);
    try std.testing.expectEqualStrings("AC123", loaded.twilio_account_sid orelse return error.MissingTwilioSid);
    try std.testing.expectEqualStrings("twilio-secret", loaded.twilio_auth_token orelse return error.MissingTwilioToken);
}

test "Credentials reject non-object json" {
    const test_path = try testCredentialsPath(std.testing.allocator, "abi_credentials_invalid_shape");
    defer std.testing.allocator.free(test_path);
    defer {
        var threaded: std.Io.Threaded = .init(std.heap.page_allocator, .{});
        defer threaded.deinit();
        std.Io.Dir.deleteFileAbsolute(threaded.io(), test_path) catch |err| switch (err) {
            error.FileNotFound => {},
            else => std.log.warn("cleanup failed: {s}", .{@errorName(err)}),
        };
    }

    try io.asyncWriteFile(test_path, "[]");
    try std.testing.expectError(error.InvalidCredentialsJson, loadCredentialsFromPath(std.testing.allocator, test_path));
}

test "replaceOwnedString preserves old value on allocation failure" {
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    const allocator = std.testing.allocator;
    var field: ?[]const u8 = try allocator.dupe(u8, "old-value");
    defer if (field) |value| allocator.free(value);

    try std.testing.expectError(error.OutOfMemory, replaceOwnedString(failing.allocator(), &field, "new-value"));
    try std.testing.expectEqualStrings("old-value", field orelse return error.MissingOldValue);
}
