//! File-based credential persistence (default backend).
//! Package-private building blocks for credentials.zig (hub).
//! POSIX owner-only perms (0700/0600) + Windows owner-only DACL (SDDL OW);
//! not encryption, OS keychain, or Secure Enclave.
const std = @import("std");
const builtin = @import("builtin");
const io = @import("io/mod.zig");
const utils = @import("utils.zig");
const env = @import("env.zig");
const temp_path = @import("temp_path.zig");
const types = @import("credentials_types.zig");

const Credentials = types.Credentials;

pub fn getCredentialsPath(allocator: std.mem.Allocator) ![]const u8 {
    // Allow explicit override
    if (env.get("ABI_CREDENTIALS_PATH")) |p| return try allocator.dupe(u8, p);

    // XDG_CONFIG_HOME on non-Windows
    if (builtin.target.os.tag != .windows) {
        if (env.get("XDG_CONFIG_HOME")) |xdg| {
            return try utils.pathJoin(xdg, "abi/credentials.json", allocator);
        }
    }

    // Fallback: HOME/USERPROFILE/.abi/credentials.json
    const home_var = if (builtin.target.os.tag == .windows) "USERPROFILE" else "HOME";
    const home = env.get(home_var) orelse return error.HomeNotFound;
    return try utils.pathJoin(home, ".abi/credentials.json", allocator);
}

pub fn loadCredentialsFromPath(allocator: std.mem.Allocator, path: []const u8) !Credentials {
    if (!io.fileExists(path)) return Credentials{};

    const content = try io.asyncReadFile(allocator, path);
    defer {
        // File bytes may contain secrets — wipe before free.
        std.crypto.secureZero(u8, content);
        allocator.free(content);
    }

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

pub fn saveCredentialsToPath(allocator: std.mem.Allocator, path: []const u8, creds: Credentials) !void {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer {
        std.crypto.secureZero(u8, out.written());
        out.deinit();
    }

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

    try writeCredentialsFile(path, out.written());
}

pub fn ensureCredentialsDir(path: []const u8) !void {
    const io_context = std.Options.debug_io;
    const dir_permissions = restrictedPermissions(0o700);
    _ = try std.Io.Dir.createDirPathStatus(.cwd(), io_context, path, dir_permissions);
    try setDirectoryPermissions(path, dir_permissions);
}

fn writeCredentialsFile(path: []const u8, data: []const u8) !void {
    const io_context = std.Options.debug_io;
    const file_permissions = restrictedPermissions(0o600);
    const file = try std.Io.Dir.createFileAbsolute(io_context, path, .{
        .read = true,
        .truncate = true,
        .permissions = file_permissions,
    });
    defer file.close(io_context);

    // Existing files keep their old mode across create+truncate, so repair
    // permissions before any secret bytes are written.
    try file.setPermissions(io_context, file_permissions);
    try file.writeStreamingAll(io_context, data);
    try file.setPermissions(io_context, file_permissions);
    try applyWindowsOwnerOnlyAcl(path);
}

fn setDirectoryPermissions(path: []const u8, permissions: std.Io.Dir.Permissions) !void {
    const io_context = std.Options.debug_io;
    // `iterate = true` avoids Linux O_PATH fds that make fchmod return EBADF.
    const dir = try std.Io.Dir.openDirAbsolute(io_context, path, .{ .iterate = true });
    defer dir.close(io_context);
    try dir.setPermissions(io_context, permissions);
    try applyWindowsOwnerOnlyAcl(path);
}

fn restrictedPermissions(comptime mode: std.posix.mode_t) std.Io.File.Permissions {
    if (comptime std.Io.File.Permissions.has_executable_bit) {
        return std.Io.File.Permissions.fromMode(mode);
    }
    return .default_file;
}

/// On Windows, set a DACL granting full access only to the owner (SDDL OW).
/// No-op elsewhere. Runtime verification needs a Windows host (cross-smoke is
/// compile-only). macOS login keychain is a separate opt-in path
/// (`ABI_CREDENTIALS_BACKEND=keychain`); Windows/Linux OS secret stores remain Proposed.
fn applyWindowsOwnerOnlyAcl(path: []const u8) !void {
    if (comptime builtin.target.os.tag == .windows) {
        try windows_owner_acl.apply(path);
    }
}

const windows_owner_acl = if (builtin.target.os.tag == .windows) struct {
    const PATH_MAX_WIDE = 32768;

    extern "advapi32" fn ConvertStringSecurityDescriptorToSecurityDescriptorW(
        StringSecurityDescriptor: [*:0]const u16,
        StringSDRevision: u32,
        SecurityDescriptor: *?*anyopaque,
        SecurityDescriptorSize: ?*u32,
    ) callconv(.winapi) i32;

    extern "advapi32" fn SetNamedSecurityInfoW(
        pObjectName: [*:0]u16,
        ObjectType: u32,
        SecurityInfo: u32,
        psidOwner: ?*anyopaque,
        psidGroup: ?*anyopaque,
        pDacl: ?*anyopaque,
        pSacl: ?*anyopaque,
    ) callconv(.winapi) u32;

    extern "advapi32" fn GetSecurityDescriptorDacl(
        pSecurityDescriptor: ?*anyopaque,
        lpbDaclPresent: *i32,
        pDacl: *?*anyopaque,
        lpbDaclDefaulted: *i32,
    ) callconv(.winapi) i32;

    extern "kernel32" fn LocalFree(hMem: ?*anyopaque) callconv(.winapi) ?*anyopaque;

    fn apply(path: []const u8) !void {
        const sddl_w = std.unicode.utf8ToUtf16LeStringLiteral("D:P(A;;FA;;;OW)");
        var path_w_buf: [PATH_MAX_WIDE + 1]u16 = undefined;
        const n = try std.unicode.utf8ToUtf16Le(path_w_buf[0..PATH_MAX_WIDE], path);
        path_w_buf[n] = 0;

        var sd: ?*anyopaque = null;
        if (ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl_w.ptr, 1, &sd, null) == 0 or sd == null) {
            return error.WindowsAclSetupFailed;
        }
        defer _ = LocalFree(sd);

        var present: i32 = 0;
        var dacl: ?*anyopaque = null;
        var defaulted: i32 = 0;
        if (GetSecurityDescriptorDacl(sd, &present, &dacl, &defaulted) == 0 or present == 0) {
            return error.WindowsAclSetupFailed;
        }

        const status = SetNamedSecurityInfoW(@ptrCast(&path_w_buf), 1, 0x4, null, null, dacl, null);
        if (status != 0) return error.WindowsAclApplyFailed;
    }
} else struct {
    fn apply(path: []const u8) !void {
        _ = path;
    }
};

fn dupeStringField(allocator: std.mem.Allocator, root: std.json.ObjectMap, key: []const u8) !?[]const u8 {
    const value = root.get(key) orelse return null;
    if (value != .string) return null;
    return try allocator.dupe(u8, value.string);
}

fn testCredentialsPath(allocator: std.mem.Allocator, name: []const u8) ![]u8 {
    return try temp_path.tempFilePath(allocator, name, "json");
}

fn testCredentialsDir(allocator: std.mem.Allocator, name: []const u8) ![]u8 {
    const dir = try temp_path.getTempDir(allocator);
    defer allocator.free(dir);
    return try std.fmt.allocPrint(allocator, "{s}/{s}_{d}", .{ dir, name, std.c.getpid() });
}

fn fileMode(path: []const u8) !std.posix.mode_t {
    const io_context = std.Options.debug_io;
    const file = try std.Io.Dir.openFileAbsolute(io_context, path, .{});
    defer file.close(io_context);
    const stat = try file.stat(io_context);
    return stat.permissions.toMode() & 0o777;
}

fn dirMode(path: []const u8) !std.posix.mode_t {
    const io_context = std.Options.debug_io;
    const dir = try std.Io.Dir.openDirAbsolute(io_context, path, .{});
    defer dir.close(io_context);
    const stat = try dir.stat(io_context);
    return stat.permissions.toMode() & 0o777;
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

test "Credentials save tightens existing permissive file before writing" {
    if (!std.Io.File.Permissions.has_executable_bit) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const test_path = try testCredentialsPath(allocator, "abi_credentials_perms");
    defer allocator.free(test_path);
    defer {
        var threaded: std.Io.Threaded = .init(std.heap.page_allocator, .{});
        defer threaded.deinit();
        std.Io.Dir.deleteFileAbsolute(threaded.io(), test_path) catch |err| switch (err) {
            error.FileNotFound => {},
            else => std.log.warn("cleanup failed: {s}", .{@errorName(err)}),
        };
    }

    const io_context = std.Options.debug_io;
    const file = try std.Io.Dir.createFileAbsolute(io_context, test_path, .{
        .truncate = true,
        .permissions = std.Io.File.Permissions.fromMode(0o666),
    });
    defer file.close(io_context);
    try file.setPermissions(io_context, std.Io.File.Permissions.fromMode(0o666));

    const creds = Credentials{
        .anthropic_api_key = try allocator.dupe(u8, "anthropic-secret"),
    };
    var mut_creds = creds;
    defer mut_creds.deinit(allocator);

    try saveCredentialsToPath(allocator, test_path, mut_creds);
    try std.testing.expectEqual(@as(std.posix.mode_t, 0o600), try fileMode(test_path));
}

test "Credentials directory is owner-only on supported POSIX filesystems" {
    if (!std.Io.File.Permissions.has_executable_bit) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const test_dir = try testCredentialsDir(allocator, "abi_credentials_dir_perms");
    defer allocator.free(test_dir);
    defer {
        var threaded: std.Io.Threaded = .init(std.heap.page_allocator, .{});
        defer threaded.deinit();
        std.Io.Dir.deleteTree(.cwd(), threaded.io(), test_dir) catch |err| std.log.warn("cleanup failed: {s}", .{@errorName(err)});
    }

    try ensureCredentialsDir(test_dir);
    try std.testing.expectEqual(@as(std.posix.mode_t, 0o700), try dirMode(test_dir));
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
