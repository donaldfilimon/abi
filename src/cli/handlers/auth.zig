const std = @import("std");
const builtin = @import("builtin");
const credentials = @import("../../foundation/credentials.zig");
const io = @import("../../foundation/io/mod.zig");
const utils = @import("../../foundation/utils.zig");
const usage_mod = @import("../usage.zig");

/// `abi auth <signin|logout|status> [args...]`: manage stored connector
/// credentials. `status` prints which provider credentials are configured,
/// `signin` persists a credential, and `logout` clears stored credentials.
/// Returns the process exit code.
pub fn handleAuth(io_mod: std.Io, allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len < 3) return usage_mod.usageError("usage: abi auth <signin|logout|status> [args...]");

    const sub_cmd = args[2];
    if (usage_mod.isHelpToken(sub_cmd)) return usage_mod.printCommandHelp("auth");
    if (std.mem.eql(u8, sub_cmd, "status")) {
        if (args.len == 4 and usage_mod.isHelpToken(args[3])) return authStatusHelp();
        if (args.len != 3) return usage_mod.usageError("usage: abi auth status");
        return handleAuthStatus(allocator);
    } else if (std.mem.eql(u8, sub_cmd, "logout")) {
        if (args.len == 4 and usage_mod.isHelpToken(args[3])) return authLogoutHelp();
        if (args.len != 3) return usage_mod.usageError("usage: abi auth logout");
        return handleAuthLogout(allocator);
    } else if (std.mem.eql(u8, sub_cmd, "signin")) {
        if (args.len == 4 and usage_mod.isHelpToken(args[3])) return authSigninHelp();
        if (args.len != 4) return usage_mod.usageError("usage: abi auth signin <openai|anthropic|discord|grok|twilio>");
        return handleAuthSignin(io_mod, allocator, args[3]);
    } else {
        return usage_mod.usageError("usage: abi auth <signin|logout|status>");
    }
}

pub fn handleAuthStatus(allocator: std.mem.Allocator) !u8 {
    var creds = try credentials.loadCredentials(allocator);
    defer creds.deinit(allocator);

    std.debug.print("Authentication Status:\n", .{});
    std.debug.print("  OpenAI:    {s}\n", .{if (creds.openai_api_key != null) "configured" else "not configured"});
    std.debug.print("  Anthropic: {s}\n", .{if (creds.anthropic_api_key != null) "configured" else "not configured"});
    std.debug.print("  Discord:   {s}\n", .{if (creds.discord_token != null) "configured" else "not configured"});
    std.debug.print("  Grok:      {s}\n", .{if (creds.grok_api_key != null) "configured" else "not configured"});
    std.debug.print("  Twilio:    {s}\n", .{if (creds.twilio_account_sid != null and creds.twilio_auth_token != null) "configured" else "not configured"});
    return 0;
}

pub fn handleAuthLogout(allocator: std.mem.Allocator) !u8 {
    const path = try credentials.getCredentialsPath(allocator);
    defer allocator.free(path);
    if (io.fileExists(path)) {
        var threaded: std.Io.Threaded = .init(std.heap.page_allocator, .{});
        defer threaded.deinit();
        try std.Io.Dir.deleteFileAbsolute(threaded.io(), path);
        std.debug.print("Logged out. Credentials cleared.\n", .{});
    } else {
        std.debug.print("No credentials found.\n", .{});
    }
    return 0;
}

pub fn handleAuthSignin(io_mod: std.Io, allocator: std.mem.Allocator, service: []const u8) !u8 {
    // Reject unknown services before any credential I/O or stdin prompt, so
    // malformed grammar fails fast with exit 2 and never touches the credential
    // store (and stays independent of $HOME for pure validation).
    if (!isKnownService(service)) {
        return usage_mod.usageError("usage: abi auth signin <openai|anthropic|discord|grok|twilio>");
    }

    var creds = try credentials.loadCredentials(allocator);
    defer creds.deinit(allocator);

    var buf: [1024]u8 = undefined;
    var stdin_reader = std.Io.File.stdin().reader(io_mod, &buf);

    if (std.mem.eql(u8, service, "openai")) {
        const key = try readSecretLine(&stdin_reader, "Enter API key/token for openai: ");
        if (isBlankCredential(key)) return emptyCredentialError();
        try credentials.replaceOwnedString(allocator, &creds.openai_api_key, key);
    } else if (std.mem.eql(u8, service, "anthropic")) {
        const key = try readSecretLine(&stdin_reader, "Enter API key/token for anthropic: ");
        if (isBlankCredential(key)) return emptyCredentialError();
        try credentials.replaceOwnedString(allocator, &creds.anthropic_api_key, key);
    } else if (std.mem.eql(u8, service, "discord")) {
        const key = try readSecretLine(&stdin_reader, "Enter API key/token for discord: ");
        if (isBlankCredential(key)) return emptyCredentialError();
        try credentials.replaceOwnedString(allocator, &creds.discord_token, key);
    } else if (std.mem.eql(u8, service, "grok")) {
        const key = try readSecretLine(&stdin_reader, "Enter API key/token for grok: ");
        if (isBlankCredential(key)) return emptyCredentialError();
        try credentials.replaceOwnedString(allocator, &creds.grok_api_key, key);
    } else if (std.mem.eql(u8, service, "twilio")) {
        // Read→guard→dupe per line: `readSecretLine` returns a slice into the
        // shared reader buffer that the next read overwrites, so each secret
        // must be duped before the next is read. Guarding before
        // `saveCredentials` leaves any existing on-disk credential untouched.
        const sid = try readSecretLine(&stdin_reader, "Enter Twilio Account SID: ");
        if (isBlankCredential(sid)) return emptyCredentialError();
        try credentials.replaceOwnedString(allocator, &creds.twilio_account_sid, sid);
        const token = try readSecretLine(&stdin_reader, "Enter Twilio Auth Token: ");
        if (isBlankCredential(token)) return emptyCredentialError();
        try credentials.replaceOwnedString(allocator, &creds.twilio_auth_token, token);
    } else {
        return usage_mod.usageError("unknown service; use openai, anthropic, discord, grok, or twilio");
    }

    try credentials.saveCredentials(allocator, creds);
    std.debug.print("Credentials saved for {s}.\n", .{service});
    return 0;
}

fn authStatusHelp() u8 {
    std.debug.print(
        \\usage: abi auth status
        \\
        \\Show which local connector credentials are configured.
        \\
    , .{});
    return 0;
}

fn authLogoutHelp() u8 {
    std.debug.print(
        \\usage: abi auth logout
        \\
        \\Remove the local ABI credential file when present.
        \\
    , .{});
    return 0;
}

fn authSigninHelp() u8 {
    std.debug.print(
        \\usage: abi auth signin <openai|anthropic|discord|grok|twilio>
        \\
        \\Prompt for a credential and persist it in the local ABI credential file.
        \\On POSIX TTYs, secret entry disables terminal echo (restored after read).
        \\Windows: no echo-suppress path yet (disclosed gap; use a private console).
        \\Credentials remain plaintext JSON; Windows ACL/keychain not implemented.
        \\
    , .{});
    return 0;
}

/// Platform echo helpers: Windows never references termios/libc (required for
/// clean `x86_64-windows-gnu` cross-smoke). POSIX path is TM-010 no-echo entry.
const echo_platform = if (builtin.os.tag == .windows) echo_windows else echo_posix;

const echo_windows = struct {
    const Saved = void;

    fn disableEchoIfTty(fd: std.posix.fd_t) ?Saved {
        _ = fd;
        return null;
    }

    fn restoreEcho(fd: std.posix.fd_t, original: Saved) void {
        _ = fd;
        _ = original;
    }
};

const echo_posix = struct {
    const Saved = std.posix.termios;

    fn disableEchoIfTty(fd: std.posix.fd_t) ?Saved {
        if (@hasDecl(std.posix.system, "isatty") and std.posix.system.isatty(fd) == 0) return null;
        const original = std.posix.tcgetattr(fd) catch return null;
        var raw = original;
        raw.lflag.ECHO = false;
        std.posix.tcsetattr(fd, .FLUSH, raw) catch return null;
        return original;
    }

    fn restoreEcho(fd: std.posix.fd_t, original: Saved) void {
        std.posix.tcsetattr(fd, .FLUSH, original) catch |err| {
            std.log.warn("auth: failed to restore terminal echo: {s}", .{@errorName(err)});
        };
    }
};

const disableEchoIfTty = echo_platform.disableEchoIfTty;
const restoreEcho = echo_platform.restoreEcho;

/// Pure helper used by tests: reports whether secret entry should attempt
/// no-echo based on OS. Windows remains a disclosed gap (no termios path).
pub fn secretEntryDisablesEcho() bool {
    return builtin.target.os.tag != .windows;
}

fn readSecretLine(stdin_reader: anytype, prompt: []const u8) ![]const u8 {
    std.debug.print("{s}", .{prompt});
    const fd = std.Io.File.stdin().handle;
    const saved = disableEchoIfTty(fd);
    defer if (saved) |orig| {
        restoreEcho(fd, orig);
        // Always emit a newline after a no-echo read so the next prompt starts
        // on a fresh line (the user's Enter was not echoed).
        std.debug.print("\n", .{});
    };
    const line = (try stdin_reader.interface.takeDelimiter('\n')) orelse return error.EndOfStream;
    return utils.trimWhitespace(line);
}

/// A credential is blank when, after whitespace trimming, nothing remains.
/// Storing a blank secret would make `auth status` report a service as
/// "configured" while no usable credential exists, so signin rejects it.
fn isBlankCredential(line: []const u8) bool {
    return utils.trimWhitespace(line).len == 0;
}

fn emptyCredentialError() u8 {
    return usage_mod.usageError("empty credential provided; nothing was saved");
}

/// Services the `signin` dispatch chain knows how to prompt for. Checked up
/// front so unknown grammar rejects (exit 2) before any credential I/O.
fn isKnownService(service: []const u8) bool {
    inline for (.{ "openai", "anthropic", "discord", "grok", "twilio" }) |s| {
        if (std.mem.eql(u8, service, s)) return true;
    }
    return false;
}

test "auth dispatch rejects malformed grammar with exit code 2" {
    const allocator = std.testing.allocator;
    const t = std.testing.io;
    // No subcommand, unknown subcommand, signin wrong arity, and signin with an
    // unknown service all reject with usage (exit 2) before any stdin read.
    try std.testing.expectEqual(@as(u8, 2), try handleAuth(t, allocator, &.{ "abi", "auth" }));
    try std.testing.expectEqual(@as(u8, 2), try handleAuth(t, allocator, &.{ "abi", "auth", "bogus" }));
    try std.testing.expectEqual(@as(u8, 2), try handleAuth(t, allocator, &.{ "abi", "auth", "signin" }));
    try std.testing.expectEqual(@as(u8, 2), try handleAuth(t, allocator, &.{ "abi", "auth", "signin", "openai", "extra" }));
    try std.testing.expectEqual(@as(u8, 2), try handleAuth(t, allocator, &.{ "abi", "auth", "signin", "notaservice" }));
}

test "auth handler help returns success before credential I/O" {
    const allocator = std.testing.allocator;
    const t = std.testing.io;
    try std.testing.expectEqual(@as(u8, 0), try handleAuth(t, allocator, &.{ "abi", "auth", "--help" }));
    try std.testing.expectEqual(@as(u8, 0), try handleAuth(t, allocator, &.{ "abi", "auth", "status", "--help" }));
    try std.testing.expectEqual(@as(u8, 0), try handleAuth(t, allocator, &.{ "abi", "auth", "logout", "-h" }));
    try std.testing.expectEqual(@as(u8, 0), try handleAuth(t, allocator, &.{ "abi", "auth", "signin", "help" }));
}

test "blank credential detection trims whitespace" {
    try std.testing.expect(isBlankCredential(""));
    try std.testing.expect(isBlankCredential("   "));
    try std.testing.expect(isBlankCredential("\t \n"));
    try std.testing.expect(!isBlankCredential("sk-abc"));
    try std.testing.expect(!isBlankCredential("  sk-abc  "));
}

test "secret entry disables echo on POSIX and discloses Windows gap" {
    if (builtin.target.os.tag == .windows) {
        try std.testing.expect(!secretEntryDisablesEcho());
    } else {
        try std.testing.expect(secretEntryDisablesEcho());
    }
}

test {
    std.testing.refAllDecls(@This());
}
