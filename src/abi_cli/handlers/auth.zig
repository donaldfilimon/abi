const std = @import("std");
const credentials = @import("../../foundation/credentials.zig");
const io = @import("../../foundation/io/mod.zig");
const utils = @import("../../foundation/utils.zig");
const usage_mod = @import("../usage.zig");

pub fn handleAuth(io_mod: std.Io, allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len < 3) return usage_mod.usageError("usage: abi auth <signin|logout|status> [args...]");

    const sub_cmd = args[2];
    if (std.mem.eql(u8, sub_cmd, "status")) {
        var creds = try credentials.loadCredentials(allocator);
        defer creds.deinit(allocator);

        std.debug.print("Authentication Status:\n", .{});
        std.debug.print("  OpenAI:    {s}\n", .{if (creds.openai_api_key != null) "configured" else "not configured"});
        std.debug.print("  Anthropic: {s}\n", .{if (creds.anthropic_api_key != null) "configured" else "not configured"});
        std.debug.print("  Discord:   {s}\n", .{if (creds.discord_token != null) "configured" else "not configured"});
        std.debug.print("  Grok:      {s}\n", .{if (creds.grok_api_key != null) "configured" else "not configured"});
        std.debug.print("  Twilio:    {s}\n", .{if (creds.twilio_account_sid != null and creds.twilio_auth_token != null) "configured" else "not configured"});
        return 0;
    } else if (std.mem.eql(u8, sub_cmd, "logout")) {
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
    } else if (std.mem.eql(u8, sub_cmd, "signin")) {
        if (args.len != 4) return usage_mod.usageError("usage: abi auth signin <openai|anthropic|discord|grok|twilio>");
        const service = args[3];

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
    } else {
        return usage_mod.usageError("usage: abi auth <signin|logout|status>");
    }
}

fn readSecretLine(stdin_reader: anytype, prompt: []const u8) ![]const u8 {
    std.debug.print("{s}", .{prompt});
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

test "blank credential detection trims whitespace" {
    try std.testing.expect(isBlankCredential(""));
    try std.testing.expect(isBlankCredential("   "));
    try std.testing.expect(isBlankCredential("\t \n"));
    try std.testing.expect(!isBlankCredential("sk-abc"));
    try std.testing.expect(!isBlankCredential("  sk-abc  "));
}

test {
    std.testing.refAllDecls(@This());
}
