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
            try credentials.replaceOwnedString(allocator, &creds.openai_api_key, key);
        } else if (std.mem.eql(u8, service, "anthropic")) {
            const key = try readSecretLine(&stdin_reader, "Enter API key/token for anthropic: ");
            try credentials.replaceOwnedString(allocator, &creds.anthropic_api_key, key);
        } else if (std.mem.eql(u8, service, "discord")) {
            const key = try readSecretLine(&stdin_reader, "Enter API key/token for discord: ");
            try credentials.replaceOwnedString(allocator, &creds.discord_token, key);
        } else if (std.mem.eql(u8, service, "grok")) {
            const key = try readSecretLine(&stdin_reader, "Enter API key/token for grok: ");
            try credentials.replaceOwnedString(allocator, &creds.grok_api_key, key);
        } else if (std.mem.eql(u8, service, "twilio")) {
            const sid = try readSecretLine(&stdin_reader, "Enter Twilio Account SID: ");
            const token = try readSecretLine(&stdin_reader, "Enter Twilio Auth Token: ");
            try credentials.replaceOwnedString(allocator, &creds.twilio_account_sid, sid);
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
