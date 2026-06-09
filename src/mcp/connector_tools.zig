//! MCP `connector_test` tool: exercise a connector through its deterministic
//! local path. Extracted from handlers.zig — these routines hold no global
//! state, so they take only an allocator plus request fields.

const std = @import("std");
const abi = @import("abi");

/// Run the deterministic local path for one connector and return a one-line
/// status string. Unknown services return `error.UnknownConnector`.
pub fn runConnectorTest(allocator: std.mem.Allocator, service: []const u8, input: []const u8) ![]u8 {
    const connectors = abi.connectors;

    if (std.mem.eql(u8, service, "openai")) {
        var client = connectors.openai.Client.init(allocator, .{ .api_key = "mcp-local-key", .base_url = "https://api.openai.com" });
        defer client.deinit();
        const messages = try buildUserMessages(allocator, input);
        defer allocator.free(messages);
        var response = try client.chatCompletion(allocator, "gpt-local", messages);
        defer response.deinit(allocator);
        return try std.fmt.allocPrint(allocator, "connector=openai status={d} body={s}", .{ response.status, response.body });
    }

    if (std.mem.eql(u8, service, "anthropic")) {
        var client = connectors.anthropic.Client.init(allocator, .{ .api_key = "mcp-local-key", .base_url = "https://api.anthropic.com" });
        defer client.deinit();
        var response = try client.message(allocator, "claude-local", input, 256);
        defer response.deinit(allocator);
        return try std.fmt.allocPrint(allocator, "connector=anthropic status={d} body={s}", .{ response.status, response.body });
    }

    if (std.mem.eql(u8, service, "discord")) {
        var bot = connectors.discord.Bot.init(allocator, .{ .token = "mcp-local-token", .client_id = "123456789012345678" });
        defer bot.deinit();
        try bot.connect();
        const body = try bot.sendMessage(allocator, "234567890123456789", input);
        defer allocator.free(body);
        return try std.fmt.allocPrint(allocator, "connector=discord status=200 body={s}", .{body});
    }

    if (std.mem.eql(u8, service, "twilio")) {
        var client = connectors.twilio.Client.init(allocator, connectors.twilio.TwilioConfig.local());
        defer client.deinit();
        var response = try client.handleConversationRelayEvent(allocator, .{
            .kind = .user_transcript,
            .conversation_id = "CA" ++ "0123456789abcdef0123456789abcdef",
            .customer_id = "+15551234567",
            .transcript = input,
        }, "ABI local relay acknowledged.");
        defer response.deinit(allocator);
        return try std.fmt.allocPrint(allocator, "connector=twilio status=200 body={s}", .{response.text});
    }

    if (std.mem.eql(u8, service, "grok")) {
        var client = connectors.grok.Client.init(allocator, connectors.grok.grokConfig());
        defer client.deinit();
        const messages = try buildUserMessages(allocator, input);
        defer allocator.free(messages);
        var response = try client.chatCompletion(allocator, "grok-local", messages);
        defer response.deinit(allocator);
        return try std.fmt.allocPrint(allocator, "connector=grok status={d} body={s}", .{ response.status, response.body });
    }

    return error.UnknownConnector;
}

fn buildUserMessages(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "[{\"role\":\"user\",\"content\":");
    try abi.connectors.json.appendJsonString(&out, allocator, input);
    try out.appendSlice(allocator, "}]");
    return try out.toOwnedSlice(allocator);
}

test {
    std.testing.refAllDecls(@This());
}
