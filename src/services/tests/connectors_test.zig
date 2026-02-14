//! Connector module tests
//!
//! Tests for the API connector implementations including Discord, OpenAI,
//! HuggingFace, and Ollama connectors.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

// ============================================================================
// Discord Connector Tests
// ============================================================================

test "discord config lifecycle" {
    const allocator = std.testing.allocator;

    var config = abi.connectors.discord.Config{
        .bot_token = try allocator.dupe(u8, "test_token_123"),
        .client_id = try allocator.dupe(u8, "123456789012345678"),
        .client_secret = null,
        .public_key = null,
        .api_version = 10,
        .timeout_ms = 30_000,
        .intents = abi.connectors.discord.GatewayIntent.ALL_UNPRIVILEGED,
    };
    defer config.deinit(allocator);

    try std.testing.expectEqualStrings("test_token_123", config.bot_token);
    try std.testing.expectEqualStrings("123456789012345678", config.client_id.?);
    try std.testing.expectEqual(@as(u8, 10), config.api_version);
}

test "discord gateway intents calculation" {
    const intents = abi.connectors.discord.GatewayIntent.GUILDS |
        abi.connectors.discord.GatewayIntent.GUILD_MESSAGES |
        abi.connectors.discord.GatewayIntent.MESSAGE_CONTENT;

    try std.testing.expect(intents & abi.connectors.discord.GatewayIntent.GUILDS != 0);
    try std.testing.expect(intents & abi.connectors.discord.GatewayIntent.GUILD_MESSAGES != 0);
    try std.testing.expect(intents & abi.connectors.discord.GatewayIntent.MESSAGE_CONTENT != 0);
    try std.testing.expect(intents & abi.connectors.discord.GatewayIntent.GUILD_MEMBERS == 0);
    try std.testing.expect(intents & abi.connectors.discord.GatewayIntent.GUILD_PRESENCES == 0);
}

test "discord privileged intents" {
    const privileged = abi.connectors.discord.GatewayIntent.ALL_PRIVILEGED;

    // Privileged intents should include GUILD_MEMBERS, GUILD_PRESENCES, MESSAGE_CONTENT
    try std.testing.expect(privileged & abi.connectors.discord.GatewayIntent.GUILD_MEMBERS != 0);
    try std.testing.expect(privileged & abi.connectors.discord.GatewayIntent.GUILD_PRESENCES != 0);
    try std.testing.expect(privileged & abi.connectors.discord.GatewayIntent.MESSAGE_CONTENT != 0);

    // But not unprivileged ones
    try std.testing.expect(privileged & abi.connectors.discord.GatewayIntent.GUILDS == 0);
}

test "discord permission utilities" {
    const perms = abi.connectors.discord.Permission.SEND_MESSAGES |
        abi.connectors.discord.Permission.VIEW_CHANNEL |
        abi.connectors.discord.Permission.ADMINISTRATOR;

    try std.testing.expect(abi.connectors.discord.hasPermission(perms, abi.connectors.discord.Permission.SEND_MESSAGES));
    try std.testing.expect(abi.connectors.discord.hasPermission(perms, abi.connectors.discord.Permission.VIEW_CHANNEL));
    try std.testing.expect(abi.connectors.discord.hasPermission(perms, abi.connectors.discord.Permission.ADMINISTRATOR));
    try std.testing.expect(!abi.connectors.discord.hasPermission(perms, abi.connectors.discord.Permission.MANAGE_GUILD));
    try std.testing.expect(!abi.connectors.discord.hasPermission(perms, abi.connectors.discord.Permission.BAN_MEMBERS));
}

test "discord calculate permissions" {
    const permissions = [_]u64{
        abi.connectors.discord.Permission.SEND_MESSAGES,
        abi.connectors.discord.Permission.VIEW_CHANNEL,
        abi.connectors.discord.Permission.EMBED_LINKS,
    };

    const combined = abi.connectors.discord.calculatePermissions(&permissions);

    try std.testing.expect(abi.connectors.discord.hasPermission(combined, abi.connectors.discord.Permission.SEND_MESSAGES));
    try std.testing.expect(abi.connectors.discord.hasPermission(combined, abi.connectors.discord.Permission.VIEW_CHANNEL));
    try std.testing.expect(abi.connectors.discord.hasPermission(combined, abi.connectors.discord.Permission.EMBED_LINKS));
    try std.testing.expect(!abi.connectors.discord.hasPermission(combined, abi.connectors.discord.Permission.MANAGE_MESSAGES));
}

test "discord gateway opcode values" {
    try std.testing.expectEqual(@as(u8, 0), @intFromEnum(abi.connectors.discord.GatewayOpcode.DISPATCH));
    try std.testing.expectEqual(@as(u8, 1), @intFromEnum(abi.connectors.discord.GatewayOpcode.HEARTBEAT));
    try std.testing.expectEqual(@as(u8, 2), @intFromEnum(abi.connectors.discord.GatewayOpcode.IDENTIFY));
    try std.testing.expectEqual(@as(u8, 10), @intFromEnum(abi.connectors.discord.GatewayOpcode.HELLO));
    try std.testing.expectEqual(@as(u8, 11), @intFromEnum(abi.connectors.discord.GatewayOpcode.HEARTBEAT_ACK));
}

test "discord channel type values" {
    try std.testing.expectEqual(@as(u8, 0), @intFromEnum(abi.connectors.discord.ChannelType.GUILD_TEXT));
    try std.testing.expectEqual(@as(u8, 1), @intFromEnum(abi.connectors.discord.ChannelType.DM));
    try std.testing.expectEqual(@as(u8, 2), @intFromEnum(abi.connectors.discord.ChannelType.GUILD_VOICE));
    try std.testing.expectEqual(@as(u8, 4), @intFromEnum(abi.connectors.discord.ChannelType.GUILD_CATEGORY));
    try std.testing.expectEqual(@as(u8, 15), @intFromEnum(abi.connectors.discord.ChannelType.GUILD_FORUM));
}

test "discord interaction type values" {
    try std.testing.expectEqual(@as(u8, 1), @intFromEnum(abi.connectors.discord.InteractionType.PING));
    try std.testing.expectEqual(@as(u8, 2), @intFromEnum(abi.connectors.discord.InteractionType.APPLICATION_COMMAND));
    try std.testing.expectEqual(@as(u8, 3), @intFromEnum(abi.connectors.discord.InteractionType.MESSAGE_COMPONENT));
    try std.testing.expectEqual(@as(u8, 5), @intFromEnum(abi.connectors.discord.InteractionType.MODAL_SUBMIT));
}

test "discord component type values" {
    try std.testing.expectEqual(@as(u8, 1), @intFromEnum(abi.connectors.discord.ComponentType.ACTION_ROW));
    try std.testing.expectEqual(@as(u8, 2), @intFromEnum(abi.connectors.discord.ComponentType.BUTTON));
    try std.testing.expectEqual(@as(u8, 3), @intFromEnum(abi.connectors.discord.ComponentType.STRING_SELECT));
    try std.testing.expectEqual(@as(u8, 4), @intFromEnum(abi.connectors.discord.ComponentType.TEXT_INPUT));
}

// ============================================================================
// JSON Utilities Tests
// ============================================================================

test "json escape basic strings" {
    const allocator = std.testing.allocator;

    const escaped = try abi.shared.utils.json.escapeJsonContent(allocator, "hello world");
    defer allocator.free(escaped);

    try std.testing.expectEqualStrings("hello world", escaped);
}

test "json escape special characters" {
    const allocator = std.testing.allocator;

    const escaped = try abi.shared.utils.json.escapeJsonContent(allocator, "line1\nline2\ttab\"quote\\backslash");
    defer allocator.free(escaped);

    try std.testing.expectEqualStrings("line1\\nline2\\ttab\\\"quote\\\\backslash", escaped);
}

test "json escape with quotes" {
    const allocator = std.testing.allocator;

    const escaped = try abi.shared.utils.json.escapeString(allocator, "hello");
    defer allocator.free(escaped);

    try std.testing.expectEqualStrings("\"hello\"", escaped);
}

test "json parse string field" {
    const allocator = std.testing.allocator;

    const json_text = "{\"name\":\"test\",\"value\":42}";
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_text, .{});
    defer parsed.deinit();

    const object = try abi.shared.utils.json.getRequiredObject(parsed.value);
    const name = try abi.shared.utils.json.parseStringField(object, "name", allocator);
    try std.testing.expectEqualStrings("test", name);

    const value = try abi.shared.utils.json.parseIntField(object, "value");
    try std.testing.expectEqual(@as(i64, 42), value);
}

test "json parse optional fields" {
    const allocator = std.testing.allocator;

    const json_text = "{\"required\":\"present\"}";
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_text, .{});
    defer parsed.deinit();

    const object = try abi.shared.utils.json.getRequiredObject(parsed.value);

    const required = try abi.shared.utils.json.parseStringField(object, "required", allocator);
    try std.testing.expectEqualStrings("present", required);

    const missing = abi.shared.utils.json.parseOptionalStringField(object, "missing", allocator) catch null;
    try std.testing.expect(missing == null);
}

test "json parse bool field" {
    const allocator = std.testing.allocator;

    const json_text = "{\"enabled\":true,\"disabled\":false}";
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_text, .{});
    defer parsed.deinit();

    const object = try abi.shared.utils.json.getRequiredObject(parsed.value);

    const enabled = try abi.shared.utils.json.parseBoolField(object, "enabled");
    try std.testing.expect(enabled);

    const disabled = try abi.shared.utils.json.parseBoolField(object, "disabled");
    try std.testing.expect(!disabled);
}

test "json parse number fields" {
    const allocator = std.testing.allocator;

    const json_text = "{\"int_val\":123,\"float_val\":3.14}";
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_text, .{});
    defer parsed.deinit();

    const object = try abi.shared.utils.json.getRequiredObject(parsed.value);

    const int_val = try abi.shared.utils.json.parseIntField(object, "int_val");
    try std.testing.expectEqual(@as(i64, 123), int_val);

    const float_val = try abi.shared.utils.json.parseNumberField(object, "float_val");
    try std.testing.expectApproxEqAbs(@as(f64, 3.14), float_val, 0.001);
}

// ============================================================================
// AI Tools Tests
// ============================================================================

test "discord tools registration" {
    // Skip when AI is disabled - stubs return null for registry.get()
    if (!build_options.enable_ai) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var registry = abi.ai.ToolRegistry.init(allocator);
    defer registry.deinit();

    try abi.ai.registerDiscordTools(&registry);

    // Verify all Discord tools are registered
    try std.testing.expect(registry.get("discord_send_message") != null);
    try std.testing.expect(registry.get("discord_get_channel") != null);
    try std.testing.expect(registry.get("discord_list_guilds") != null);
    try std.testing.expect(registry.get("discord_get_bot_info") != null);
    try std.testing.expect(registry.get("discord_execute_webhook") != null);
    try std.testing.expect(registry.get("discord_add_reaction") != null);
    try std.testing.expect(registry.get("discord_get_messages") != null);
}

test "discord tool definitions are valid" {
    // Skip when AI is disabled - stubs have empty parameters
    if (!build_options.enable_ai) return error.SkipZigTest;

    // Verify send_message tool
    const send_tool = abi.ai.DiscordTools.send_message_tool;
    try std.testing.expectEqualStrings("discord_send_message", send_tool.name);
    try std.testing.expect(send_tool.parameters.len >= 2);

    // Verify get_channel tool
    const channel_tool = abi.ai.DiscordTools.get_channel_tool;
    try std.testing.expectEqualStrings("discord_get_channel", channel_tool.name);
    try std.testing.expect(channel_tool.parameters.len >= 1);

    // Verify list_guilds tool
    const guilds_tool = abi.ai.DiscordTools.list_guilds_tool;
    try std.testing.expectEqualStrings("discord_list_guilds", guilds_tool.name);
}
