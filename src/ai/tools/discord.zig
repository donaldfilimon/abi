//! Discord tools for AI agents.
//!
//! Provides Discord integration tools that can be used by AI agents to:
//! - Send messages to Discord channels
//! - Retrieve channel information
//! - List guilds and members
//! - Manage webhooks
//! - Execute Discord application commands

const std = @import("std");
const json = std.json;
const tool = @import("tool.zig");
const discord = @import("../../connectors/discord/mod.zig");
const json_utils = @import("../../shared/utils_combined.zig").json;

const Tool = tool.Tool;
const ToolResult = tool.ToolResult;
const Context = tool.Context;
const Parameter = tool.Parameter;
const ParameterType = tool.ParameterType;
const ToolExecutionError = tool.ToolExecutionError;

// ============================================================================
// Tool Definitions
// ============================================================================

/// Send a message to a Discord channel
pub const send_message_tool = Tool{
    .name = "discord_send_message",
    .description = "Send a message to a Discord channel. Requires channel_id and content.",
    .parameters = &[_]Parameter{
        .{
            .name = "channel_id",
            .type = .string,
            .required = true,
            .description = "The ID of the Discord channel to send the message to",
        },
        .{
            .name = "content",
            .type = .string,
            .required = true,
            .description = "The message content to send",
        },
        .{
            .name = "tts",
            .type = .boolean,
            .required = false,
            .description = "Whether this is a TTS (text-to-speech) message",
        },
    },
    .execute = &executeSendMessage,
};

/// Get information about a Discord channel
pub const get_channel_tool = Tool{
    .name = "discord_get_channel",
    .description = "Get information about a Discord channel by its ID.",
    .parameters = &[_]Parameter{
        .{
            .name = "channel_id",
            .type = .string,
            .required = true,
            .description = "The ID of the Discord channel",
        },
    },
    .execute = &executeGetChannel,
};

/// List guilds the bot is a member of
pub const list_guilds_tool = Tool{
    .name = "discord_list_guilds",
    .description = "List all Discord guilds (servers) the bot is a member of.",
    .parameters = &[_]Parameter{},
    .execute = &executeListGuilds,
};

/// Get bot user information
pub const get_bot_info_tool = Tool{
    .name = "discord_get_bot_info",
    .description = "Get information about the current Discord bot user.",
    .parameters = &[_]Parameter{},
    .execute = &executeGetBotInfo,
};

/// Execute a Discord webhook
pub const execute_webhook_tool = Tool{
    .name = "discord_execute_webhook",
    .description = "Execute a Discord webhook to send a message.",
    .parameters = &[_]Parameter{
        .{
            .name = "webhook_id",
            .type = .string,
            .required = true,
            .description = "The webhook ID",
        },
        .{
            .name = "webhook_token",
            .type = .string,
            .required = true,
            .description = "The webhook token",
        },
        .{
            .name = "content",
            .type = .string,
            .required = true,
            .description = "The message content to send",
        },
        .{
            .name = "username",
            .type = .string,
            .required = false,
            .description = "Override the webhook's username",
        },
    },
    .execute = &executeWebhook,
};

/// Add a reaction to a message
pub const add_reaction_tool = Tool{
    .name = "discord_add_reaction",
    .description = "Add a reaction emoji to a Discord message.",
    .parameters = &[_]Parameter{
        .{
            .name = "channel_id",
            .type = .string,
            .required = true,
            .description = "The ID of the channel containing the message",
        },
        .{
            .name = "message_id",
            .type = .string,
            .required = true,
            .description = "The ID of the message to react to",
        },
        .{
            .name = "emoji",
            .type = .string,
            .required = true,
            .description = "The emoji to react with (Unicode emoji or custom emoji format)",
        },
    },
    .execute = &executeAddReaction,
};

/// Get messages from a channel
pub const get_messages_tool = Tool{
    .name = "discord_get_messages",
    .description = "Get recent messages from a Discord channel.",
    .parameters = &[_]Parameter{
        .{
            .name = "channel_id",
            .type = .string,
            .required = true,
            .description = "The ID of the channel to get messages from",
        },
        .{
            .name = "limit",
            .type = .integer,
            .required = false,
            .description = "Maximum number of messages to return (1-100, default 50)",
        },
    },
    .execute = &executeGetMessages,
};

// ============================================================================
// Tool Execution Functions
// ============================================================================

fn executeSendMessage(ctx: *Context, args: json.Value) ToolExecutionError!ToolResult {
    const allocator = ctx.allocator;

    const obj = args.object;
    const channel_id = obj.get("channel_id") orelse {
        return ToolResult.fromError(allocator, "Missing required parameter: channel_id");
    };
    const content = obj.get("content") orelse {
        return ToolResult.fromError(allocator, "Missing required parameter: content");
    };

    if (channel_id != .string or content != .string) {
        return ToolResult.fromError(allocator, "Invalid parameter types");
    }

    var client = discord.createClient(allocator) catch {
        return ToolResult.fromError(allocator, "Failed to create Discord client");
    };
    defer client.deinit();

    const message = client.createMessage(channel_id.string, content.string) catch {
        return ToolResult.fromError(allocator, "Failed to send message");
    };

    const output = std.fmt.allocPrint(allocator, "Message sent successfully. ID: {s}", .{message.id}) catch {
        return ToolExecutionError.OutOfMemory;
    };

    return ToolResult.init(allocator, true, output);
}

fn executeGetChannel(ctx: *Context, args: json.Value) ToolExecutionError!ToolResult {
    const allocator = ctx.allocator;

    const obj = args.object;
    const channel_id = obj.get("channel_id") orelse {
        return ToolResult.fromError(allocator, "Missing required parameter: channel_id");
    };

    if (channel_id != .string) {
        return ToolResult.fromError(allocator, "Invalid parameter type for channel_id");
    }

    var client = discord.createClient(allocator) catch {
        return ToolResult.fromError(allocator, "Failed to create Discord client");
    };
    defer client.deinit();

    const channel = client.getChannel(channel_id.string) catch {
        return ToolResult.fromError(allocator, "Failed to get channel");
    };

    var output = std.ArrayListUnmanaged(u8){};
    errdefer output.deinit(allocator);

    output.print(allocator, "Channel ID: {s}\n", .{channel.id}) catch {
        return ToolExecutionError.OutOfMemory;
    };

    if (channel.name) |name| {
        output.print(allocator, "Name: {s}\n", .{name}) catch {
            return ToolExecutionError.OutOfMemory;
        };
    }

    output.print(allocator, "Type: {d}\n", .{channel.channel_type}) catch {
        return ToolExecutionError.OutOfMemory;
    };

    if (channel.guild_id) |gid| {
        output.print(allocator, "Guild ID: {s}\n", .{gid}) catch {
            return ToolExecutionError.OutOfMemory;
        };
    }

    const result = output.toOwnedSlice(allocator) catch {
        return ToolExecutionError.OutOfMemory;
    };

    return ToolResult.init(allocator, true, result);
}

fn executeListGuilds(ctx: *Context, _: json.Value) ToolExecutionError!ToolResult {
    const allocator = ctx.allocator;

    var client = discord.createClient(allocator) catch {
        return ToolResult.fromError(allocator, "Failed to create Discord client");
    };
    defer client.deinit();

    const guilds = client.getCurrentUserGuilds() catch {
        return ToolResult.fromError(allocator, "Failed to get guilds");
    };
    defer allocator.free(guilds);

    if (guilds.len == 0) {
        return ToolResult.init(allocator, true, "Bot is not a member of any guilds.");
    }

    var output = std.ArrayListUnmanaged(u8){};
    errdefer output.deinit(allocator);

    output.print(allocator, "Guilds ({d}):\n", .{guilds.len}) catch {
        return ToolExecutionError.OutOfMemory;
    };

    for (guilds) |guild| {
        output.print(allocator, "  - {s} (ID: {s})\n", .{ guild.name, guild.id }) catch {
            return ToolExecutionError.OutOfMemory;
        };
    }

    const result = output.toOwnedSlice(allocator) catch {
        return ToolExecutionError.OutOfMemory;
    };

    return ToolResult.init(allocator, true, result);
}

fn executeGetBotInfo(ctx: *Context, _: json.Value) ToolExecutionError!ToolResult {
    const allocator = ctx.allocator;

    var client = discord.createClient(allocator) catch {
        return ToolResult.fromError(allocator, "Failed to create Discord client");
    };
    defer client.deinit();

    const user = client.getCurrentUser() catch {
        return ToolResult.fromError(allocator, "Failed to get bot user info");
    };

    var output = std.ArrayListUnmanaged(u8){};
    errdefer output.deinit(allocator);

    output.print(allocator, "Bot User Information:\n", .{}) catch {
        return ToolExecutionError.OutOfMemory;
    };
    output.print(allocator, "  ID: {s}\n", .{user.id}) catch {
        return ToolExecutionError.OutOfMemory;
    };
    output.print(allocator, "  Username: {s}\n", .{user.username}) catch {
        return ToolExecutionError.OutOfMemory;
    };
    output.print(allocator, "  Discriminator: {s}\n", .{user.discriminator}) catch {
        return ToolExecutionError.OutOfMemory;
    };
    if (user.global_name) |name| {
        output.print(allocator, "  Global Name: {s}\n", .{name}) catch {
            return ToolExecutionError.OutOfMemory;
        };
    }
    output.print(allocator, "  Bot: {}\n", .{user.bot}) catch {
        return ToolExecutionError.OutOfMemory;
    };

    const result = output.toOwnedSlice(allocator) catch {
        return ToolExecutionError.OutOfMemory;
    };

    return ToolResult.init(allocator, true, result);
}

fn executeWebhook(ctx: *Context, args: json.Value) ToolExecutionError!ToolResult {
    const allocator = ctx.allocator;

    const obj = args.object;
    const webhook_id = obj.get("webhook_id") orelse {
        return ToolResult.fromError(allocator, "Missing required parameter: webhook_id");
    };
    const webhook_token = obj.get("webhook_token") orelse {
        return ToolResult.fromError(allocator, "Missing required parameter: webhook_token");
    };
    const content = obj.get("content") orelse {
        return ToolResult.fromError(allocator, "Missing required parameter: content");
    };

    if (webhook_id != .string or webhook_token != .string or content != .string) {
        return ToolResult.fromError(allocator, "Invalid parameter types");
    }

    var client = discord.createClient(allocator) catch {
        return ToolResult.fromError(allocator, "Failed to create Discord client");
    };
    defer client.deinit();

    client.executeWebhook(webhook_id.string, webhook_token.string, content.string) catch {
        return ToolResult.fromError(allocator, "Failed to execute webhook");
    };

    return ToolResult.init(allocator, true, "Webhook executed successfully.");
}

fn executeAddReaction(ctx: *Context, args: json.Value) ToolExecutionError!ToolResult {
    const allocator = ctx.allocator;

    const obj = args.object;
    const channel_id = obj.get("channel_id") orelse {
        return ToolResult.fromError(allocator, "Missing required parameter: channel_id");
    };
    const message_id = obj.get("message_id") orelse {
        return ToolResult.fromError(allocator, "Missing required parameter: message_id");
    };
    const emoji = obj.get("emoji") orelse {
        return ToolResult.fromError(allocator, "Missing required parameter: emoji");
    };

    if (channel_id != .string or message_id != .string or emoji != .string) {
        return ToolResult.fromError(allocator, "Invalid parameter types");
    }

    var client = discord.createClient(allocator) catch {
        return ToolResult.fromError(allocator, "Failed to create Discord client");
    };
    defer client.deinit();

    client.createReaction(channel_id.string, message_id.string, emoji.string) catch {
        return ToolResult.fromError(allocator, "Failed to add reaction");
    };

    return ToolResult.init(allocator, true, "Reaction added successfully.");
}

fn executeGetMessages(ctx: *Context, args: json.Value) ToolExecutionError!ToolResult {
    const allocator = ctx.allocator;

    const obj = args.object;
    const channel_id = obj.get("channel_id") orelse {
        return ToolResult.fromError(allocator, "Missing required parameter: channel_id");
    };

    if (channel_id != .string) {
        return ToolResult.fromError(allocator, "Invalid parameter type for channel_id");
    }

    var limit: ?u8 = 50;
    if (obj.get("limit")) |limit_val| {
        if (limit_val == .integer) {
            const l = limit_val.integer;
            if (l > 0 and l <= 100) {
                limit = @intCast(l);
            }
        }
    }

    var client = discord.createClient(allocator) catch {
        return ToolResult.fromError(allocator, "Failed to create Discord client");
    };
    defer client.deinit();

    const messages = client.getChannelMessages(channel_id.string, limit) catch {
        return ToolResult.fromError(allocator, "Failed to get messages");
    };
    defer allocator.free(messages);

    if (messages.len == 0) {
        return ToolResult.init(allocator, true, "No messages found in channel.");
    }

    var output = std.ArrayListUnmanaged(u8){};
    errdefer output.deinit(allocator);

    output.print(allocator, "Messages ({d}):\n", .{messages.len}) catch {
        return ToolExecutionError.OutOfMemory;
    };

    for (messages) |msg| {
        output.print(allocator, "  [{s}] {s}: {s}\n", .{ msg.id, msg.author.username, msg.content }) catch {
            return ToolExecutionError.OutOfMemory;
        };
    }

    const result = output.toOwnedSlice(allocator) catch {
        return ToolExecutionError.OutOfMemory;
    };

    return ToolResult.init(allocator, true, result);
}

// ============================================================================
// Tool Registry Integration
// ============================================================================

/// All available Discord tools
pub const all_tools = [_]*const Tool{
    &send_message_tool,
    &get_channel_tool,
    &list_guilds_tool,
    &get_bot_info_tool,
    &execute_webhook_tool,
    &add_reaction_tool,
    &get_messages_tool,
};

/// Register all Discord tools with a tool registry
pub fn registerAll(registry: *tool.ToolRegistry) !void {
    for (all_tools) |t| {
        try registry.register(t);
    }
}

// ============================================================================
// Tests
// ============================================================================

test "discord tool definitions" {
    // Verify tool structures are valid
    try std.testing.expectEqualStrings("discord_send_message", send_message_tool.name);
    try std.testing.expectEqualStrings("discord_get_channel", get_channel_tool.name);
    try std.testing.expectEqualStrings("discord_list_guilds", list_guilds_tool.name);
    try std.testing.expectEqualStrings("discord_get_bot_info", get_bot_info_tool.name);
    try std.testing.expectEqualStrings("discord_execute_webhook", execute_webhook_tool.name);
    try std.testing.expectEqualStrings("discord_add_reaction", add_reaction_tool.name);
    try std.testing.expectEqualStrings("discord_get_messages", get_messages_tool.name);
}

test "discord tool registry integration" {
    const allocator = std.testing.allocator;

    var registry = tool.ToolRegistry.init(allocator);
    defer registry.deinit();

    try registerAll(&registry);

    // Verify all tools are registered
    try std.testing.expect(registry.get("discord_send_message") != null);
    try std.testing.expect(registry.get("discord_get_channel") != null);
    try std.testing.expect(registry.get("discord_list_guilds") != null);
    try std.testing.expect(registry.get("discord_get_bot_info") != null);
}
