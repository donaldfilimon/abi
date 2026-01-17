//! Discord CLI command.
//!
//! Provides commands for interacting with Discord API:
//! - Check bot status and information
//! - List guilds the bot is in
//! - Send messages to channels
//! - Manage application commands
//! - Execute webhooks

const std = @import("std");
const abi = @import("abi");
const utils = @import("../utils/mod.zig");

const discord = abi.connectors.discord;

/// Run the discord command with the provided arguments.
pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    // Check if web feature is enabled (Discord requires web/HTTP support)
    if (!abi.web.isEnabled()) {
        std.debug.print("Error: Web feature is disabled.\n", .{});
        std.debug.print("Rebuild with: zig build -Denable-web=true\n", .{});
        return;
    }

    if (args.len == 0) {
        try printStatus(allocator);
        return;
    }

    const command = std.mem.sliceTo(args[0], 0);
    const remaining_args = if (args.len > 1) args[1..] else &[_][:0]const u8{};

    if (utils.args.matchesAny(command, &[_][]const u8{ "help", "--help", "-h" })) {
        printHelp();
        return;
    }

    if (std.mem.eql(u8, command, "status")) {
        try printStatus(allocator);
        return;
    }

    if (std.mem.eql(u8, command, "info")) {
        try printBotInfo(allocator);
        return;
    }

    if (std.mem.eql(u8, command, "guilds")) {
        try listGuilds(allocator);
        return;
    }

    if (std.mem.eql(u8, command, "send")) {
        try sendMessage(allocator, remaining_args);
        return;
    }

    if (std.mem.eql(u8, command, "commands")) {
        try manageCommands(allocator, remaining_args);
        return;
    }

    if (std.mem.eql(u8, command, "webhook")) {
        try executeWebhook(allocator, remaining_args);
        return;
    }

    if (std.mem.eql(u8, command, "channel")) {
        try channelInfo(allocator, remaining_args);
        return;
    }

    std.debug.print("Unknown discord command: {s}\n", .{command});
    printHelp();
}

/// Print a short discord summary for system-info.
pub fn printSummary(allocator: std.mem.Allocator) void {
    const config = abi.connectors.tryLoadDiscord(allocator) catch {
        std.debug.print("  Discord: error loading config\n", .{});
        return;
    };

    if (config) |cfg| {
        var mutable_cfg = cfg;
        defer mutable_cfg.deinit(allocator);
        std.debug.print("  Discord: configured (token: {s}...)\n", .{
            if (cfg.bot_token.len > 8) cfg.bot_token[0..8] else cfg.bot_token,
        });
    } else {
        std.debug.print("  Discord: not configured (set DISCORD_BOT_TOKEN)\n", .{});
    }
}

fn printHelp() void {
    const text =
        \\Usage: abi discord <command> [options]
        \\
        \\Commands:
        \\  status                        Show Discord configuration status
        \\  info                          Get bot user information
        \\  guilds                        List guilds the bot is in
        \\  send <channel_id> <message>   Send a message to a channel
        \\  channel <channel_id>          Get channel information
        \\  commands list [guild_id]      List application commands
        \\  commands create <name> <desc> Create a global command
        \\  webhook <url> <message>       Execute a webhook
        \\
        \\Environment Variables:
        \\  DISCORD_BOT_TOKEN             Bot authentication token (required)
        \\  DISCORD_CLIENT_ID             Application client ID
        \\  DISCORD_CLIENT_SECRET         OAuth2 client secret
        \\  DISCORD_PUBLIC_KEY            Interaction verification key
        \\
    ;
    std.debug.print("{s}", .{text});
}

fn printStatus(allocator: std.mem.Allocator) !void {
    const config = abi.connectors.tryLoadDiscord(allocator) catch |err| {
        std.debug.print("Discord: error loading config: {}\n", .{err});
        return;
    };

    if (config) |cfg| {
        var mutable_cfg = cfg;
        defer mutable_cfg.deinit(allocator);

        std.debug.print("Discord Configuration:\n", .{});
        std.debug.print("  Bot Token: {s}... (configured)\n", .{
            if (cfg.bot_token.len > 8) cfg.bot_token[0..8] else cfg.bot_token,
        });
        std.debug.print("  API Version: v{d}\n", .{cfg.api_version});
        std.debug.print("  Timeout: {d}ms\n", .{cfg.timeout_ms});

        if (cfg.client_id) |id| {
            std.debug.print("  Client ID: {s}\n", .{id});
        } else {
            std.debug.print("  Client ID: not set\n", .{});
        }

        if (cfg.public_key != null) {
            std.debug.print("  Public Key: configured\n", .{});
        } else {
            std.debug.print("  Public Key: not set\n", .{});
        }

        std.debug.print("  Gateway Intents: 0x{x}\n", .{cfg.intents});
    } else {
        std.debug.print("Discord: not configured\n", .{});
        std.debug.print("\nTo configure Discord, set the following environment variables:\n", .{});
        std.debug.print("  DISCORD_BOT_TOKEN - Your bot token from Discord Developer Portal\n", .{});
    }
}

fn printBotInfo(allocator: std.mem.Allocator) !void {
    var client = discord.createClient(allocator) catch |err| {
        std.debug.print("Failed to create Discord client: {}\n", .{err});
        return;
    };
    defer client.deinit();

    const user = client.getCurrentUser() catch |err| {
        std.debug.print("Failed to get bot info: {}\n", .{err});
        return;
    };

    std.debug.print("Bot Information:\n", .{});
    std.debug.print("  ID: {s}\n", .{user.id});
    std.debug.print("  Username: {s}\n", .{user.username});
    std.debug.print("  Discriminator: {s}\n", .{user.discriminator});
    if (user.global_name) |name| {
        std.debug.print("  Global Name: {s}\n", .{name});
    }
    std.debug.print("  Bot: {}\n", .{user.bot});
    std.debug.print("  Flags: {d}\n", .{user.public_flags});
}

fn listGuilds(allocator: std.mem.Allocator) !void {
    var client = discord.createClient(allocator) catch |err| {
        std.debug.print("Failed to create Discord client: {}\n", .{err});
        return;
    };
    defer client.deinit();

    const guilds = client.getCurrentUserGuilds() catch |err| {
        std.debug.print("Failed to get guilds: {}\n", .{err});
        return;
    };
    defer allocator.free(guilds);

    if (guilds.len == 0) {
        std.debug.print("Bot is not in any guilds.\n", .{});
        return;
    }

    std.debug.print("Guilds ({d}):\n", .{guilds.len});
    for (guilds) |guild| {
        std.debug.print("  {s} (ID: {s})\n", .{ guild.name, guild.id });
    }
}

fn sendMessage(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (args.len < 2) {
        std.debug.print("Usage: abi discord send <channel_id> <message>\n", .{});
        return;
    }

    const channel_id = std.mem.sliceTo(args[0], 0);

    // Join remaining args as message
    var message = std.ArrayListUnmanaged(u8){};
    defer message.deinit(allocator);

    for (args[1..], 0..) |arg, i| {
        if (i > 0) try message.append(allocator, ' ');
        try message.appendSlice(allocator, std.mem.sliceTo(arg, 0));
    }

    var client = discord.createClient(allocator) catch |err| {
        std.debug.print("Failed to create Discord client: {}\n", .{err});
        return;
    };
    defer client.deinit();

    const sent_message = client.createMessage(channel_id, message.items) catch |err| {
        std.debug.print("Failed to send message: {}\n", .{err});
        return;
    };

    std.debug.print("Message sent successfully!\n", .{});
    std.debug.print("  ID: {s}\n", .{sent_message.id});
    std.debug.print("  Channel: {s}\n", .{sent_message.channel_id});
    std.debug.print("  Content: {s}\n", .{sent_message.content});
}

fn channelInfo(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (args.len < 1) {
        std.debug.print("Usage: abi discord channel <channel_id>\n", .{});
        return;
    }

    const channel_id = std.mem.sliceTo(args[0], 0);

    var client = discord.createClient(allocator) catch |err| {
        std.debug.print("Failed to create Discord client: {}\n", .{err});
        return;
    };
    defer client.deinit();

    const channel = client.getChannel(channel_id) catch |err| {
        std.debug.print("Failed to get channel: {}\n", .{err});
        return;
    };

    std.debug.print("Channel Information:\n", .{});
    std.debug.print("  ID: {s}\n", .{channel.id});
    if (channel.name) |name| {
        std.debug.print("  Name: {s}\n", .{name});
    }
    std.debug.print("  Type: {d}\n", .{channel.channel_type});
    if (channel.guild_id) |gid| {
        std.debug.print("  Guild ID: {s}\n", .{gid});
    }
}

fn manageCommands(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (args.len < 1) {
        std.debug.print("Usage: abi discord commands <list|create|delete> [options]\n", .{});
        return;
    }

    const subcommand = std.mem.sliceTo(args[0], 0);
    const remaining = if (args.len > 1) args[1..] else &[_][:0]const u8{};

    var client = discord.createClient(allocator) catch |err| {
        std.debug.print("Failed to create Discord client: {}\n", .{err});
        return;
    };
    defer client.deinit();

    if (std.mem.eql(u8, subcommand, "list")) {
        const app_id = client.config.client_id orelse {
            std.debug.print("Error: DISCORD_CLIENT_ID not configured.\n", .{});
            std.debug.print("Set DISCORD_CLIENT_ID environment variable to your application ID.\n", .{});
            return;
        };

        const guild_id: ?[]const u8 = if (remaining.len > 0)
            std.mem.sliceTo(remaining[0], 0)
        else
            null;

        const commands = if (guild_id) |gid|
            client.getGuildApplicationCommands(app_id, gid) catch |err| {
                std.debug.print("Failed to get guild commands: {}\n", .{err});
                return;
            }
        else
            client.getGlobalApplicationCommands(app_id) catch |err| {
                std.debug.print("Failed to get global commands: {}\n", .{err});
                return;
            };
        defer allocator.free(commands);

        if (commands.len == 0) {
            std.debug.print("No application commands found.\n", .{});
            return;
        }

        std.debug.print("Application Commands ({d}):\n", .{commands.len});
        for (commands) |cmd| {
            std.debug.print("  /{s} - {s} (ID: {s})\n", .{ cmd.name, cmd.description, cmd.id });
        }
        return;
    }

    if (std.mem.eql(u8, subcommand, "create")) {
        const app_id = client.config.client_id orelse {
            std.debug.print("Error: DISCORD_CLIENT_ID not configured.\n", .{});
            return;
        };

        if (remaining.len < 2) {
            std.debug.print("Usage: abi discord commands create <name> <description>\n", .{});
            return;
        }

        const name = std.mem.sliceTo(remaining[0], 0);
        const description = std.mem.sliceTo(remaining[1], 0);

        const cmd = client.createGlobalApplicationCommand(app_id, name, description, &.{}) catch |err| {
            std.debug.print("Failed to create command: {}\n", .{err});
            return;
        };

        std.debug.print("Command created successfully!\n", .{});
        std.debug.print("  Name: /{s}\n", .{cmd.name});
        std.debug.print("  ID: {s}\n", .{cmd.id});
        return;
    }

    if (std.mem.eql(u8, subcommand, "delete")) {
        const app_id = client.config.client_id orelse {
            std.debug.print("Error: DISCORD_CLIENT_ID not configured.\n", .{});
            return;
        };

        if (remaining.len < 1) {
            std.debug.print("Usage: abi discord commands delete <command_id>\n", .{});
            return;
        }

        const command_id = std.mem.sliceTo(remaining[0], 0);
        client.deleteGlobalApplicationCommand(app_id, command_id) catch |err| {
            std.debug.print("Failed to delete command: {}\n", .{err});
            return;
        };

        std.debug.print("Command deleted successfully.\n", .{});
        return;
    }

    std.debug.print("Unknown commands subcommand: {s}\n", .{subcommand});
}

fn executeWebhook(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (args.len < 2) {
        std.debug.print("Usage: abi discord webhook <webhook_url> <message>\n", .{});
        std.debug.print("\nWebhook URL format: https://discord.com/api/webhooks/<id>/<token>\n", .{});
        return;
    }

    const webhook_url = std.mem.sliceTo(args[0], 0);

    // Join remaining args as message
    var message = std.ArrayListUnmanaged(u8){};
    defer message.deinit(allocator);

    for (args[1..], 0..) |arg, i| {
        if (i > 0) try message.append(allocator, ' ');
        try message.appendSlice(allocator, std.mem.sliceTo(arg, 0));
    }

    // Parse webhook URL to extract ID and token
    // Format: https://discord.com/api/webhooks/<id>/<token>
    const webhooks_prefix = "/api/webhooks/";
    const prefix_pos = std.mem.indexOf(u8, webhook_url, webhooks_prefix) orelse {
        std.debug.print("Invalid webhook URL format.\n", .{});
        return;
    };

    const path_start = prefix_pos + webhooks_prefix.len;
    const path = webhook_url[path_start..];

    const slash_pos = std.mem.indexOf(u8, path, "/") orelse {
        std.debug.print("Invalid webhook URL format.\n", .{});
        return;
    };

    const webhook_id = path[0..slash_pos];
    const webhook_token = path[slash_pos + 1 ..];

    var client = discord.createClient(allocator) catch |err| {
        std.debug.print("Failed to create Discord client: {}\n", .{err});
        return;
    };
    defer client.deinit();

    client.executeWebhook(webhook_id, webhook_token, message.items) catch |err| {
        std.debug.print("Failed to execute webhook: {}\n", .{err});
        return;
    };

    std.debug.print("Webhook executed successfully!\n", .{});
}
