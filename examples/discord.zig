//! Discord Bot Example
//!
//! Demonstrates how to use the ABI Discord connector to:
//! - Connect to Discord API
//! - Get bot information
//! - List guilds
//! - Send messages
//!
//! Environment Variables Required:
//! - DISCORD_BOT_TOKEN: Your Discord bot token
//! - DISCORD_CLIENT_ID: Your application client ID (optional)
//!
//! Usage: zig build run-discord

const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== ABI Discord Bot Example ===\n\n", .{});

    // Create Discord client from environment
    var client = abi.connectors.discord.createClient(allocator) catch |err| {
        switch (err) {
            abi.connectors.discord.DiscordError.MissingBotToken => {
                std.debug.print("Error: DISCORD_BOT_TOKEN environment variable not set.\n", .{});
                std.debug.print("\nTo use this example:\n", .{});
                std.debug.print("  1. Create a Discord application at https://discord.com/developers/applications\n", .{});
                std.debug.print("  2. Create a bot and copy the token\n", .{});
                std.debug.print("  3. Set DISCORD_BOT_TOKEN=<your-token>\n", .{});
                std.debug.print("  4. Invite the bot to a server with appropriate permissions\n", .{});
                return;
            },
            else => {
                std.debug.print("Failed to create Discord client: {t}\n", .{err});
                return;
            },
        }
    };
    defer client.deinit();

    std.debug.print("Discord client created successfully!\n", .{});
    std.debug.print("API Version: v{d}\n\n", .{client.config.api_version});

    // Get bot user information
    std.debug.print("--- Bot Information ---\n", .{});
    const user = client.getCurrentUser() catch |err| {
        std.debug.print("Failed to get bot user: {t}\n", .{err});
        return;
    };

    std.debug.print("Bot ID: {s}\n", .{user.id});
    std.debug.print("Username: {s}\n", .{user.username});
    std.debug.print("Discriminator: {s}\n", .{user.discriminator});
    if (user.global_name) |name| {
        std.debug.print("Global Name: {s}\n", .{name});
    }
    std.debug.print("Bot Account: {}\n", .{user.bot});
    std.debug.print("Public Flags: {d}\n\n", .{user.public_flags});

    // List guilds the bot is in
    std.debug.print("--- Guilds ---\n", .{});
    const guilds = client.getCurrentUserGuilds() catch |err| {
        std.debug.print("Failed to get guilds: {t}\n", .{err});
        return;
    };
    defer allocator.free(guilds);

    if (guilds.len == 0) {
        std.debug.print("Bot is not in any guilds.\n", .{});
        std.debug.print("Invite the bot to a server to see it here.\n\n", .{});
    } else {
        std.debug.print("Bot is in {d} guild(s):\n", .{guilds.len});
        for (guilds) |guild| {
            std.debug.print("  - {s} (ID: {s})\n", .{ guild.name, guild.id });
        }
        std.debug.print("\n", .{});
    }

    // Get gateway information
    std.debug.print("--- Gateway Info ---\n", .{});
    const gateway_url = client.getGateway() catch |err| {
        std.debug.print("Failed to get gateway: {t}\n", .{err});
        return;
    };
    defer allocator.free(gateway_url);
    std.debug.print("Gateway URL: {s}\n", .{gateway_url});

    // Get gateway bot info (includes shard recommendations)
    const gateway_bot = client.getGatewayBot() catch |err| {
        std.debug.print("Failed to get gateway bot info: {t}\n", .{err});
        return;
    };
    defer allocator.free(gateway_bot.url);

    std.debug.print("Recommended Shards: {d}\n", .{gateway_bot.shards});
    std.debug.print("Session Start Limit: {d}/{d}\n", .{
        gateway_bot.session_start_limit.remaining,
        gateway_bot.session_start_limit.total,
    });
    std.debug.print("\n", .{});

    // Show available intents
    std.debug.print("--- Gateway Intents ---\n", .{});
    const intents = client.config.intents;
    std.debug.print("Configured Intents: 0x{x}\n", .{intents});

    if (intents & abi.connectors.discord.GatewayIntent.GUILDS != 0)
        std.debug.print("  - GUILDS\n", .{});
    if (intents & abi.connectors.discord.GatewayIntent.GUILD_MESSAGES != 0)
        std.debug.print("  - GUILD_MESSAGES\n", .{});
    if (intents & abi.connectors.discord.GatewayIntent.DIRECT_MESSAGES != 0)
        std.debug.print("  - DIRECT_MESSAGES\n", .{});
    if (intents & abi.connectors.discord.GatewayIntent.MESSAGE_CONTENT != 0)
        std.debug.print("  - MESSAGE_CONTENT (privileged)\n", .{});
    if (intents & abi.connectors.discord.GatewayIntent.GUILD_MEMBERS != 0)
        std.debug.print("  - GUILD_MEMBERS (privileged)\n", .{});
    if (intents & abi.connectors.discord.GatewayIntent.GUILD_PRESENCES != 0)
        std.debug.print("  - GUILD_PRESENCES (privileged)\n", .{});

    std.debug.print("\n=== Example Complete ===\n", .{});
}
