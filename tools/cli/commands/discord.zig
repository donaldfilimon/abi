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
const command_mod = @import("../command.zig");
const utils = @import("../utils/mod.zig");

const discord = abi.connectors.discord;

// Wrapper functions for comptime children dispatch
fn wrapDcStatus(allocator: std.mem.Allocator, _: []const [:0]const u8) !void {
    try printStatus(allocator);
}
fn wrapDcInfo(allocator: std.mem.Allocator, _: []const [:0]const u8) !void {
    try printBotInfo(allocator);
}
fn wrapDcGuilds(allocator: std.mem.Allocator, _: []const [:0]const u8) !void {
    try listGuilds(allocator);
}
fn wrapDcSend(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    try sendMessage(allocator, args);
}
fn wrapDcCommands(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var parser = utils.args.ArgParser.init(allocator, args);
    try manageCommands(allocator, &parser);
}
fn wrapDcWebhook(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    try executeWebhook(allocator, args);
}
fn wrapDcChannel(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    try channelInfo(allocator, args);
}

pub const meta: command_mod.Meta = .{
    .name = "discord",
    .description = "Discord bot operations (status, guilds, send, commands)",
    .subcommands = &.{ "status", "info", "guilds", "send", "commands", "webhook", "channel", "help" },
    .children = &.{
        .{ .name = "status", .description = "Show configuration status", .handler = .{ .basic = wrapDcStatus } },
        .{ .name = "info", .description = "Get bot user information", .handler = .{ .basic = wrapDcInfo } },
        .{ .name = "guilds", .description = "List guilds the bot is in", .handler = .{ .basic = wrapDcGuilds } },
        .{ .name = "send", .description = "Send message to a channel", .handler = .{ .basic = wrapDcSend } },
        .{ .name = "commands", .description = "Manage application commands", .handler = .{ .basic = wrapDcCommands } },
        .{ .name = "webhook", .description = "Execute a webhook", .handler = .{ .basic = wrapDcWebhook } },
        .{ .name = "channel", .description = "Get channel information", .handler = .{ .basic = wrapDcChannel } },
    },
};

const discord_subcommands = [_][]const u8{
    "status", "info", "guilds", "send", "commands", "webhook", "channel", "help",
};

/// Run the discord command with the provided arguments.
pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    // Check if web feature is enabled (Discord requires web/HTTP support)
    if (!abi.web.isEnabled()) {
        utils.output.printError("Web feature is disabled.", .{});
        utils.output.printInfo("Rebuild with: zig build -Denable-web=true", .{});
        return;
    }
    if (args.len == 0) {
        // Default action: show status
        try printStatus(allocator);
        return;
    }
    const cmd = std.mem.sliceTo(args[0], 0);
    if (utils.args.matchesAny(cmd, &.{ "--help", "-h", "help" })) {
        printHelp(allocator);
        return;
    }
    // Unknown subcommand
    utils.output.printError("Unknown discord command: {s}", .{cmd});
    if (utils.args.suggestCommand(cmd, &discord_subcommands)) |suggestion| {
        std.debug.print("Did you mean: {s}\n", .{suggestion});
    }
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

fn printHelp(allocator: std.mem.Allocator) void {
    var builder = utils.help.HelpBuilder.init(allocator);
    defer builder.deinit();

    _ = builder
        .usage("abi discord", "<command> [options]")
        .description("Discord API interaction commands.")
        .section("Commands")
        .subcommand(.{ .name = "status", .description = "Show configuration status" })
        .subcommand(.{ .name = "info", .description = "Get bot user information" })
        .subcommand(.{ .name = "guilds", .description = "List guilds the bot is in" })
        .subcommand(.{ .name = "send <channel_id> <msg>", .description = "Send message to a channel" })
        .subcommand(.{ .name = "channel <channel_id>", .description = "Get channel information" })
        .subcommand(.{ .name = "commands list [guild]", .description = "List application commands" })
        .subcommand(.{ .name = "commands create <n> <d>", .description = "Create a global command" })
        .subcommand(.{ .name = "webhook <url> <msg>", .description = "Execute a webhook" })
        .newline()
        .section("Environment Variables")
        .text("  DISCORD_BOT_TOKEN       Bot authentication token (required)\n")
        .text("  DISCORD_CLIENT_ID       Application client ID\n")
        .text("  DISCORD_CLIENT_SECRET   OAuth2 client secret\n")
        .text("  DISCORD_PUBLIC_KEY      Interaction verification key\n");

    builder.print();
}

fn printStatus(allocator: std.mem.Allocator) !void {
    const config = abi.connectors.tryLoadDiscord(allocator) catch |err| {
        utils.output.printError("Failed to load Discord configuration: {t}", .{err});
        return;
    };

    if (config) |cfg| {
        var mutable_cfg = cfg;
        defer mutable_cfg.deinit(allocator);

        utils.output.printHeader("Discord Configuration");
        utils.output.printKeyValueFmt(
            "Bot Token",
            "{s}...",
            .{if (cfg.bot_token.len > 8) cfg.bot_token[0..8] else cfg.bot_token},
        );
        utils.output.printKeyValueFmt("API Version", "v{d}", .{cfg.api_version});
        utils.output.printKeyValueFmt("Timeout", "{d}ms", .{cfg.timeout_ms});

        if (cfg.client_id) |id| {
            utils.output.printKeyValue("Client ID", id);
        } else {
            utils.output.printKeyValue("Client ID", "not set");
        }

        utils.output.printKeyValue("Public Key", if (cfg.public_key != null) "configured" else "not set");
        utils.output.printKeyValueFmt("Gateway Intents", "0x{x}", .{cfg.intents});
    } else {
        utils.output.printWarning("Discord not configured", .{});
        utils.output.printInfo("To configure Discord, set the following environment variables:", .{});
        utils.output.printBulletList("Environment", &[_][]const u8{
            "DISCORD_BOT_TOKEN - Your bot token from Discord Developer Portal",
        });
    }
}

fn printBotInfo(allocator: std.mem.Allocator) !void {
    var client = discord.createClient(allocator) catch |err| {
        utils.output.printError("Failed to create Discord client: {t}", .{err});
        return;
    };
    defer client.deinit();

    const user = client.getCurrentUser() catch |err| {
        utils.output.printError("Failed to get bot info: {t}", .{err});
        return;
    };

    utils.output.printHeader("Bot Information");
    utils.output.printKeyValue("ID", user.id);
    utils.output.printKeyValue("Username", user.username);
    utils.output.printKeyValue("Discriminator", user.discriminator);
    if (user.global_name) |name| {
        utils.output.printKeyValue("Global Name", name);
    }
    utils.output.printKeyValue("Bot", utils.output.boolLabel(user.bot));
    utils.output.printKeyValueFmt("Flags", "{d}", .{user.public_flags});
}

fn listGuilds(allocator: std.mem.Allocator) !void {
    var client = discord.createClient(allocator) catch |err| {
        utils.output.printError("Failed to create Discord client: {t}", .{err});
        return;
    };
    defer client.deinit();

    const guilds = client.getCurrentUserGuilds() catch |err| {
        utils.output.printError("Failed to get guilds: {t}", .{err});
        return;
    };
    defer allocator.free(guilds);

    if (guilds.len == 0) {
        utils.output.printInfo("Bot is not in any guilds.", .{});
        return;
    }

    utils.output.printHeaderFmt("Guilds ({d})", .{guilds.len});
    for (guilds) |guild| {
        std.debug.print("  " ++ utils.output.color.green ++ "•" ++ utils.output.color.reset ++ " {s: <20} (ID: {s})\n", .{ guild.name, guild.id });
    }
}

fn sendMessage(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (args.len < 2) {
        utils.output.printError("Missing channel ID or message content", .{});
        utils.output.printInfo("Usage: abi discord send <channel_id> <message>", .{});
        return;
    }

    const channel_id = std.mem.sliceTo(args[0], 0);

    // Join remaining args as message
    var message = std.ArrayListUnmanaged(u8).empty;
    defer message.deinit(allocator);

    for (args[1..], 0..) |arg, i| {
        if (i > 0) try message.append(allocator, ' ');
        try message.appendSlice(allocator, std.mem.sliceTo(arg, 0));
    }

    var client = discord.createClient(allocator) catch |err| {
        utils.output.printError("Failed to create Discord client: {t}", .{err});
        return;
    };
    defer client.deinit();

    const sent_message = client.createMessage(channel_id, message.items) catch |err| {
        utils.output.printError("Failed to send message: {t}", .{err});
        return;
    };

    utils.output.printSuccess("Message sent successfully!", .{});
    utils.output.printKeyValue("ID", sent_message.id);
    utils.output.printKeyValue("Channel", sent_message.channel_id);
}

fn channelInfo(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (args.len < 1) {
        utils.output.printError("Missing channel ID", .{});
        utils.output.printInfo("Usage: abi discord channel <channel_id>", .{});
        return;
    }

    const channel_id = std.mem.sliceTo(args[0], 0);

    var client = discord.createClient(allocator) catch |err| {
        utils.output.printError("Failed to create Discord client: {t}", .{err});
        return;
    };
    defer client.deinit();

    const channel = client.getChannel(channel_id) catch |err| {
        utils.output.printError("Failed to get channel: {t}", .{err});
        return;
    };

    utils.output.printHeader("Channel Information");
    utils.output.printKeyValue("ID", channel.id);
    if (channel.name) |name| {
        utils.output.printKeyValue("Name", name);
    }
    utils.output.printKeyValueFmt("Type", "{d}", .{channel.channel_type});
    if (channel.guild_id) |gid| {
        utils.output.printKeyValue("Guild ID", gid);
    }
}

fn manageCommands(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    var subparser = utils.args.ArgParser.init(allocator, parser.remaining());

    const commands = [_]utils.subcommand.Command{
        .{ .names = &.{"list"}, .run = dcCommandsList },
        .{ .names = &.{"create"}, .run = dcCommandsCreate },
        .{ .names = &.{"delete"}, .run = dcCommandsDelete },
    };

    try utils.subcommand.runSubcommand(
        allocator,
        &subparser,
        &commands,
        dcCommandsDefault,
        printCommandsHelp,
        dcCommandsUnknown,
    );
}

fn dcCommandsDefault(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    _ = allocator;
    _ = parser;
    utils.output.printError("Missing application command subcommand", .{});
    utils.output.printInfo("Usage: abi discord commands <list|create|delete> [options]", .{});
}

fn dcCommandsUnknown(command: []const u8) void {
    utils.output.printError("Unknown commands subcommand: {s}", .{command});
    utils.output.printInfo("Usage: abi discord commands <list|create|delete> [options]", .{});
}

fn printCommandsHelp(allocator: std.mem.Allocator) void {
    var builder = utils.help.HelpBuilder.init(allocator);
    defer builder.deinit();

    _ = builder
        .usage("abi discord commands", "<subcommand> [options]")
        .description("Manage Discord application commands.")
        .section("Subcommands")
        .subcommand(.{ .name = "list [guild_id]", .description = "List global commands or guild commands" })
        .subcommand(.{ .name = "create <name> <description>", .description = "Create a global command" })
        .subcommand(.{ .name = "delete <command_id>", .description = "Delete a global command" })
        .newline()
        .section("Options")
        .option(utils.help.common_options.help)
        .newline()
        .section("Examples")
        .example("abi discord commands list", "List global application commands")
        .example("abi discord commands list 1234567890", "List guild commands")
        .example("abi discord commands create ping \"Replies with pong\"", "Create global command")
        .example("abi discord commands delete 1234567890", "Delete global command");

    builder.print();
}

fn dcCommandsList(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    const guild_id: ?[]const u8 = if (parser.hasMore())
        parser.next().?
    else
        null;
    if (parser.hasMore()) {
        utils.output.printError("Usage: abi discord commands list [guild_id]", .{});
        return;
    }

    var client = discord.createClient(allocator) catch |err| {
        if (err == error.MissingBotToken) {
            utils.output.printWarning("Discord not configured", .{});
            utils.output.printInfo("Set DISCORD_BOT_TOKEN to list application commands.", .{});
            return;
        }
        utils.output.printError("Failed to create Discord client: {t}", .{err});
        return;
    };
    defer client.deinit();

    const app_id = client.config.client_id orelse {
        utils.output.printError("DISCORD_CLIENT_ID not configured.", .{});
        utils.output.printInfo("Set DISCORD_CLIENT_ID environment variable to your application ID.", .{});
        return;
    };

    const commands = if (guild_id) |gid|
        client.getGuildApplicationCommands(app_id, gid) catch |err| {
            utils.output.printError("Failed to get guild commands: {t}", .{err});
            return;
        }
    else
        client.getGlobalApplicationCommands(app_id) catch |err| {
            utils.output.printError("Failed to get global commands: {t}", .{err});
            return;
        };
    defer allocator.free(commands);

    if (commands.len == 0) {
        utils.output.printInfo("No application commands found.", .{});
        return;
    }

    utils.output.printHeaderFmt("Application Commands ({d})", .{commands.len});
    for (commands) |cmd| {
        std.debug.print("  " ++ utils.output.color.cyan ++ "/" ++ utils.output.color.reset ++ "{s: <15} - {s} (ID: {s})\n", .{ cmd.name, cmd.description, cmd.id });
    }
}

fn dcCommandsCreate(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    const name = parser.next() orelse {
        utils.output.printError("Missing command name", .{});
        utils.output.printInfo("Usage: abi discord commands create <name> <description>", .{});
        return;
    };
    if (!parser.hasMore()) {
        utils.output.printError("Missing command description", .{});
        utils.output.printInfo("Usage: abi discord commands create <name> <description>", .{});
        return;
    }

    var description = std.ArrayListUnmanaged(u8).empty;
    defer description.deinit(allocator);
    while (parser.hasMore()) {
        const token = parser.next().?;
        if (description.items.len > 0) try description.append(allocator, ' ');
        try description.appendSlice(allocator, token);
    }

    var client = discord.createClient(allocator) catch |err| {
        utils.output.printError("Failed to create Discord client: {t}", .{err});
        return;
    };
    defer client.deinit();

    const app_id = client.config.client_id orelse {
        utils.output.printError("DISCORD_CLIENT_ID not configured.", .{});
        return;
    };

    const cmd = client.createGlobalApplicationCommand(app_id, name, description.items, &.{}) catch |err| {
        utils.output.printError("Failed to create command: {t}", .{err});
        return;
    };

    utils.output.printSuccess("Command created successfully!", .{});
    utils.output.printKeyValueFmt("Name", "/{s}", .{cmd.name});
    utils.output.printKeyValue("ID", cmd.id);
}

fn dcCommandsDelete(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    const command_id = parser.next() orelse {
        utils.output.printError("Missing command ID", .{});
        utils.output.printInfo("Usage: abi discord commands delete <command_id>", .{});
        return;
    };
    if (parser.hasMore()) {
        utils.output.printError("Usage: abi discord commands delete <command_id>", .{});
        return;
    }

    var client = discord.createClient(allocator) catch |err| {
        utils.output.printError("Failed to create Discord client: {t}", .{err});
        return;
    };
    defer client.deinit();

    const app_id = client.config.client_id orelse {
        utils.output.printError("DISCORD_CLIENT_ID not configured.", .{});
        return;
    };

    client.deleteGlobalApplicationCommand(app_id, command_id) catch |err| {
        utils.output.printError("Failed to delete command: {t}", .{err});
        return;
    };

    utils.output.printSuccess("Command deleted successfully.", .{});
}

fn executeWebhook(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (args.len < 2) {
        utils.output.printError("Missing webhook URL or message", .{});
        utils.output.printInfo("Usage: abi discord webhook <webhook_url> <message>", .{});
        return;
    }

    const webhook_url = std.mem.sliceTo(args[0], 0);

    // Join remaining args as message
    var message = std.ArrayListUnmanaged(u8).empty;
    defer message.deinit(allocator);

    for (args[1..], 0..) |arg, i| {
        if (i > 0) try message.append(allocator, ' ');
        try message.appendSlice(allocator, std.mem.sliceTo(arg, 0));
    }

    // Parse webhook URL to extract ID and token
    const webhooks_prefix = "/api/webhooks/";
    const prefix_pos = std.mem.indexOf(u8, webhook_url, webhooks_prefix) orelse {
        utils.output.printError("Invalid webhook URL format.", .{});
        utils.output.printInfo("Format: https://discord.com/api/webhooks/<id>/<token>", .{});
        return;
    };

    const path_start = prefix_pos + webhooks_prefix.len;
    const path = webhook_url[path_start..];

    const slash_pos = std.mem.indexOf(u8, path, "/") orelse {
        utils.output.printError("Invalid webhook URL format.", .{});
        return;
    };

    const webhook_id = path[0..slash_pos];
    const webhook_token = path[slash_pos + 1 ..];

    var client = discord.createClient(allocator) catch |err| {
        utils.output.printError("Failed to create Discord client: {t}", .{err});
        return;
    };
    defer client.deinit();

    client.executeWebhook(webhook_id, webhook_token, message.items) catch |err| {
        utils.output.printError("Failed to execute webhook: {t}", .{err});
        return;
    };

    utils.output.printSuccess("Webhook executed successfully!", .{});
}
