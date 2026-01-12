//! Discord API Connector
//!
//! Comprehensive Discord API integration supporting:
//! - REST API for all Discord resources
//! - Gateway WebSocket for real-time events
//! - Application commands (slash commands, context menus)
//! - Interactions (buttons, select menus, modals)
//! - Webhooks for sending messages
//! - OAuth2 for user authentication
//! - Voice connection management
//!
//! Environment Variables:
//! - ABI_DISCORD_BOT_TOKEN / DISCORD_BOT_TOKEN - Bot authentication token
//! - ABI_DISCORD_CLIENT_ID / DISCORD_CLIENT_ID - Application client ID
//! - ABI_DISCORD_CLIENT_SECRET / DISCORD_CLIENT_SECRET - OAuth2 client secret
//! - ABI_DISCORD_PUBLIC_KEY / DISCORD_PUBLIC_KEY - Interaction verification key

const std = @import("std");
const connectors = @import("../mod.zig");

// Re-export submodules
pub const types = @import("types.zig");
pub const rest = @import("rest.zig");
pub const utils = @import("utils.zig");

// Re-export common types for convenience
pub const DiscordError = types.DiscordError;
pub const Snowflake = types.Snowflake;
pub const User = types.User;
pub const UserFlags = types.UserFlags;
pub const Guild = types.Guild;
pub const GuildMember = types.GuildMember;
pub const Role = types.Role;
pub const RoleTags = types.RoleTags;
pub const ChannelType = types.ChannelType;
pub const Channel = types.Channel;
pub const PermissionOverwrite = types.PermissionOverwrite;
pub const ThreadMetadata = types.ThreadMetadata;
pub const ThreadMember = types.ThreadMember;
pub const DefaultReaction = types.DefaultReaction;
pub const Message = types.Message;
pub const ChannelMention = types.ChannelMention;
pub const Attachment = types.Attachment;
pub const Embed = types.Embed;
pub const EmbedFooter = types.EmbedFooter;
pub const EmbedMedia = types.EmbedMedia;
pub const EmbedProvider = types.EmbedProvider;
pub const EmbedAuthor = types.EmbedAuthor;
pub const EmbedField = types.EmbedField;
pub const Reaction = types.Reaction;
pub const ReactionCountDetails = types.ReactionCountDetails;
pub const Emoji = types.Emoji;
pub const MessageActivity = types.MessageActivity;
pub const MessageReference = types.MessageReference;
pub const MessageInteraction = types.MessageInteraction;
pub const StickerItem = types.StickerItem;
pub const Application = types.Application;
pub const Team = types.Team;
pub const TeamMember = types.TeamMember;
pub const InstallParams = types.InstallParams;
pub const IntegrationTypesConfig = types.IntegrationTypesConfig;
pub const IntegrationTypeConfig = types.IntegrationTypeConfig;
pub const InteractionType = types.InteractionType;
pub const Interaction = types.Interaction;
pub const InteractionData = types.InteractionData;
pub const ResolvedData = types.ResolvedData;
pub const ApplicationCommandInteractionDataOption = types.ApplicationCommandInteractionDataOption;
pub const Entitlement = types.Entitlement;
pub const AuthorizingIntegrationOwners = types.AuthorizingIntegrationOwners;
pub const ComponentType = types.ComponentType;
pub const Component = types.Component;
pub const ButtonStyle = types.ButtonStyle;
pub const SelectOption = types.SelectOption;
pub const DefaultValue = types.DefaultValue;
pub const TextInputStyle = types.TextInputStyle;
pub const ApplicationCommandType = types.ApplicationCommandType;
pub const ApplicationCommand = types.ApplicationCommand;
pub const ApplicationCommandOptionType = types.ApplicationCommandOptionType;
pub const ApplicationCommandOption = types.ApplicationCommandOption;
pub const ApplicationCommandOptionChoice = types.ApplicationCommandOptionChoice;
pub const InteractionCallbackType = types.InteractionCallbackType;
pub const InteractionResponse = types.InteractionResponse;
pub const InteractionCallbackData = types.InteractionCallbackData;
pub const AllowedMentions = types.AllowedMentions;
pub const MessageFlags = types.MessageFlags;
pub const Webhook = types.Webhook;
pub const WebhookType = types.WebhookType;
pub const VoiceState = types.VoiceState;
pub const VoiceRegion = types.VoiceRegion;
pub const GatewayOpcode = types.GatewayOpcode;
pub const GatewayIntent = types.GatewayIntent;
pub const GatewayPayload = types.GatewayPayload;
pub const IdentifyProperties = types.IdentifyProperties;
pub const PresenceUpdate = types.PresenceUpdate;
pub const Activity = types.Activity;
pub const ActivityType = types.ActivityType;
pub const ActivityTimestamps = types.ActivityTimestamps;
pub const ActivityParty = types.ActivityParty;
pub const ActivityAssets = types.ActivityAssets;
pub const ActivitySecrets = types.ActivitySecrets;
pub const ActivityButton = types.ActivityButton;
pub const OAuth2Scope = types.OAuth2Scope;
pub const OAuth2Token = types.OAuth2Token;
pub const GatewayBotInfo = types.GatewayBotInfo;
pub const SessionStartLimit = types.SessionStartLimit;

// Re-export REST client
pub const Config = rest.Config;
pub const Client = rest.Client;

// Re-export utilities
pub const TimestampStyle = utils.TimestampStyle;
pub const Permission = utils.Permission;
pub const parseTimestamp = utils.parseTimestamp;
pub const formatTimestamp = utils.formatTimestamp;
pub const calculatePermissions = utils.calculatePermissions;
pub const hasPermission = utils.hasPermission;

// ============================================================================
// Environment Loading
// ============================================================================

pub fn loadFromEnv(allocator: std.mem.Allocator) !Config {
    const bot_token = (try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_DISCORD_BOT_TOKEN",
        "DISCORD_BOT_TOKEN",
    })) orelse return DiscordError.MissingBotToken;

    const client_id = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_DISCORD_CLIENT_ID",
        "DISCORD_CLIENT_ID",
    });

    const client_secret = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_DISCORD_CLIENT_SECRET",
        "DISCORD_CLIENT_SECRET",
    });

    const public_key = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_DISCORD_PUBLIC_KEY",
        "DISCORD_PUBLIC_KEY",
    });

    return .{
        .bot_token = bot_token,
        .client_id = client_id,
        .client_secret = client_secret,
        .public_key = public_key,
    };
}

pub fn createClient(allocator: std.mem.Allocator) !Client {
    const config = try loadFromEnv(allocator);
    return try Client.init(allocator, config);
}

// ============================================================================
// Tests
// ============================================================================

test "discord config lifecycle" {
    const allocator = std.testing.allocator;

    var config = Config{
        .bot_token = try allocator.dupe(u8, "test_token"),
        .client_id = try allocator.dupe(u8, "123456789"),
        .client_secret = null,
        .public_key = null,
    };
    defer config.deinit(allocator);

    try std.testing.expectEqualStrings("test_token", config.bot_token);
    try std.testing.expectEqualStrings("123456789", config.client_id.?);
}

test "gateway intents calculation" {
    const intents = GatewayIntent.GUILDS |
        GatewayIntent.GUILD_MESSAGES |
        GatewayIntent.MESSAGE_CONTENT;

    try std.testing.expect(intents & GatewayIntent.GUILDS != 0);
    try std.testing.expect(intents & GatewayIntent.GUILD_MESSAGES != 0);
    try std.testing.expect(intents & GatewayIntent.MESSAGE_CONTENT != 0);
    try std.testing.expect(intents & GatewayIntent.GUILD_MEMBERS == 0);
}

test "permission check" {
    const perms = Permission.SEND_MESSAGES |
        Permission.VIEW_CHANNEL |
        Permission.ADMINISTRATOR;

    try std.testing.expect(hasPermission(perms, Permission.SEND_MESSAGES));
    try std.testing.expect(hasPermission(perms, Permission.VIEW_CHANNEL));
    try std.testing.expect(hasPermission(perms, Permission.ADMINISTRATOR));
    try std.testing.expect(!hasPermission(perms, Permission.MANAGE_GUILD));
}
