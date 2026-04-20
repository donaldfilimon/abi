//! Discord REST API Client
//!
//! HTTP client for Discord REST API endpoints including:
//! - User, Guild, Channel endpoints
//! - Message operations
//! - Application commands
//! - Interactions
//! - Webhooks
//! - Gateway and Voice
//!
//! This facade re-exports from focused submodules under `rest/`.

const std = @import("std");
const types = @import("types.zig");

// Submodules
pub const core = @import("rest/core.zig");
pub const channels = @import("rest/channels.zig");
pub const guilds = @import("rest/guilds.zig");
pub const users = @import("rest/users.zig");
pub const interactions = @import("rest/interactions.zig");
pub const webhooks = @import("rest/webhooks.zig");
pub const voice = @import("rest/voice.zig");
pub const oauth = @import("rest/oauth.zig");

// Re-export types used in API
pub const DiscordError = types.DiscordError;
pub const Snowflake = types.Snowflake;
pub const User = types.User;
pub const Guild = types.Guild;
pub const GuildMember = types.GuildMember;
pub const Role = types.Role;
pub const Channel = types.Channel;
pub const Message = types.Message;
pub const Embed = types.Embed;
pub const ApplicationCommand = types.ApplicationCommand;
pub const ApplicationCommandOption = types.ApplicationCommandOption;
pub const InteractionCallbackType = types.InteractionCallbackType;
pub const Webhook = types.Webhook;
pub const VoiceRegion = types.VoiceRegion;
pub const OAuth2Token = types.OAuth2Token;
pub const GatewayBotInfo = types.GatewayBotInfo;
pub const SessionStartLimit = types.SessionStartLimit;
pub const GatewayIntent = types.GatewayIntent;

// Re-export Config from core
pub const Config = core.Config;

// ============================================================================
// REST API Client — Thin facade delegating to submodules
// ============================================================================

pub const Client = struct {
    _core: core.ClientCore,

    pub fn init(allocator: std.mem.Allocator, config: Config) !Client {
        return .{
            ._core = try core.ClientCore.init(allocator, config),
        };
    }

    pub fn deinit(self: *Client) void {
        self._core.deinit();
        self.* = undefined;
    }

    // ========================================================================
    // User Endpoints
    // ========================================================================

    pub fn getCurrentUser(self: *Client) !User {
        return users.getCurrentUser(&self._core);
    }

    pub fn getUser(self: *Client, user_id: Snowflake) !User {
        return users.getUser(&self._core, user_id);
    }

    pub fn modifyCurrentUser(self: *Client, username: ?[]const u8, avatar: ?[]const u8) !User {
        return users.modifyCurrentUser(&self._core, username, avatar);
    }

    pub fn getCurrentUserGuilds(self: *Client) ![]Guild {
        return users.getCurrentUserGuilds(&self._core);
    }

    pub fn leaveGuild(self: *Client, guild_id: Snowflake) !void {
        return users.leaveGuild(&self._core, guild_id);
    }

    pub fn createDM(self: *Client, recipient_id: Snowflake) !Channel {
        return users.createDM(&self._core, recipient_id);
    }

    // ========================================================================
    // Guild Endpoints
    // ========================================================================

    pub fn getGuild(self: *Client, guild_id: Snowflake) !Guild {
        return guilds.getGuild(&self._core, guild_id);
    }

    pub fn getGuildChannels(self: *Client, guild_id: Snowflake) ![]Channel {
        return guilds.getGuildChannels(&self._core, guild_id);
    }

    pub fn getGuildMember(self: *Client, guild_id: Snowflake, user_id: Snowflake) !GuildMember {
        return guilds.getGuildMember(&self._core, guild_id, user_id);
    }

    pub fn getGuildRoles(self: *Client, guild_id: Snowflake) ![]Role {
        return guilds.getGuildRoles(&self._core, guild_id);
    }

    // ========================================================================
    // Channel Endpoints
    // ========================================================================

    pub fn getChannel(self: *Client, channel_id: Snowflake) !Channel {
        return channels.getChannel(&self._core, channel_id);
    }

    pub fn deleteChannel(self: *Client, channel_id: Snowflake) !void {
        return channels.deleteChannel(&self._core, channel_id);
    }

    // ========================================================================
    // Message Endpoints
    // ========================================================================

    pub fn getChannelMessages(self: *Client, channel_id: Snowflake, limit: ?u8) ![]Message {
        return channels.getChannelMessages(&self._core, channel_id, limit);
    }

    pub fn getMessage(self: *Client, channel_id: Snowflake, message_id: Snowflake) !Message {
        return channels.getMessage(&self._core, channel_id, message_id);
    }

    pub fn createMessage(self: *Client, channel_id: Snowflake, content: []const u8) !Message {
        return channels.createMessage(&self._core, channel_id, content);
    }

    pub fn createMessageWithEmbed(self: *Client, channel_id: Snowflake, content: ?[]const u8, embed: Embed) !Message {
        return channels.createMessageWithEmbed(&self._core, channel_id, content, embed);
    }

    pub fn editMessage(self: *Client, channel_id: Snowflake, message_id: Snowflake, content: []const u8) !Message {
        return channels.editMessage(&self._core, channel_id, message_id, content);
    }

    pub fn deleteMessage(self: *Client, channel_id: Snowflake, message_id: Snowflake) !void {
        return channels.deleteMessage(&self._core, channel_id, message_id);
    }

    pub fn createReaction(self: *Client, channel_id: Snowflake, message_id: Snowflake, emoji: []const u8) !void {
        return channels.createReaction(&self._core, channel_id, message_id, emoji);
    }

    pub fn deleteOwnReaction(self: *Client, channel_id: Snowflake, message_id: Snowflake, emoji: []const u8) !void {
        return channels.deleteOwnReaction(&self._core, channel_id, message_id, emoji);
    }

    pub fn triggerTypingIndicator(self: *Client, channel_id: Snowflake) !void {
        return channels.triggerTypingIndicator(&self._core, channel_id);
    }

    // ========================================================================
    // Application Command Endpoints
    // ========================================================================

    pub fn getGlobalApplicationCommands(self: *Client, application_id: Snowflake) ![]ApplicationCommand {
        return interactions.getGlobalApplicationCommands(&self._core, application_id);
    }

    pub fn createGlobalApplicationCommand(
        self: *Client,
        application_id: Snowflake,
        name: []const u8,
        description: []const u8,
        options: []const ApplicationCommandOption,
    ) !ApplicationCommand {
        return interactions.createGlobalApplicationCommand(&self._core, application_id, name, description, options);
    }

    pub fn deleteGlobalApplicationCommand(self: *Client, application_id: Snowflake, command_id: Snowflake) !void {
        return interactions.deleteGlobalApplicationCommand(&self._core, application_id, command_id);
    }

    pub fn getGuildApplicationCommands(self: *Client, application_id: Snowflake, guild_id: Snowflake) ![]ApplicationCommand {
        return interactions.getGuildApplicationCommands(&self._core, application_id, guild_id);
    }

    pub fn createGuildApplicationCommand(
        self: *Client,
        application_id: Snowflake,
        guild_id: Snowflake,
        name: []const u8,
        description: []const u8,
        options: []const ApplicationCommandOption,
    ) !ApplicationCommand {
        return interactions.createGuildApplicationCommand(&self._core, application_id, guild_id, name, description, options);
    }

    // ========================================================================
    // Interaction Response Endpoints
    // ========================================================================

    pub fn createInteractionResponse(
        self: *Client,
        interaction_id: Snowflake,
        interaction_token: []const u8,
        response_type: InteractionCallbackType,
        content: ?[]const u8,
    ) !void {
        return interactions.createInteractionResponse(&self._core, interaction_id, interaction_token, response_type, content);
    }

    pub fn editOriginalInteractionResponse(
        self: *Client,
        application_id: Snowflake,
        interaction_token: []const u8,
        content: []const u8,
    ) !Message {
        return interactions.editOriginalInteractionResponse(&self._core, application_id, interaction_token, content);
    }

    pub fn deleteOriginalInteractionResponse(
        self: *Client,
        application_id: Snowflake,
        interaction_token: []const u8,
    ) !void {
        return interactions.deleteOriginalInteractionResponse(&self._core, application_id, interaction_token);
    }

    pub fn createFollowupMessage(
        self: *Client,
        application_id: Snowflake,
        interaction_token: []const u8,
        content: []const u8,
    ) !Message {
        return interactions.createFollowupMessage(&self._core, application_id, interaction_token, content);
    }

    // ========================================================================
    // Webhook Endpoints
    // ========================================================================

    pub fn getWebhook(self: *Client, webhook_id: Snowflake) !Webhook {
        return webhooks.getWebhook(&self._core, webhook_id);
    }

    pub fn executeWebhook(self: *Client, webhook_id: Snowflake, webhook_token: []const u8, content: []const u8) !void {
        return webhooks.executeWebhook(&self._core, webhook_id, webhook_token, content);
    }

    pub fn executeWebhookWithEmbed(self: *Client, webhook_id: Snowflake, webhook_token: []const u8, content: ?[]const u8, embed: Embed) !void {
        return webhooks.executeWebhookWithEmbed(&self._core, webhook_id, webhook_token, content, embed);
    }

    pub fn deleteWebhook(self: *Client, webhook_id: Snowflake) !void {
        return webhooks.deleteWebhook(&self._core, webhook_id);
    }

    // ========================================================================
    // Gateway Endpoints
    // ========================================================================

    pub fn getGateway(self: *Client) ![]const u8 {
        return voice.getGateway(&self._core);
    }

    pub fn getGatewayBot(self: *Client) !GatewayBotInfo {
        return voice.getGatewayBot(&self._core);
    }

    // ========================================================================
    // Voice Endpoints
    // ========================================================================

    pub fn getVoiceRegions(self: *Client) ![]VoiceRegion {
        return voice.getVoiceRegions(&self._core);
    }

    // ========================================================================
    // OAuth2 Endpoints
    // ========================================================================

    pub fn getAuthorizationUrl(
        self: *Client,
        scopes: []const []const u8,
        redirect_uri: []const u8,
        state: ?[]const u8,
    ) ![]u8 {
        return oauth.getAuthorizationUrl(&self._core, scopes, redirect_uri, state);
    }

    pub fn exchangeCode(self: *Client, code: []const u8, redirect_uri: []const u8) !OAuth2Token {
        return oauth.exchangeCode(&self._core, code, redirect_uri);
    }

    pub fn refreshToken(self: *Client, refresh_token: []const u8) !OAuth2Token {
        return oauth.refreshToken(&self._core, refresh_token);
    }
};

test {
    std.testing.refAllDecls(@This());
}
