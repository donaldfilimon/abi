//! Discord REST API Client
//!
//! HTTP client for Discord REST API endpoints including:
//! - User, Guild, Channel endpoints
//! - Message operations
//! - Application commands
//! - Interactions
//! - Webhooks
//! - Gateway and Voice

const std = @import("std");
const types = @import("types.zig");
const async_http = @import("../../shared/utils/http/async_http.zig");
const json_utils = @import("../../shared/utils/json/mod.zig");

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

// ============================================================================
// Configuration
// ============================================================================

pub const Config = struct {
    bot_token: []u8,
    client_id: ?[]u8 = null,
    client_secret: ?[]u8 = null,
    public_key: ?[]u8 = null,
    api_version: u8 = 10,
    timeout_ms: u32 = 30_000,
    intents: u32 = GatewayIntent.ALL_UNPRIVILEGED,

    pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
        allocator.free(self.bot_token);
        if (self.client_id) |id| allocator.free(id);
        if (self.client_secret) |secret| allocator.free(secret);
        if (self.public_key) |key| allocator.free(key);
        self.* = undefined;
    }

    pub fn getBaseUrl(self: *const Config) []const u8 {
        _ = self;
        return "https://discord.com/api/v10";
    }
};

// ============================================================================
// REST API Client
// ============================================================================

pub const Client = struct {
    allocator: std.mem.Allocator,
    config: Config,
    http: async_http.AsyncHttpClient,

    pub fn init(allocator: std.mem.Allocator, config: Config) !Client {
        const http = try async_http.AsyncHttpClient.init(allocator);
        errdefer http.deinit();

        return .{
            .allocator = allocator,
            .config = config,
            .http = http,
        };
    }

    pub fn deinit(self: *Client) void {
        self.http.deinit();
        self.config.deinit(self.allocator);
        self.* = undefined;
    }

    // ========================================================================
    // HTTP Helpers
    // ========================================================================

    fn makeRequest(
        self: *Client,
        method: async_http.Method,
        endpoint: []const u8,
    ) !async_http.HttpRequest {
        const url = try std.fmt.allocPrint(
            self.allocator,
            "{s}{s}",
            .{ self.config.getBaseUrl(), endpoint },
        );
        errdefer self.allocator.free(url);

        var request = try async_http.HttpRequest.init(self.allocator, method, url);
        errdefer request.deinit();

        const auth = try std.fmt.allocPrint(
            self.allocator,
            "Bot {s}",
            .{self.config.bot_token},
        );
        defer self.allocator.free(auth);
        try request.setHeader("Authorization", auth);
        try request.setHeader("User-Agent", "DiscordBot (https://github.com/abi, 1.0)");

        return request;
    }

    fn doRequest(
        self: *Client,
        request: *async_http.HttpRequest,
    ) !async_http.HttpResponse {
        const response = try self.http.fetch(request);

        if (response.status_code == 401) {
            return DiscordError.Unauthorized;
        } else if (response.status_code == 403) {
            return DiscordError.Forbidden;
        } else if (response.status_code == 404) {
            return DiscordError.NotFound;
        } else if (response.status_code == 429) {
            return DiscordError.RateLimitExceeded;
        } else if (!response.isSuccess()) {
            return DiscordError.ApiRequestFailed;
        }

        return response;
    }

    // ========================================================================
    // User Endpoints
    // ========================================================================

    /// Get the current user
    pub fn getCurrentUser(self: *Client) !User {
        var request = try self.makeRequest(.get, "/users/@me");
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseUser(response.body);
    }

    /// Get a user by ID
    pub fn getUser(self: *Client, user_id: Snowflake) !User {
        const endpoint = try std.fmt.allocPrint(
            self.allocator,
            "/users/{s}",
            .{user_id},
        );
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.get, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseUser(response.body);
    }

    /// Modify the current user
    pub fn modifyCurrentUser(
        self: *Client,
        username: ?[]const u8,
        avatar: ?[]const u8,
    ) !User {
        var request = try self.makeRequest(.patch, "/users/@me");
        defer request.deinit();

        var body = std.ArrayListUnmanaged(u8){};
        defer body.deinit(self.allocator);

        try body.appendSlice(self.allocator, "{");
        var first = true;

        if (username) |name| {
            try body.print(self.allocator, "\"username\":\"{s}\"", .{name});
            first = false;
        }

        if (avatar) |av| {
            if (!first) try body.appendSlice(self.allocator, ",");
            try body.print(self.allocator, "\"avatar\":\"{s}\"", .{av});
        }

        try body.appendSlice(self.allocator, "}");
        try request.setJsonBody(body.items);

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseUser(response.body);
    }

    /// Get current user's guilds
    pub fn getCurrentUserGuilds(self: *Client) ![]Guild {
        var request = try self.makeRequest(.get, "/users/@me/guilds");
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseGuildArray(response.body);
    }

    /// Leave a guild
    pub fn leaveGuild(self: *Client, guild_id: Snowflake) !void {
        const endpoint = try std.fmt.allocPrint(
            self.allocator,
            "/users/@me/guilds/{s}",
            .{guild_id},
        );
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.delete, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();
    }

    /// Create a DM channel
    pub fn createDM(self: *Client, recipient_id: Snowflake) !Channel {
        var request = try self.makeRequest(.post, "/users/@me/channels");
        defer request.deinit();

        const body = try std.fmt.allocPrint(
            self.allocator,
            "{{\"recipient_id\":\"{s}\"}}",
            .{recipient_id},
        );
        defer self.allocator.free(body);
        try request.setJsonBody(body);

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseChannel(response.body);
    }

    // ========================================================================
    // Guild Endpoints
    // ========================================================================

    /// Get a guild
    pub fn getGuild(self: *Client, guild_id: Snowflake) !Guild {
        const endpoint = try std.fmt.allocPrint(
            self.allocator,
            "/guilds/{s}",
            .{guild_id},
        );
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.get, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseGuild(response.body);
    }

    /// Get guild channels
    pub fn getGuildChannels(self: *Client, guild_id: Snowflake) ![]Channel {
        const endpoint = try std.fmt.allocPrint(
            self.allocator,
            "/guilds/{s}/channels",
            .{guild_id},
        );
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.get, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseChannelArray(response.body);
    }

    /// Get guild member
    pub fn getGuildMember(
        self: *Client,
        guild_id: Snowflake,
        user_id: Snowflake,
    ) !GuildMember {
        const endpoint = try std.fmt.allocPrint(
            self.allocator,
            "/guilds/{s}/members/{s}",
            .{ guild_id, user_id },
        );
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.get, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseGuildMember(response.body);
    }

    /// Get guild roles
    pub fn getGuildRoles(self: *Client, guild_id: Snowflake) ![]Role {
        const endpoint = try std.fmt.allocPrint(
            self.allocator,
            "/guilds/{s}/roles",
            .{guild_id},
        );
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.get, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseRoleArray(response.body);
    }

    // ========================================================================
    // Channel Endpoints
    // ========================================================================

    /// Get a channel
    pub fn getChannel(self: *Client, channel_id: Snowflake) !Channel {
        const endpoint = try std.fmt.allocPrint(
            self.allocator,
            "/channels/{s}",
            .{channel_id},
        );
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.get, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseChannel(response.body);
    }

    /// Delete a channel
    pub fn deleteChannel(self: *Client, channel_id: Snowflake) !void {
        const endpoint = try std.fmt.allocPrint(
            self.allocator,
            "/channels/{s}",
            .{channel_id},
        );
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.delete, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();
    }

    // ========================================================================
    // Message Endpoints
    // ========================================================================

    /// Get channel messages
    pub fn getChannelMessages(
        self: *Client,
        channel_id: Snowflake,
        limit: ?u8,
    ) ![]Message {
        const lim = limit orelse 50;
        const endpoint = try std.fmt.allocPrint(
            self.allocator,
            "/channels/{s}/messages?limit={d}",
            .{ channel_id, lim },
        );
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.get, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseMessageArray(response.body);
    }

    /// Get a specific message
    pub fn getMessage(
        self: *Client,
        channel_id: Snowflake,
        message_id: Snowflake,
    ) !Message {
        const endpoint = try std.fmt.allocPrint(
            self.allocator,
            "/channels/{s}/messages/{s}",
            .{ channel_id, message_id },
        );
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.get, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseMessage(response.body);
    }

    /// Create a message
    pub fn createMessage(
        self: *Client,
        channel_id: Snowflake,
        content: []const u8,
    ) !Message {
        const endpoint = try std.fmt.allocPrint(
            self.allocator,
            "/channels/{s}/messages",
            .{channel_id},
        );
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.post, endpoint);
        defer request.deinit();

        const body = try std.fmt.allocPrint(
            self.allocator,
            "{{\"content\":\"{}\"}}",
            .{json_utils.jsonEscape(content)},
        );
        defer self.allocator.free(body);
        try request.setJsonBody(body);

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseMessage(response.body);
    }

    /// Create a message with embed
    pub fn createMessageWithEmbed(
        self: *Client,
        channel_id: Snowflake,
        content: ?[]const u8,
        embed: Embed,
    ) !Message {
        const endpoint = try std.fmt.allocPrint(
            self.allocator,
            "/channels/{s}/messages",
            .{channel_id},
        );
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.post, endpoint);
        defer request.deinit();

        const body = try self.encodeMessageWithEmbed(content, embed);
        defer self.allocator.free(body);
        try request.setJsonBody(body);

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseMessage(response.body);
    }

    /// Edit a message
    pub fn editMessage(
        self: *Client,
        channel_id: Snowflake,
        message_id: Snowflake,
        content: []const u8,
    ) !Message {
        const endpoint = try std.fmt.allocPrint(
            self.allocator,
            "/channels/{s}/messages/{s}",
            .{ channel_id, message_id },
        );
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.patch, endpoint);
        defer request.deinit();

        const body = try std.fmt.allocPrint(
            self.allocator,
            "{{\"content\":\"{}\"}}",
            .{json_utils.jsonEscape(content)},
        );
        defer self.allocator.free(body);
        try request.setJsonBody(body);

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseMessage(response.body);
    }

    /// Delete a message
    pub fn deleteMessage(
        self: *Client,
        channel_id: Snowflake,
        message_id: Snowflake,
    ) !void {
        const endpoint = try std.fmt.allocPrint(
            self.allocator,
            "/channels/{s}/messages/{s}",
            .{ channel_id, message_id },
        );
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.delete, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();
    }

    /// Add a reaction to a message
    pub fn createReaction(
        self: *Client,
        channel_id: Snowflake,
        message_id: Snowflake,
        emoji: []const u8,
    ) !void {
        const endpoint = try std.fmt.allocPrint(
            self.allocator,
            "/channels/{s}/messages/{s}/reactions/{s}/@me",
            .{ channel_id, message_id, emoji },
        );
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.put, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();
    }

    /// Delete own reaction
    pub fn deleteOwnReaction(
        self: *Client,
        channel_id: Snowflake,
        message_id: Snowflake,
        emoji: []const u8,
    ) !void {
        const endpoint = try std.fmt.allocPrint(
            self.allocator,
            "/channels/{s}/messages/{s}/reactions/{s}/@me",
            .{ channel_id, message_id, emoji },
        );
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.delete, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();
    }

    // ========================================================================
    // Application Command Endpoints
    // ========================================================================

    /// Get global application commands
    pub fn getGlobalApplicationCommands(
        self: *Client,
        application_id: Snowflake,
    ) ![]ApplicationCommand {
        const endpoint = try std.fmt.allocPrint(
            self.allocator,
            "/applications/{s}/commands",
            .{application_id},
        );
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.get, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseApplicationCommandArray(response.body);
    }

    /// Create a global application command
    pub fn createGlobalApplicationCommand(
        self: *Client,
        application_id: Snowflake,
        name: []const u8,
        description: []const u8,
        options: []const ApplicationCommandOption,
    ) !ApplicationCommand {
        const endpoint = try std.fmt.allocPrint(
            self.allocator,
            "/applications/{s}/commands",
            .{application_id},
        );
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.post, endpoint);
        defer request.deinit();

        const body = try self.encodeApplicationCommand(name, description, options);
        defer self.allocator.free(body);
        try request.setJsonBody(body);

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseApplicationCommand(response.body);
    }

    /// Delete a global application command
    pub fn deleteGlobalApplicationCommand(
        self: *Client,
        application_id: Snowflake,
        command_id: Snowflake,
    ) !void {
        const endpoint = try std.fmt.allocPrint(
            self.allocator,
            "/applications/{s}/commands/{s}",
            .{ application_id, command_id },
        );
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.delete, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();
    }

    /// Get guild application commands
    pub fn getGuildApplicationCommands(
        self: *Client,
        application_id: Snowflake,
        guild_id: Snowflake,
    ) ![]ApplicationCommand {
        const endpoint = try std.fmt.allocPrint(
            self.allocator,
            "/applications/{s}/guilds/{s}/commands",
            .{ application_id, guild_id },
        );
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.get, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseApplicationCommandArray(response.body);
    }

    /// Create a guild application command
    pub fn createGuildApplicationCommand(
        self: *Client,
        application_id: Snowflake,
        guild_id: Snowflake,
        name: []const u8,
        description: []const u8,
        options: []const ApplicationCommandOption,
    ) !ApplicationCommand {
        const endpoint = try std.fmt.allocPrint(
            self.allocator,
            "/applications/{s}/guilds/{s}/commands",
            .{ application_id, guild_id },
        );
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.post, endpoint);
        defer request.deinit();

        const body = try self.encodeApplicationCommand(name, description, options);
        defer self.allocator.free(body);
        try request.setJsonBody(body);

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseApplicationCommand(response.body);
    }

    // ========================================================================
    // Interaction Response Endpoints
    // ========================================================================

    /// Create an interaction response
    pub fn createInteractionResponse(
        self: *Client,
        interaction_id: Snowflake,
        interaction_token: []const u8,
        response_type: InteractionCallbackType,
        content: ?[]const u8,
    ) !void {
        const endpoint = try std.fmt.allocPrint(
            self.allocator,
            "/interactions/{s}/{s}/callback",
            .{ interaction_id, interaction_token },
        );
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.post, endpoint);
        defer request.deinit();

        var body = std.ArrayListUnmanaged(u8){};
        defer body.deinit(self.allocator);

        try body.print(self.allocator, "{{\"type\":{d}", .{@intFromEnum(response_type)});

        if (content) |c| {
            try body.print(
                self.allocator,
                ",\"data\":{{\"content\":\"{}\"}}",
                .{json_utils.jsonEscape(c)},
            );
        }

        try body.appendSlice(self.allocator, "}");
        try request.setJsonBody(body.items);

        var response = try self.doRequest(&request);
        defer response.deinit();
    }

    /// Edit the original interaction response
    pub fn editOriginalInteractionResponse(
        self: *Client,
        application_id: Snowflake,
        interaction_token: []const u8,
        content: []const u8,
    ) !Message {
        const endpoint = try std.fmt.allocPrint(
            self.allocator,
            "/webhooks/{s}/{s}/messages/@original",
            .{ application_id, interaction_token },
        );
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.patch, endpoint);
        defer request.deinit();

        const body = try std.fmt.allocPrint(
            self.allocator,
            "{{\"content\":\"{}\"}}",
            .{json_utils.jsonEscape(content)},
        );
        defer self.allocator.free(body);
        try request.setJsonBody(body);

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseMessage(response.body);
    }

    /// Delete the original interaction response
    pub fn deleteOriginalInteractionResponse(
        self: *Client,
        application_id: Snowflake,
        interaction_token: []const u8,
    ) !void {
        const endpoint = try std.fmt.allocPrint(
            self.allocator,
            "/webhooks/{s}/{s}/messages/@original",
            .{ application_id, interaction_token },
        );
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.delete, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();
    }

    /// Create a followup message
    pub fn createFollowupMessage(
        self: *Client,
        application_id: Snowflake,
        interaction_token: []const u8,
        content: []const u8,
    ) !Message {
        const endpoint = try std.fmt.allocPrint(
            self.allocator,
            "/webhooks/{s}/{s}",
            .{ application_id, interaction_token },
        );
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.post, endpoint);
        defer request.deinit();

        const body = try std.fmt.allocPrint(
            self.allocator,
            "{{\"content\":\"{}\"}}",
            .{json_utils.jsonEscape(content)},
        );
        defer self.allocator.free(body);
        try request.setJsonBody(body);

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseMessage(response.body);
    }

    // ========================================================================
    // Webhook Endpoints
    // ========================================================================

    /// Get a webhook by ID
    pub fn getWebhook(self: *Client, webhook_id: Snowflake) !Webhook {
        const endpoint = try std.fmt.allocPrint(
            self.allocator,
            "/webhooks/{s}",
            .{webhook_id},
        );
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.get, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseWebhook(response.body);
    }

    /// Execute a webhook
    pub fn executeWebhook(
        self: *Client,
        webhook_id: Snowflake,
        webhook_token: []const u8,
        content: []const u8,
    ) !void {
        const endpoint = try std.fmt.allocPrint(
            self.allocator,
            "/webhooks/{s}/{s}",
            .{ webhook_id, webhook_token },
        );
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.post, endpoint);
        defer request.deinit();

        const body = try std.fmt.allocPrint(
            self.allocator,
            "{{\"content\":\"{}\"}}",
            .{json_utils.jsonEscape(content)},
        );
        defer self.allocator.free(body);
        try request.setJsonBody(body);

        var response = try self.doRequest(&request);
        defer response.deinit();
    }

    /// Execute a webhook with embeds
    pub fn executeWebhookWithEmbed(
        self: *Client,
        webhook_id: Snowflake,
        webhook_token: []const u8,
        content: ?[]const u8,
        embed: Embed,
    ) !void {
        const endpoint = try std.fmt.allocPrint(
            self.allocator,
            "/webhooks/{s}/{s}",
            .{ webhook_id, webhook_token },
        );
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.post, endpoint);
        defer request.deinit();

        const body = try self.encodeMessageWithEmbed(content, embed);
        defer self.allocator.free(body);
        try request.setJsonBody(body);

        var response = try self.doRequest(&request);
        defer response.deinit();
    }

    /// Delete a webhook
    pub fn deleteWebhook(self: *Client, webhook_id: Snowflake) !void {
        const endpoint = try std.fmt.allocPrint(
            self.allocator,
            "/webhooks/{s}",
            .{webhook_id},
        );
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.delete, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();
    }

    // ========================================================================
    // Gateway Endpoints
    // ========================================================================

    /// Get the gateway URL
    pub fn getGateway(self: *Client) ![]const u8 {
        var request = try self.makeRequest(.get, "/gateway");
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();

        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            response.body,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const object = try json_utils.getRequiredObject(parsed.value);
        return try json_utils.parseStringField(object, "url", self.allocator);
    }

    /// Get the gateway URL with bot info
    pub fn getGatewayBot(self: *Client) !GatewayBotInfo {
        var request = try self.makeRequest(.get, "/gateway/bot");
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();

        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            response.body,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const object = try json_utils.getRequiredObject(parsed.value);

        return .{
            .url = try json_utils.parseStringField(object, "url", self.allocator),
            .shards = @intCast(try json_utils.parseIntField(object, "shards")),
            .session_start_limit = .{
                .total = 1000,
                .remaining = 1000,
                .reset_after = 0,
                .max_concurrency = 1,
            },
        };
    }

    // ========================================================================
    // Voice Endpoints
    // ========================================================================

    /// Get voice regions
    pub fn getVoiceRegions(self: *Client) ![]VoiceRegion {
        var request = try self.makeRequest(.get, "/voice/regions");
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseVoiceRegionArray(response.body);
    }

    // ========================================================================
    // OAuth2 Endpoints
    // ========================================================================

    /// Get the authorization URL for OAuth2
    pub fn getAuthorizationUrl(
        self: *Client,
        scopes: []const []const u8,
        redirect_uri: []const u8,
        state: ?[]const u8,
    ) ![]u8 {
        var scope_str = std.ArrayListUnmanaged(u8){};
        defer scope_str.deinit(self.allocator);

        for (scopes, 0..) |scope, i| {
            if (i > 0) try scope_str.appendSlice(self.allocator, "%20");
            try scope_str.appendSlice(self.allocator, scope);
        }

        const client_id = self.config.client_id orelse return DiscordError.MissingClientId;

        var url = std.ArrayListUnmanaged(u8){};
        errdefer url.deinit(self.allocator);

        try url.print(
            self.allocator,
            "https://discord.com/oauth2/authorize?" ++
                "client_id={s}&response_type=code&redirect_uri={s}&scope={s}",
            .{ client_id, redirect_uri, scope_str.items },
        );

        if (state) |s| {
            try url.print(self.allocator, "&state={s}", .{s});
        }

        return try url.toOwnedSlice(self.allocator);
    }

    /// Exchange an authorization code for an access token
    pub fn exchangeCode(
        self: *Client,
        code: []const u8,
        redirect_uri: []const u8,
    ) !OAuth2Token {
        const client_id = self.config.client_id orelse return DiscordError.MissingClientId;
        const client_secret = self.config.client_secret orelse {
            return DiscordError.MissingClientSecret;
        };

        const url = "https://discord.com/api/oauth2/token";

        var request = try async_http.HttpRequest.init(self.allocator, .post, url);
        defer request.deinit();

        try request.setHeader("Content-Type", "application/x-www-form-urlencoded");

        const body = try std.fmt.allocPrint(
            self.allocator,
            "grant_type=authorization_code&code={s}&redirect_uri={s}" ++
                "&client_id={s}&client_secret={s}",
            .{ code, redirect_uri, client_id, client_secret },
        );
        defer self.allocator.free(body);
        try request.setBody(body);

        var response = try self.http.fetch(&request);
        defer response.deinit();

        if (!response.isSuccess()) {
            return DiscordError.ApiRequestFailed;
        }

        return try self.parseOAuth2Token(response.body);
    }

    /// Refresh an access token
    pub fn refreshToken(self: *Client, refresh_token: []const u8) !OAuth2Token {
        const client_id = self.config.client_id orelse return DiscordError.MissingClientId;
        const client_secret = self.config.client_secret orelse {
            return DiscordError.MissingClientSecret;
        };

        const url = "https://discord.com/api/oauth2/token";

        var request = try async_http.HttpRequest.init(self.allocator, .post, url);
        defer request.deinit();

        try request.setHeader("Content-Type", "application/x-www-form-urlencoded");

        const body = try std.fmt.allocPrint(
            self.allocator,
            "grant_type=refresh_token&refresh_token={s}" ++
                "&client_id={s}&client_secret={s}",
            .{ refresh_token, client_id, client_secret },
        );
        defer self.allocator.free(body);
        try request.setBody(body);

        var response = try self.http.fetch(&request);
        defer response.deinit();

        if (!response.isSuccess()) {
            return DiscordError.ApiRequestFailed;
        }

        return try self.parseOAuth2Token(response.body);
    }

    // ========================================================================
    // JSON Encoding Helpers
    // ========================================================================

    fn encodeMessageWithEmbed(self: *Client, content: ?[]const u8, embed: Embed) ![]u8 {
        var json = std.ArrayListUnmanaged(u8){};
        errdefer json.deinit(self.allocator);

        try json.appendSlice(self.allocator, "{");

        if (content) |c| {
            try json.print(
                self.allocator,
                "\"content\":\"{}\",",
                .{json_utils.jsonEscape(c)},
            );
        }

        try json.appendSlice(self.allocator, "\"embeds\":[{");

        var first = true;
        if (embed.title) |title| {
            try json.print(
                self.allocator,
                "\"title\":\"{}\"",
                .{json_utils.jsonEscape(title)},
            );
            first = false;
        }

        if (embed.description) |desc| {
            if (!first) try json.appendSlice(self.allocator, ",");
            try json.print(
                self.allocator,
                "\"description\":\"{}\"",
                .{json_utils.jsonEscape(desc)},
            );
            first = false;
        }

        if (embed.color) |color| {
            if (!first) try json.appendSlice(self.allocator, ",");
            try json.print(self.allocator, "\"color\":{d}", .{color});
            first = false;
        }

        if (embed.url) |url_val| {
            if (!first) try json.appendSlice(self.allocator, ",");
            try json.print(self.allocator, "\"url\":\"{s}\"", .{url_val});
            first = false;
        }

        if (embed.timestamp) |ts| {
            if (!first) try json.appendSlice(self.allocator, ",");
            try json.print(self.allocator, "\"timestamp\":\"{s}\"", .{ts});
            first = false;
        }

        if (embed.footer) |footer| {
            if (!first) try json.appendSlice(self.allocator, ",");
            try json.print(
                self.allocator,
                "\"footer\":{{\"text\":\"{}\"",
                .{json_utils.jsonEscape(footer.text)},
            );
            if (footer.icon_url) |icon| {
                try json.print(self.allocator, ",\"icon_url\":\"{s}\"", .{icon});
            }
            try json.appendSlice(self.allocator, "}");
            first = false;
        }

        if (embed.author) |author| {
            if (!first) try json.appendSlice(self.allocator, ",");
            try json.print(
                self.allocator,
                "\"author\":{{\"name\":\"{}\"",
                .{json_utils.jsonEscape(author.name)},
            );
            if (author.url) |url_val| {
                try json.print(self.allocator, ",\"url\":\"{s}\"", .{url_val});
            }
            if (author.icon_url) |icon| {
                try json.print(self.allocator, ",\"icon_url\":\"{s}\"", .{icon});
            }
            try json.appendSlice(self.allocator, "}");
            first = false;
        }

        if (embed.fields.len > 0) {
            if (!first) try json.appendSlice(self.allocator, ",");
            try json.appendSlice(self.allocator, "\"fields\":[");
            for (embed.fields, 0..) |field, i| {
                if (i > 0) try json.appendSlice(self.allocator, ",");
                try json.print(
                    self.allocator,
                    "{{\"name\":\"{}\",\"value\":\"{}\",\"inline\":{s}}}",
                    .{
                        json_utils.jsonEscape(field.name),
                        json_utils.jsonEscape(field.value),
                        if (field.inline_field) "true" else "false",
                    },
                );
            }
            try json.appendSlice(self.allocator, "]");
        }

        try json.appendSlice(self.allocator, "}]}");

        return try json.toOwnedSlice(self.allocator);
    }

    fn encodeApplicationCommand(
        self: *Client,
        name: []const u8,
        description: []const u8,
        options: []const ApplicationCommandOption,
    ) ![]u8 {
        var json = std.ArrayListUnmanaged(u8){};
        errdefer json.deinit(self.allocator);

        try json.print(
            self.allocator,
            "{{\"name\":\"{s}\",\"description\":\"{}\"",
            .{ name, json_utils.jsonEscape(description) },
        );

        if (options.len > 0) {
            try json.appendSlice(self.allocator, ",\"options\":[");
            for (options, 0..) |opt, i| {
                if (i > 0) try json.appendSlice(self.allocator, ",");
                try json.print(
                    self.allocator,
                    "{{\"type\":{d},\"name\":\"{s}\",\"description\":\"{}\",\"required\":{s}}}",
                    .{
                        opt.option_type,
                        opt.name,
                        json_utils.jsonEscape(opt.description),
                        if (opt.required) "true" else "false",
                    },
                );
            }
            try json.appendSlice(self.allocator, "]");
        }

        try json.appendSlice(self.allocator, "}");

        return try json.toOwnedSlice(self.allocator);
    }

    // ========================================================================
    // JSON Parsing Helpers
    // ========================================================================

    fn parseUser(self: *Client, json: []const u8) !User {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const object = try json_utils.getRequiredObject(parsed.value);

        return User{
            .id = try json_utils.parseStringField(object, "id", self.allocator),
            .username = try json_utils.parseStringField(object, "username", self.allocator),
            .discriminator = try json_utils.parseStringField(
                object,
                "discriminator",
                self.allocator,
            ),
            .global_name = json_utils.parseOptionalStringField(
                object,
                "global_name",
                self.allocator,
            ) catch null,
            .avatar = json_utils.parseOptionalStringField(
                object,
                "avatar",
                self.allocator,
            ) catch null,
            .bot = json_utils.parseBoolField(object, "bot") catch false,
        };
    }

    fn parseGuild(self: *Client, json: []const u8) !Guild {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const object = try json_utils.getRequiredObject(parsed.value);

        return Guild{
            .id = try json_utils.parseStringField(object, "id", self.allocator),
            .name = try json_utils.parseStringField(object, "name", self.allocator),
            .owner_id = try json_utils.parseStringField(object, "owner_id", self.allocator),
            .icon = json_utils.parseOptionalStringField(
                object,
                "icon",
                self.allocator,
            ) catch null,
        };
    }

    fn parseGuildArray(self: *Client, json: []const u8) ![]Guild {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const array = parsed.value.array;
        var guilds = try self.allocator.alloc(Guild, array.items.len);
        errdefer self.allocator.free(guilds);

        for (array.items, 0..) |item, i| {
            const object = try json_utils.getRequiredObject(item);
            guilds[i] = Guild{
                .id = try json_utils.parseStringField(object, "id", self.allocator),
                .name = try json_utils.parseStringField(object, "name", self.allocator),
                .owner_id = (json_utils.parseOptionalStringField(
                    object,
                    "owner_id",
                    self.allocator,
                ) catch null) orelse "",
                .icon = json_utils.parseOptionalStringField(
                    object,
                    "icon",
                    self.allocator,
                ) catch null,
            };
        }

        return guilds;
    }

    fn parseChannel(self: *Client, json: []const u8) !Channel {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const object = try json_utils.getRequiredObject(parsed.value);

        return Channel{
            .id = try json_utils.parseStringField(object, "id", self.allocator),
            .channel_type = @intCast(try json_utils.parseIntField(object, "type")),
            .name = json_utils.parseOptionalStringField(
                object,
                "name",
                self.allocator,
            ) catch null,
            .guild_id = json_utils.parseOptionalStringField(
                object,
                "guild_id",
                self.allocator,
            ) catch null,
        };
    }

    fn parseChannelArray(self: *Client, json: []const u8) ![]Channel {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const array = parsed.value.array;
        var channels = try self.allocator.alloc(Channel, array.items.len);
        errdefer self.allocator.free(channels);

        for (array.items, 0..) |item, i| {
            const object = try json_utils.getRequiredObject(item);
            channels[i] = Channel{
                .id = try json_utils.parseStringField(object, "id", self.allocator),
                .channel_type = @intCast(try json_utils.parseIntField(object, "type")),
                .name = json_utils.parseOptionalStringField(
                    object,
                    "name",
                    self.allocator,
                ) catch null,
                .guild_id = json_utils.parseOptionalStringField(
                    object,
                    "guild_id",
                    self.allocator,
                ) catch null,
            };
        }

        return channels;
    }

    fn parseGuildMember(self: *Client, json: []const u8) !GuildMember {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const object = try json_utils.getRequiredObject(parsed.value);

        return GuildMember{
            .joined_at = try json_utils.parseStringField(object, "joined_at", self.allocator),
            .nick = json_utils.parseOptionalStringField(
                object,
                "nick",
                self.allocator,
            ) catch null,
            .deaf = json_utils.parseBoolField(object, "deaf") catch false,
            .mute = json_utils.parseBoolField(object, "mute") catch false,
        };
    }

    fn parseRoleArray(self: *Client, json: []const u8) ![]Role {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const array = parsed.value.array;
        var roles = try self.allocator.alloc(Role, array.items.len);
        errdefer self.allocator.free(roles);

        for (array.items, 0..) |item, i| {
            const object = try json_utils.getRequiredObject(item);
            roles[i] = Role{
                .id = try json_utils.parseStringField(object, "id", self.allocator),
                .name = try json_utils.parseStringField(object, "name", self.allocator),
                .permissions = try json_utils.parseStringField(
                    object,
                    "permissions",
                    self.allocator,
                ),
                .color = @intCast(json_utils.parseIntField(object, "color") catch 0),
                .position = @intCast(json_utils.parseIntField(object, "position") catch 0),
            };
        }

        return roles;
    }

    fn parseMessage(self: *Client, json: []const u8) !Message {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const object = try json_utils.getRequiredObject(parsed.value);

        const author_obj = try json_utils.parseObjectField(object, "author");

        return Message{
            .id = try json_utils.parseStringField(object, "id", self.allocator),
            .channel_id = try json_utils.parseStringField(object, "channel_id", self.allocator),
            .content = try json_utils.parseStringField(object, "content", self.allocator),
            .timestamp = try json_utils.parseStringField(object, "timestamp", self.allocator),
            .author = User{
                .id = try json_utils.parseStringField(author_obj, "id", self.allocator),
                .username = try json_utils.parseStringField(
                    author_obj,
                    "username",
                    self.allocator,
                ),
                .discriminator = try json_utils.parseStringField(
                    author_obj,
                    "discriminator",
                    self.allocator,
                ),
            },
        };
    }

    fn parseMessageArray(self: *Client, json: []const u8) ![]Message {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const array = parsed.value.array;
        var messages = try self.allocator.alloc(Message, array.items.len);
        errdefer self.allocator.free(messages);

        for (array.items, 0..) |item, i| {
            const object = try json_utils.getRequiredObject(item);
            const author_obj = try json_utils.parseObjectField(object, "author");

            messages[i] = Message{
                .id = try json_utils.parseStringField(object, "id", self.allocator),
                .channel_id = try json_utils.parseStringField(
                    object,
                    "channel_id",
                    self.allocator,
                ),
                .content = try json_utils.parseStringField(object, "content", self.allocator),
                .timestamp = try json_utils.parseStringField(
                    object,
                    "timestamp",
                    self.allocator,
                ),
                .author = User{
                    .id = try json_utils.parseStringField(author_obj, "id", self.allocator),
                    .username = try json_utils.parseStringField(
                        author_obj,
                        "username",
                        self.allocator,
                    ),
                    .discriminator = try json_utils.parseStringField(
                        author_obj,
                        "discriminator",
                        self.allocator,
                    ),
                },
            };
        }

        return messages;
    }

    fn parseApplicationCommand(self: *Client, json: []const u8) !ApplicationCommand {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const object = try json_utils.getRequiredObject(parsed.value);

        return ApplicationCommand{
            .id = try json_utils.parseStringField(object, "id", self.allocator),
            .application_id = try json_utils.parseStringField(
                object,
                "application_id",
                self.allocator,
            ),
            .name = try json_utils.parseStringField(object, "name", self.allocator),
            .description = try json_utils.parseStringField(
                object,
                "description",
                self.allocator,
            ),
            .version = try json_utils.parseStringField(object, "version", self.allocator),
        };
    }

    fn parseApplicationCommandArray(self: *Client, json: []const u8) ![]ApplicationCommand {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const array = parsed.value.array;
        var commands = try self.allocator.alloc(ApplicationCommand, array.items.len);
        errdefer self.allocator.free(commands);

        for (array.items, 0..) |item, i| {
            const object = try json_utils.getRequiredObject(item);
            commands[i] = ApplicationCommand{
                .id = try json_utils.parseStringField(object, "id", self.allocator),
                .application_id = try json_utils.parseStringField(
                    object,
                    "application_id",
                    self.allocator,
                ),
                .name = try json_utils.parseStringField(object, "name", self.allocator),
                .description = try json_utils.parseStringField(
                    object,
                    "description",
                    self.allocator,
                ),
                .version = try json_utils.parseStringField(object, "version", self.allocator),
            };
        }

        return commands;
    }

    fn parseWebhook(self: *Client, json: []const u8) !Webhook {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const object = try json_utils.getRequiredObject(parsed.value);

        return Webhook{
            .id = try json_utils.parseStringField(object, "id", self.allocator),
            .webhook_type = @intCast(try json_utils.parseIntField(object, "type")),
            .name = json_utils.parseOptionalStringField(
                object,
                "name",
                self.allocator,
            ) catch null,
            .token = json_utils.parseOptionalStringField(
                object,
                "token",
                self.allocator,
            ) catch null,
        };
    }

    fn parseVoiceRegionArray(self: *Client, json: []const u8) ![]VoiceRegion {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const array = parsed.value.array;
        var regions = try self.allocator.alloc(VoiceRegion, array.items.len);
        errdefer self.allocator.free(regions);

        for (array.items, 0..) |item, i| {
            const object = try json_utils.getRequiredObject(item);
            regions[i] = VoiceRegion{
                .id = try json_utils.parseStringField(object, "id", self.allocator),
                .name = try json_utils.parseStringField(object, "name", self.allocator),
                .optimal = json_utils.parseBoolField(object, "optimal") catch false,
                .deprecated = json_utils.parseBoolField(object, "deprecated") catch false,
            };
        }

        return regions;
    }

    fn parseOAuth2Token(self: *Client, json: []const u8) !OAuth2Token {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const object = try json_utils.getRequiredObject(parsed.value);

        return OAuth2Token{
            .access_token = try json_utils.parseStringField(
                object,
                "access_token",
                self.allocator,
            ),
            .token_type = try json_utils.parseStringField(
                object,
                "token_type",
                self.allocator,
            ),
            .expires_in = @intCast(try json_utils.parseIntField(object, "expires_in")),
            .refresh_token = json_utils.parseOptionalStringField(
                object,
                "refresh_token",
                self.allocator,
            ) catch null,
            .scope = try json_utils.parseStringField(object, "scope", self.allocator),
        };
    }
};
