//! Discord API integration for the Abi AI framework
//!
//! This module provides comprehensive Discord API integration including:
//! - REST API client for Discord endpoints
//! - Webhook support for automated messaging
//! - Rate limiting and error handling
//! - Message formatting and embed support

const std = @import("std");
const core = @import("../core/mod.zig");

/// Re-export commonly used types
pub const Allocator = core.Allocator;

/// Discord API specific error types
pub const DiscordError = error{
    InvalidToken,
    RateLimited,
    Forbidden,
    NotFound,
    BadRequest,
    ServerError,
    GatewayError,
    ConnectionFailed,
    AuthenticationFailed,
    InvalidPermissions,
    MissingIntent,
    InvalidInteraction,
} || core.Error;

/// Discord API configuration
pub const DiscordConfig = struct {
    token: []const u8,
    base_url: []const u8 = "https://discord.com/api/v10",
    timeout_seconds: u32 = 30,
    max_retries: u32 = 3,
    rate_limit_retry_delay_ms: u64 = 1000,
};

/// Discord message structure
pub const DiscordMessage = struct {
    content: []const u8,
    embeds: ?[]Embed = null,
    tts: bool = false,
    flags: u32 = 0,
    allowed_mentions: ?AllowedMentions = null,

    pub fn deinit(self: DiscordMessage, allocator: std.mem.Allocator) void {
        if (self.embeds) |embeds| {
            for (embeds) |embed| {
                embed.deinit(allocator);
            }
            allocator.free(embeds);
        }
        if (self.allowed_mentions) |mentions| {
            mentions.deinit(allocator);
        }
    }
};

/// Allowed mentions configuration
pub const AllowedMentions = struct {
    parse: ?[]const []const u8 = null,
    users: ?[]const []const u8 = null,
    roles: ?[]const []const u8 = null,
    replied_user: bool = false,

    pub fn deinit(self: AllowedMentions, allocator: std.mem.Allocator) void {
        if (self.parse) |p| allocator.free(p);
        if (self.users) |u| allocator.free(u);
        if (self.roles) |r| allocator.free(r);
    }
};

/// Discord embed structure
pub const Embed = struct {
    title: ?[]const u8 = null,
    description: ?[]const u8 = null,
    color: ?u32 = null,
    fields: ?[]Field = null,
    footer: ?Footer = null,
    timestamp: ?[]const u8 = null,

    pub fn deinit(self: Embed, allocator: std.mem.Allocator) void {
        if (self.title) |title| allocator.free(title);
        if (self.description) |desc| allocator.free(desc);
        if (self.fields) |fields| {
            for (fields) |field| {
                field.deinit(allocator);
            }
            allocator.free(fields);
        }
        if (self.footer) |footer| {
            footer.deinit(allocator);
        }
        if (self.timestamp) |ts| allocator.free(ts);
    }
};

/// Embed footer
pub const Footer = struct {
    text: []const u8,
    icon_url: ?[]const u8 = null,

    pub fn deinit(self: Footer, allocator: std.mem.Allocator) void {
        allocator.free(self.text);
        if (self.icon_url) |url| allocator.free(url);
    }
};

/// Embed field structure
pub const Field = struct {
    name: []const u8,
    value: []const u8,
    @"inline": bool = false,

    pub fn deinit(self: Field, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.value);
    }
};

/// Discord API error types - improved based on Zig best practices
pub const DiscordError = error{
    InvalidToken,
    InvalidChannelId,
    InvalidMessage,
    NetworkError,
    RateLimited,
    Unauthorized,
    Forbidden,
    NotFound,
    ServerError,
    InvalidResponse,
    JsonParseError,
    TimeoutError,
    MaxRetriesExceeded,
    ContentTooLong,
} || std.mem.Allocator.Error || std.fs.File.OpenError;

/// Rate limit information
pub const RateLimitInfo = struct {
    limit: u32,
    remaining: u32,
    reset_after_ms: u64,
    bucket: []const u8,

    pub fn deinit(self: RateLimitInfo, allocator: std.mem.Allocator) void {
        allocator.free(self.bucket);
    }
};

/// Discord API client - optimized structure
pub const DiscordClient = struct {
    allocator: std.mem.Allocator,
    config: DiscordConfig,
    client: std.http.Client,
    rate_limits: std.StringHashMap(RateLimitInfo),
    stats: ClientStats = .{},

    pub fn init(allocator: std.mem.Allocator, config: DiscordConfig) !*DiscordClient {
        const self = try allocator.create(DiscordClient);
        errdefer allocator.destroy(self);

        self.* = .{
            .allocator = allocator,
            .config = config,
            .client = std.http.Client{ .allocator = allocator },
            .rate_limits = std.StringHashMap(RateLimitInfo).init(allocator),
        };

        return self;
    }

    pub fn deinit(self: *DiscordClient) void {
        var iter = self.rate_limits.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
        }
        self.rate_limits.deinit();
        self.client.deinit();
        self.allocator.destroy(self);
    }

    /// Send a message to a Discord channel with retry logic
    pub fn sendMessage(self: *DiscordClient, channel_id: []const u8, message: DiscordMessage) DiscordError!void {
        // Fast path validation
        if (channel_id.len == 0) return DiscordError.InvalidChannelId;
        if (message.content.len == 0 and (message.embeds == null or message.embeds.?.len == 0)) {
            return DiscordError.InvalidMessage;
        }
        if (message.content.len > 2000) return DiscordError.ContentTooLong;

        const endpoint = try std.fmt.allocPrint(
            self.allocator,
            "{s}/channels/{s}/messages",
            .{ self.config.base_url, channel_id },
        );
        defer self.allocator.free(endpoint);

        var retry_count: u32 = 0;
        while (retry_count < self.config.max_retries) : (retry_count += 1) {
            const result = self.sendRequest(.POST, endpoint, message) catch |err| {
                if (err == DiscordError.RateLimited and retry_count < self.config.max_retries - 1) {
                    std.time.sleep(self.config.rate_limit_retry_delay_ms * std.time.ns_per_ms);
                    continue;
                }
                return err;
            };

            // Success
            return result;
        }

        return DiscordError.MaxRetriesExceeded;
    }

    /// Send a simple text message
    pub fn sendText(self: *DiscordClient, channel_id: []const u8, content: []const u8) DiscordError!void {
        const message = DiscordMessage{ .content = content };
        try self.sendMessage(channel_id, message);
    }

    /// Send an embed message
    pub fn sendEmbed(self: *DiscordClient, channel_id: []const u8, embed: Embed) DiscordError!void {
        const embeds = [_]Embed{embed};
        const message = DiscordMessage{
            .content = "", // Discord requires either content or embeds
            .embeds = &embeds,
        };
        try self.sendMessage(channel_id, message);
    }

    /// Get channel information
    pub fn getChannel(self: *DiscordClient, channel_id: []const u8) DiscordError![]u8 {
        if (channel_id.len == 0) return DiscordError.InvalidChannelId;

        const endpoint = try std.fmt.allocPrint(
            self.allocator,
            "{s}/channels/{s}",
            .{ self.config.base_url, channel_id },
        );
        defer self.allocator.free(endpoint);

        var response = std.ArrayList(u8).init(self.allocator);
        defer response.deinit();

        try self.sendRequest(.GET, endpoint, null);

        return self.allocator.dupe(u8, response.items) catch DiscordError.OutOfMemory;
    }

    /// Send HTTP request with proper headers and error handling
    fn sendRequest(self: *DiscordClient, method: std.http.Method, endpoint: []const u8, payload: anytype) DiscordError!void {
        const start_time = std.time.microTimestamp();
        defer {
            self.stats.request_count += 1;
            self.stats.total_request_time_us += @intCast(std.time.microTimestamp() - start_time);
        }

        const auth_value = try std.fmt.allocPrint(self.allocator, "Bot {s}", .{self.config.token});
        defer self.allocator.free(auth_value);

        var headers = std.ArrayList(std.http.Header).init(self.allocator);
        defer headers.deinit();

        try headers.append(.{ .name = "Authorization", .value = auth_value });
        try headers.append(.{ .name = "Content-Type", .value = "application/json" });
        try headers.append(.{ .name = "User-Agent", .value = "AbiBot/1.0" });

        var body_buf = std.ArrayList(u8).init(self.allocator);
        defer body_buf.deinit();

        if (payload) |p| {
            try std.json.stringify(p, .{}, body_buf.writer());
        }

        var response = std.ArrayList(u8).init(self.allocator);
        defer response.deinit();

        const result = self.client.fetch(.{
            .location = .{ .url = endpoint },
            .method = method,
            .headers = .{ .authorization = .omit },
            .extra_headers = headers.items,
            .payload = if (body_buf.items.len > 0) body_buf.items else null,
            .response_storage = .{ .dynamic = &response },
        }) catch return DiscordError.NetworkError;

        try self.handleResponse(result, response.items);
    }

    fn handleResponse(self: *DiscordClient, result: std.http.Client.FetchResult, body: []const u8) DiscordError!void {
        const status = @intFromEnum(result.status);

        // Check for rate limit headers
        if (result.headers.getFirstValue("X-RateLimit-Limit")) |_| {
            // TODO: Parse and store rate limit information
        }

        switch (status) {
            200, 201, 204 => {
                // Success
                self.stats.success_count += 1;
                if (body.len > 0 and std.log.scopeEnabled(.discord, .debug)) {
                    std.log.debug("Discord API response: {s}", .{body});
                }
            },
            400 => {
                std.log.err("Discord API bad request: {s}", .{body});
                return DiscordError.InvalidResponse;
            },
            401 => {
                self.stats.error_count += 1;
                return DiscordError.Unauthorized;
            },
            403 => {
                self.stats.error_count += 1;
                return DiscordError.Forbidden;
            },
            404 => {
                self.stats.error_count += 1;
                return DiscordError.NotFound;
            },
            429 => {
                self.stats.rate_limit_count += 1;
                return DiscordError.RateLimited;
            },
            500...599 => {
                self.stats.error_count += 1;
                return DiscordError.ServerError;
            },
            else => {
                std.log.err("Discord API unexpected status: {d}, body: {s}", .{ status, body });
                self.stats.error_count += 1;
                return DiscordError.InvalidResponse;
            },
        }
    }

    /// Get client statistics
    pub fn getStats(self: *const DiscordClient) ClientStats {
        return self.stats;
    }
};

/// Client statistics
pub const ClientStats = struct {
    request_count: u64 = 0,
    success_count: u64 = 0,
    error_count: u64 = 0,
    rate_limit_count: u64 = 0,
    total_request_time_us: u64 = 0,

    pub fn getAverageRequestTime(self: *const ClientStats) u64 {
        if (self.request_count == 0) return 0;
        return self.total_request_time_us / self.request_count;
    }

    pub fn getSuccessRate(self: *const ClientStats) f64 {
        if (self.request_count == 0) return 0.0;
        return @as(f64, @floatFromInt(self.success_count)) / @as(f64, @floatFromInt(self.request_count));
    }
};

/// Create a simple embed
pub fn createEmbed(allocator: std.mem.Allocator, title: []const u8, description: []const u8, color: ?u32) !Embed {
    const title_copy = try allocator.dupe(u8, title);
    errdefer allocator.free(title_copy);

    const desc_copy = try allocator.dupe(u8, description);
    errdefer allocator.free(desc_copy);

    return Embed{
        .title = title_copy,
        .description = desc_copy,
        .color = color,
    };
}

/// Create an embed with fields
pub fn createEmbedWithFields(
    allocator: std.mem.Allocator,
    title: []const u8,
    description: []const u8,
    fields: []const Field,
    color: ?u32,
) !Embed {
    const title_copy = try allocator.dupe(u8, title);
    errdefer allocator.free(title_copy);

    const desc_copy = try allocator.dupe(u8, description);
    errdefer allocator.free(desc_copy);

    const fields_copy = try allocator.alloc(Field, fields.len);
    errdefer allocator.free(fields_copy);

    for (fields, 0..) |field, i| {
        fields_copy[i] = .{
            .name = try allocator.dupe(u8, field.name),
            .value = try allocator.dupe(u8, field.value),
            .@"inline" = field.@"inline",
        };
    }

    return Embed{
        .title = title_copy,
        .description = desc_copy,
        .fields = fields_copy,
        .color = color,
    };
}

/// Embed builder for fluent API
pub const EmbedBuilder = struct {
    allocator: std.mem.Allocator,
    title: ?[]const u8 = null,
    description: ?[]const u8 = null,
    color: ?u32 = null,
    fields: std.ArrayList(Field),
    footer: ?Footer = null,
    timestamp: ?[]const u8 = null,

    pub fn init(allocator: std.mem.Allocator) EmbedBuilder {
        return .{
            .allocator = allocator,
            .fields = std.ArrayList(Field).init(allocator),
        };
    }

    pub fn deinit(self: *EmbedBuilder) void {
        for (self.fields.items) |field| {
            field.deinit(self.allocator);
        }
        self.fields.deinit();
        if (self.title) |t| self.allocator.free(t);
        if (self.description) |d| self.allocator.free(d);
        if (self.footer) |f| f.deinit(self.allocator);
        if (self.timestamp) |ts| self.allocator.free(ts);
    }

    pub fn setTitle(self: *EmbedBuilder, title: []const u8) !*EmbedBuilder {
        if (self.title) |t| self.allocator.free(t);
        self.title = try self.allocator.dupe(u8, title);
        return self;
    }

    pub fn setDescription(self: *EmbedBuilder, description: []const u8) !*EmbedBuilder {
        if (self.description) |d| self.allocator.free(d);
        self.description = try self.allocator.dupe(u8, description);
        return self;
    }

    pub fn setColor(self: *EmbedBuilder, color: u32) *EmbedBuilder {
        self.color = color;
        return self;
    }

    pub fn addField(self: *EmbedBuilder, name: []const u8, value: []const u8, inline_field: bool) !*EmbedBuilder {
        try self.fields.append(.{
            .name = try self.allocator.dupe(u8, name),
            .value = try self.allocator.dupe(u8, value),
            .@"inline" = inline_field,
        });
        return self;
    }

    pub fn build(self: *EmbedBuilder) !Embed {
        const fields_slice = try self.fields.toOwnedSlice();

        const embed = Embed{
            .title = self.title,
            .description = self.description,
            .color = self.color,
            .fields = if (fields_slice.len > 0) fields_slice else null,
            .footer = self.footer,
            .timestamp = self.timestamp,
        };

        // Clear builder state
        self.title = null;
        self.description = null;
        self.footer = null;
        self.timestamp = null;

        return embed;
    }
};

test "Discord client initialization" {
    const allocator = std.testing.allocator;
    const config = DiscordConfig{ .token = "test_token" };
    var client = try DiscordClient.init(allocator, config);
    defer client.deinit();

    try std.testing.expectEqualStrings("test_token", client.config.token);
}

test "Embed creation" {
    const allocator = std.testing.allocator;
    const embed = try createEmbed(allocator, "Test Title", "Test Description", 0xFF0000);
    defer embed.deinit(allocator);

    try std.testing.expectEqualStrings("Test Title", embed.title.?);
    try std.testing.expectEqualStrings("Test Description", embed.description.?);
    try std.testing.expectEqual(@as(u32, 0xFF0000), embed.color.?);
}

test "Embed builder" {
    const allocator = std.testing.allocator;
    var builder = EmbedBuilder.init(allocator);
    defer builder.deinit();

    _ = try builder.setTitle("Test Title");
    _ = try builder.setDescription("Test Description");
    _ = builder.setColor(0x00FF00);
    _ = try builder.addField("Field 1", "Value 1", true);
    _ = try builder.addField("Field 2", "Value 2", false);

    var embed = try builder.build();
    defer embed.deinit(allocator);

    try std.testing.expectEqualStrings("Test Title", embed.title.?);
    try std.testing.expectEqual(@as(usize, 2), embed.fields.?.len);
}
