//! Discord REST API JSON Parsers
//!
//! Standalone parsing functions for Discord API responses.
//! Each function takes an allocator and raw JSON string, returning
//! the parsed Discord type.

const std = @import("std");
const types = @import("types.zig");
const json_utils = @import("../../foundation/mod.zig").utils.json;

const User = types.User;
const Guild = types.Guild;
const GuildMember = types.GuildMember;
const Role = types.Role;
const Channel = types.Channel;
const Message = types.Message;
const ApplicationCommand = types.ApplicationCommand;
const Webhook = types.Webhook;
const VoiceRegion = types.VoiceRegion;
const OAuth2Token = types.OAuth2Token;
const GatewayBotInfo = types.GatewayBotInfo;

pub fn parseUser(allocator: std.mem.Allocator, json: []const u8) !User {
    const parsed = try std.json.parseFromSlice(
        std.json.Value,
        allocator,
        json,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();

    const object = try json_utils.getRequiredObject(parsed.value);

    return User{
        .id = try json_utils.parseStringField(object, "id", allocator),
        .username = try json_utils.parseStringField(object, "username", allocator),
        .discriminator = try json_utils.parseStringField(
            object,
            "discriminator",
            allocator,
        ),
        .global_name = json_utils.parseOptionalStringField(
            object,
            "global_name",
            allocator,
        ) catch null,
        .avatar = json_utils.parseOptionalStringField(
            object,
            "avatar",
            allocator,
        ) catch null,
        .bot = json_utils.parseBoolField(object, "bot") catch false,
    };
}

pub fn parseGuild(allocator: std.mem.Allocator, json: []const u8) !Guild {
    const parsed = try std.json.parseFromSlice(
        std.json.Value,
        allocator,
        json,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();

    const object = try json_utils.getRequiredObject(parsed.value);

    return Guild{
        .id = try json_utils.parseStringField(object, "id", allocator),
        .name = try json_utils.parseStringField(object, "name", allocator),
        .owner_id = try json_utils.parseStringField(object, "owner_id", allocator),
        .icon = json_utils.parseOptionalStringField(
            object,
            "icon",
            allocator,
        ) catch null,
    };
}

pub fn parseGuildArray(allocator: std.mem.Allocator, json: []const u8) ![]Guild {
    const parsed = try std.json.parseFromSlice(
        std.json.Value,
        allocator,
        json,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();

    const array = parsed.value.array;
    var guilds = try allocator.alloc(Guild, array.items.len);
    errdefer allocator.free(guilds);

    for (array.items, 0..) |item, i| {
        const object = try json_utils.getRequiredObject(item);
        guilds[i] = Guild{
            .id = try json_utils.parseStringField(object, "id", allocator),
            .name = try json_utils.parseStringField(object, "name", allocator),
            .owner_id = (json_utils.parseOptionalStringField(
                object,
                "owner_id",
                allocator,
            ) catch null) orelse "",
            .icon = json_utils.parseOptionalStringField(
                object,
                "icon",
                allocator,
            ) catch null,
        };
    }

    return guilds;
}

pub fn parseChannel(allocator: std.mem.Allocator, json: []const u8) !Channel {
    const parsed = try std.json.parseFromSlice(
        std.json.Value,
        allocator,
        json,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();

    const object = try json_utils.getRequiredObject(parsed.value);

    return Channel{
        .id = try json_utils.parseStringField(object, "id", allocator),
        .channel_type = @intCast(try json_utils.parseIntField(object, "type")),
        .name = json_utils.parseOptionalStringField(
            object,
            "name",
            allocator,
        ) catch null,
        .guild_id = json_utils.parseOptionalStringField(
            object,
            "guild_id",
            allocator,
        ) catch null,
    };
}

pub fn parseChannelArray(allocator: std.mem.Allocator, json: []const u8) ![]Channel {
    const parsed = try std.json.parseFromSlice(
        std.json.Value,
        allocator,
        json,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();

    const array = parsed.value.array;
    var channels = try allocator.alloc(Channel, array.items.len);
    errdefer allocator.free(channels);

    for (array.items, 0..) |item, i| {
        const object = try json_utils.getRequiredObject(item);
        channels[i] = Channel{
            .id = try json_utils.parseStringField(object, "id", allocator),
            .channel_type = @intCast(try json_utils.parseIntField(object, "type")),
            .name = json_utils.parseOptionalStringField(
                object,
                "name",
                allocator,
            ) catch null,
            .guild_id = json_utils.parseOptionalStringField(
                object,
                "guild_id",
                allocator,
            ) catch null,
        };
    }

    return channels;
}

pub fn parseGuildMember(allocator: std.mem.Allocator, json: []const u8) !GuildMember {
    const parsed = try std.json.parseFromSlice(
        std.json.Value,
        allocator,
        json,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();

    const object = try json_utils.getRequiredObject(parsed.value);

    return GuildMember{
        .joined_at = try json_utils.parseStringField(object, "joined_at", allocator),
        .nick = json_utils.parseOptionalStringField(
            object,
            "nick",
            allocator,
        ) catch null,
        .deaf = json_utils.parseBoolField(object, "deaf") catch false,
        .mute = json_utils.parseBoolField(object, "mute") catch false,
    };
}

pub fn parseRoleArray(allocator: std.mem.Allocator, json: []const u8) ![]Role {
    const parsed = try std.json.parseFromSlice(
        std.json.Value,
        allocator,
        json,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();

    const array = parsed.value.array;
    var roles = try allocator.alloc(Role, array.items.len);
    errdefer allocator.free(roles);

    for (array.items, 0..) |item, i| {
        const object = try json_utils.getRequiredObject(item);
        roles[i] = Role{
            .id = try json_utils.parseStringField(object, "id", allocator),
            .name = try json_utils.parseStringField(object, "name", allocator),
            .permissions = try json_utils.parseStringField(
                object,
                "permissions",
                allocator,
            ),
            .color = @intCast(json_utils.parseIntField(object, "color") catch 0),
            .position = @intCast(json_utils.parseIntField(object, "position") catch 0),
        };
    }

    return roles;
}

pub fn parseMessage(allocator: std.mem.Allocator, json: []const u8) !Message {
    const parsed = try std.json.parseFromSlice(
        std.json.Value,
        allocator,
        json,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();

    const object = try json_utils.getRequiredObject(parsed.value);

    const author_obj = try json_utils.parseObjectField(object, "author");

    return Message{
        .id = try json_utils.parseStringField(object, "id", allocator),
        .channel_id = try json_utils.parseStringField(object, "channel_id", allocator),
        .content = try json_utils.parseStringField(object, "content", allocator),
        .timestamp = try json_utils.parseStringField(object, "timestamp", allocator),
        .author = User{
            .id = try json_utils.parseStringField(author_obj, "id", allocator),
            .username = try json_utils.parseStringField(
                author_obj,
                "username",
                allocator,
            ),
            .discriminator = try json_utils.parseStringField(
                author_obj,
                "discriminator",
                allocator,
            ),
        },
    };
}

pub fn parseMessageArray(allocator: std.mem.Allocator, json: []const u8) ![]Message {
    const parsed = try std.json.parseFromSlice(
        std.json.Value,
        allocator,
        json,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();

    const array = parsed.value.array;
    var messages = try allocator.alloc(Message, array.items.len);
    errdefer allocator.free(messages);

    for (array.items, 0..) |item, i| {
        const object = try json_utils.getRequiredObject(item);
        const author_obj = try json_utils.parseObjectField(object, "author");

        messages[i] = Message{
            .id = try json_utils.parseStringField(object, "id", allocator),
            .channel_id = try json_utils.parseStringField(
                object,
                "channel_id",
                allocator,
            ),
            .content = try json_utils.parseStringField(object, "content", allocator),
            .timestamp = try json_utils.parseStringField(
                object,
                "timestamp",
                allocator,
            ),
            .author = User{
                .id = try json_utils.parseStringField(author_obj, "id", allocator),
                .username = try json_utils.parseStringField(
                    author_obj,
                    "username",
                    allocator,
                ),
                .discriminator = try json_utils.parseStringField(
                    author_obj,
                    "discriminator",
                    allocator,
                ),
            },
        };
    }

    return messages;
}

pub fn parseApplicationCommand(allocator: std.mem.Allocator, json: []const u8) !ApplicationCommand {
    const parsed = try std.json.parseFromSlice(
        std.json.Value,
        allocator,
        json,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();

    const object = try json_utils.getRequiredObject(parsed.value);

    return ApplicationCommand{
        .id = try json_utils.parseStringField(object, "id", allocator),
        .application_id = try json_utils.parseStringField(
            object,
            "application_id",
            allocator,
        ),
        .name = try json_utils.parseStringField(object, "name", allocator),
        .description = try json_utils.parseStringField(
            object,
            "description",
            allocator,
        ),
        .version = try json_utils.parseStringField(object, "version", allocator),
    };
}

pub fn parseApplicationCommandArray(allocator: std.mem.Allocator, json: []const u8) ![]ApplicationCommand {
    const parsed = try std.json.parseFromSlice(
        std.json.Value,
        allocator,
        json,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();

    const array = parsed.value.array;
    var commands = try allocator.alloc(ApplicationCommand, array.items.len);
    errdefer allocator.free(commands);

    for (array.items, 0..) |item, i| {
        const object = try json_utils.getRequiredObject(item);
        commands[i] = ApplicationCommand{
            .id = try json_utils.parseStringField(object, "id", allocator),
            .application_id = try json_utils.parseStringField(
                object,
                "application_id",
                allocator,
            ),
            .name = try json_utils.parseStringField(object, "name", allocator),
            .description = try json_utils.parseStringField(
                object,
                "description",
                allocator,
            ),
            .version = try json_utils.parseStringField(object, "version", allocator),
        };
    }

    return commands;
}

pub fn parseInteraction(allocator: std.mem.Allocator, json: []const u8) !types.Interaction {
    const parsed = try std.json.parseFromSlice(
        std.json.Value,
        allocator,
        json,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();

    const object = try json_utils.getRequiredObject(parsed.value);

    var interaction = types.Interaction{
        .id = try json_utils.parseStringField(object, "id", allocator),
        .application_id = try json_utils.parseStringField(object, "application_id", allocator),
        .interaction_type = @intCast(try json_utils.parseIntField(object, "type")),
        .token = try json_utils.parseStringField(object, "token", allocator),
        .version = @intCast(json_utils.parseIntField(object, "version") catch 1),
    };

    interaction.guild_id = json_utils.parseOptionalStringField(object, "guild_id", allocator) catch null;
    interaction.channel_id = json_utils.parseOptionalStringField(object, "channel_id", allocator) catch null;
    interaction.locale = json_utils.parseOptionalStringField(object, "locale", allocator) catch null;
    interaction.guild_locale = json_utils.parseOptionalStringField(object, "guild_locale", allocator) catch null;

    if (object.get("data")) |data_val| {
        interaction.data = try parseInteractionData(allocator, data_val);
    }

    if (object.get("user")) |user_val| {
        const user_obj = try json_utils.getRequiredObject(user_val);
        interaction.user = types.User{
            .id = try json_utils.parseStringField(user_obj, "id", allocator),
            .username = try json_utils.parseStringField(user_obj, "username", allocator),
            .discriminator = try json_utils.parseStringField(user_obj, "discriminator", allocator),
        };
    } else if (object.get("member")) |member_val| {
        const member_obj = try json_utils.getRequiredObject(member_val);
        if (member_obj.get("user")) |user_val| {
            const user_obj = try json_utils.getRequiredObject(user_val);
            interaction.user = types.User{
                .id = try json_utils.parseStringField(user_obj, "id", allocator),
                .username = try json_utils.parseStringField(user_obj, "username", allocator),
                .discriminator = try json_utils.parseStringField(user_obj, "discriminator", allocator),
            };
        }
    }

    return interaction;
}

fn parseInteractionData(allocator: std.mem.Allocator, val: std.json.Value) !types.InteractionData {
    const object = try json_utils.getRequiredObject(val);

    var data = types.InteractionData{
        .id = try json_utils.parseStringField(object, "id", allocator),
        .name = try json_utils.parseStringField(object, "name", allocator),
        .data_type = @intCast(json_utils.parseIntField(object, "type") catch 1),
    };

    data.custom_id = json_utils.parseOptionalStringField(object, "custom_id", allocator) catch null;
    data.guild_id = json_utils.parseOptionalStringField(object, "guild_id", allocator) catch null;
    data.target_id = json_utils.parseOptionalStringField(object, "target_id", allocator) catch null;

    if (object.get("options")) |options_val| {
        const array = options_val.array;
        var options = try allocator.alloc(types.ApplicationCommandInteractionDataOption, array.items.len);
        errdefer allocator.free(options);

        for (array.items, 0..) |opt_val, i| {
            options[i] = try parseInteractionOption(allocator, opt_val);
        }
        data.options = options;
    }

    return data;
}

fn parseInteractionOption(allocator: std.mem.Allocator, val: std.json.Value) !types.ApplicationCommandInteractionDataOption {
    const object = try json_utils.getRequiredObject(val);

    var opt = types.ApplicationCommandInteractionDataOption{
        .name = try json_utils.parseStringField(object, "name", allocator),
        .option_type = @intCast(try json_utils.parseIntField(object, "type")),
    };

    if (object.get("value")) |v| {
        opt.value = try std.fmt.allocPrint(allocator, "{}", .{v});
    }

    if (object.get("options")) |opts_val| {
        const array = opts_val.array;
        var sub_options = try allocator.alloc(types.ApplicationCommandInteractionDataOption, array.items.len);
        errdefer allocator.free(sub_options);

        for (array.items, 0..) |sub_opt_val, i| {
            sub_options[i] = try parseInteractionOption(allocator, sub_opt_val);
        }
        opt.options = sub_options;
    }

    return opt;
}

pub fn parseWebhook(allocator: std.mem.Allocator, json: []const u8) !Webhook {
    const parsed = try std.json.parseFromSlice(
        std.json.Value,
        allocator,
        json,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();

    const object = try json_utils.getRequiredObject(parsed.value);

    return Webhook{
        .id = try json_utils.parseStringField(object, "id", allocator),
        .webhook_type = @intCast(try json_utils.parseIntField(object, "type")),
        .name = json_utils.parseOptionalStringField(
            object,
            "name",
            allocator,
        ) catch null,
        .token = json_utils.parseOptionalStringField(
            object,
            "token",
            allocator,
        ) catch null,
    };
}

pub fn parseVoiceRegionArray(allocator: std.mem.Allocator, json: []const u8) ![]VoiceRegion {
    const parsed = try std.json.parseFromSlice(
        std.json.Value,
        allocator,
        json,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();

    const array = parsed.value.array;
    var regions = try allocator.alloc(VoiceRegion, array.items.len);
    errdefer allocator.free(regions);

    for (array.items, 0..) |item, i| {
        const object = try json_utils.getRequiredObject(item);
        regions[i] = VoiceRegion{
            .id = try json_utils.parseStringField(object, "id", allocator),
            .name = try json_utils.parseStringField(object, "name", allocator),
            .optimal = json_utils.parseBoolField(object, "optimal") catch false,
            .deprecated = json_utils.parseBoolField(object, "deprecated") catch false,
        };
    }

    return regions;
}

pub fn parseOAuth2Token(allocator: std.mem.Allocator, json: []const u8) !OAuth2Token {
    const parsed = try std.json.parseFromSlice(
        std.json.Value,
        allocator,
        json,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();

    const object = try json_utils.getRequiredObject(parsed.value);

    return OAuth2Token{
        .access_token = try json_utils.parseStringField(
            object,
            "access_token",
            allocator,
        ),
        .token_type = try json_utils.parseStringField(
            object,
            "token_type",
            allocator,
        ),
        .expires_in = @intCast(try json_utils.parseIntField(object, "expires_in")),
        .refresh_token = json_utils.parseOptionalStringField(
            object,
            "refresh_token",
            allocator,
        ) catch null,
        .scope = try json_utils.parseStringField(object, "scope", allocator),
    };
}

pub fn parseGatewayBotInfo(allocator: std.mem.Allocator, json: []const u8) !GatewayBotInfo {
    const parsed = try std.json.parseFromSlice(
        std.json.Value,
        allocator,
        json,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();

    const object = try json_utils.getRequiredObject(parsed.value);
    const limit_object = try json_utils.parseObjectField(object, "session_start_limit");

    return GatewayBotInfo{
        .url = try json_utils.parseStringField(object, "url", allocator),
        .shards = @intCast(try json_utils.parseIntField(object, "shards")),
        .session_start_limit = .{
            .total = @intCast(try json_utils.parseIntField(limit_object, "total")),
            .remaining = @intCast(try json_utils.parseIntField(limit_object, "remaining")),
            .reset_after = @intCast(try json_utils.parseIntField(limit_object, "reset_after")),
            .max_concurrency = @intCast(try json_utils.parseIntField(limit_object, "max_concurrency")),
        },
    };
}

test "parseWebhook parses offline payload" {
    const allocator = std.testing.allocator;
    const webhook = try parseWebhook(allocator,
        \\{"id":"1","type":1,"name":"deploy","token":"tok"}
    );
    defer allocator.free(webhook.id);
    defer if (webhook.name) |name| allocator.free(name);
    defer if (webhook.token) |token| allocator.free(token);

    try std.testing.expectEqualStrings("1", webhook.id);
    try std.testing.expectEqual(@as(u8, 1), webhook.webhook_type);
    try std.testing.expectEqualStrings("deploy", webhook.name.?);
    try std.testing.expectEqualStrings("tok", webhook.token.?);
}

test "parseApplicationCommand parses command payload" {
    const allocator = std.testing.allocator;
    const command = try parseApplicationCommand(allocator,
        \\{"id":"10","application_id":"20","name":"ping","description":"Ping","version":"30"}
    );
    defer allocator.free(command.id);
    defer allocator.free(command.application_id);
    defer allocator.free(command.name);
    defer allocator.free(command.description);
    defer allocator.free(command.version);

    try std.testing.expectEqualStrings("10", command.id);
    try std.testing.expectEqualStrings("20", command.application_id);
    try std.testing.expectEqualStrings("ping", command.name);
    try std.testing.expectEqualStrings("Ping", command.description);
    try std.testing.expectEqualStrings("30", command.version);
}

test "parseVoiceRegionArray parses metadata" {
    const allocator = std.testing.allocator;
    const regions = try parseVoiceRegionArray(allocator,
        \\[{"id":"us-east","name":"US East","optimal":true,"deprecated":false}]
    );
    defer {
        for (regions) |region| {
            allocator.free(region.id);
            allocator.free(region.name);
        }
        allocator.free(regions);
    }

    try std.testing.expectEqual(@as(usize, 1), regions.len);
    try std.testing.expectEqualStrings("us-east", regions[0].id);
    try std.testing.expect(regions[0].optimal);
    try std.testing.expect(!regions[0].deprecated);
}

test "parseOAuth2Token parses token payload" {
    const allocator = std.testing.allocator;
    const token = try parseOAuth2Token(allocator,
        \\{"access_token":"access","token_type":"Bearer","expires_in":3600,"refresh_token":"refresh","scope":"identify"}
    );
    defer allocator.free(token.access_token);
    defer allocator.free(token.token_type);
    defer if (token.refresh_token) |refresh| allocator.free(refresh);
    defer allocator.free(token.scope);

    try std.testing.expectEqualStrings("access", token.access_token);
    try std.testing.expectEqualStrings("Bearer", token.token_type);
    try std.testing.expectEqual(@as(u64, 3600), token.expires_in);
    try std.testing.expectEqualStrings("refresh", token.refresh_token.?);
    try std.testing.expectEqualStrings("identify", token.scope);
}

test "parseGatewayBotInfo parses gateway metadata" {
    const allocator = std.testing.allocator;
    const info = try parseGatewayBotInfo(allocator,
        \\{"url":"wss://gateway.discord.gg","shards":2,"session_start_limit":{"total":1000,"remaining":999,"reset_after":14400000,"max_concurrency":1}}
    );
    defer allocator.free(info.url);

    try std.testing.expectEqualStrings("wss://gateway.discord.gg", info.url);
    try std.testing.expectEqual(@as(u32, 2), info.shards);
    try std.testing.expectEqual(@as(u32, 1000), info.session_start_limit.total);
    try std.testing.expectEqual(@as(u32, 999), info.session_start_limit.remaining);
    try std.testing.expectEqual(@as(u64, 14400000), info.session_start_limit.reset_after);
    try std.testing.expectEqual(@as(u32, 1), info.session_start_limit.max_concurrency);
}

test {
    std.testing.refAllDecls(@This());
}
