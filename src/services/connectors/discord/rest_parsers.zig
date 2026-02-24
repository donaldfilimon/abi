//! Discord REST API JSON Parsers
//!
//! Standalone parsing functions for Discord API responses.
//! Each function takes an allocator and raw JSON string, returning
//! the parsed Discord type.

const std = @import("std");
const types = @import("types.zig");
const json_utils = @import("../../shared/utils.zig").json;

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

test {
    std.testing.refAllDecls(@This());
}
