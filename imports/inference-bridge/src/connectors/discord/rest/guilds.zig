//! Discord REST API — Guild Endpoints
//!
//! Guild management, roles, members, and guild channels.

const std = @import("std");
const types = @import("../types.zig");
const parsers = @import("../rest_parsers.zig");
const ClientCore = @import("core.zig").ClientCore;

const Snowflake = types.Snowflake;
const Guild = types.Guild;
const GuildMember = types.GuildMember;
const Role = types.Role;
const Channel = types.Channel;

/// Get a guild
pub fn getGuild(core: *ClientCore, guild_id: Snowflake) !Guild {
    const endpoint = try std.fmt.allocPrint(
        core.allocator,
        "/guilds/{s}",
        .{guild_id},
    );
    defer core.allocator.free(endpoint);

    var request = try core.makeRequest(.get, endpoint);
    defer request.deinit();

    var response = try core.doRequest(&request);
    defer response.deinit();

    return try parsers.parseGuild(core.allocator, response.body);
}

/// Get guild channels
pub fn getGuildChannels(core: *ClientCore, guild_id: Snowflake) ![]Channel {
    const endpoint = try std.fmt.allocPrint(
        core.allocator,
        "/guilds/{s}/channels",
        .{guild_id},
    );
    defer core.allocator.free(endpoint);

    var request = try core.makeRequest(.get, endpoint);
    defer request.deinit();

    var response = try core.doRequest(&request);
    defer response.deinit();

    return try parsers.parseChannelArray(core.allocator, response.body);
}

/// Get guild member
pub fn getGuildMember(
    core: *ClientCore,
    guild_id: Snowflake,
    user_id: Snowflake,
) !GuildMember {
    const endpoint = try std.fmt.allocPrint(
        core.allocator,
        "/guilds/{s}/members/{s}",
        .{ guild_id, user_id },
    );
    defer core.allocator.free(endpoint);

    var request = try core.makeRequest(.get, endpoint);
    defer request.deinit();

    var response = try core.doRequest(&request);
    defer response.deinit();

    return try parsers.parseGuildMember(core.allocator, response.body);
}

/// Get guild roles
pub fn getGuildRoles(core: *ClientCore, guild_id: Snowflake) ![]Role {
    const endpoint = try std.fmt.allocPrint(
        core.allocator,
        "/guilds/{s}/roles",
        .{guild_id},
    );
    defer core.allocator.free(endpoint);

    var request = try core.makeRequest(.get, endpoint);
    defer request.deinit();

    var response = try core.doRequest(&request);
    defer response.deinit();

    return try parsers.parseRoleArray(core.allocator, response.body);
}

test {
    std.testing.refAllDecls(@This());
}
