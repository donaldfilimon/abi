//! Discord REST API — OAuth2 Endpoints
//!
//! Authorization URL generation, code exchange, and token refresh.

const std = @import("std");
const types = @import("../types.zig");
const parsers = @import("../rest_parsers.zig");
const async_http = @import("../../../foundation/mod.zig").utils.async_http;
const ClientCore = @import("core.zig").ClientCore;

const DiscordError = types.DiscordError;
const OAuth2Token = types.OAuth2Token;

/// Get the authorization URL for OAuth2
pub fn getAuthorizationUrl(
    core: *ClientCore,
    scopes: []const []const u8,
    redirect_uri: []const u8,
    state: ?[]const u8,
) ![]u8 {
    var scope_str = std.ArrayListUnmanaged(u8).empty;
    defer scope_str.deinit(core.allocator);

    for (scopes, 0..) |scope, i| {
        if (i > 0) try scope_str.appendSlice(core.allocator, "%20");
        try scope_str.appendSlice(core.allocator, scope);
    }

    const client_id = core.config.client_id orelse return DiscordError.MissingClientId;

    var url = std.ArrayListUnmanaged(u8).empty;
    errdefer url.deinit(core.allocator);

    try url.print(
        core.allocator,
        "https://discord.com/oauth2/authorize?" ++
            "client_id={s}&response_type=code&redirect_uri={s}&scope={s}",
        .{ client_id, redirect_uri, scope_str.items },
    );

    if (state) |s| {
        try url.print(core.allocator, "&state={s}", .{s});
    }

    return try url.toOwnedSlice(core.allocator);
}

/// Exchange an authorization code for an access token
pub fn exchangeCode(
    core: *ClientCore,
    code: []const u8,
    redirect_uri: []const u8,
) !OAuth2Token {
    const client_id = core.config.client_id orelse return DiscordError.MissingClientId;
    const client_secret = core.config.client_secret orelse {
        return DiscordError.MissingClientSecret;
    };

    const url = "https://discord.com/api/oauth2/token";

    var request = try async_http.HttpRequest.init(core.allocator, .post, url);
    defer request.deinit();

    try request.setHeader("Content-Type", "application/x-www-form-urlencoded");

    const body = try std.fmt.allocPrint(
        core.allocator,
        "grant_type=authorization_code&code={s}&redirect_uri={s}" ++
            "&client_id={s}&client_secret={s}",
        .{ code, redirect_uri, client_id, client_secret },
    );
    defer core.allocator.free(body);
    try request.setBody(body);

    var response = try core.http.fetch(&request);
    defer response.deinit();

    if (!response.isSuccess()) {
        return DiscordError.ApiRequestFailed;
    }

    return try parsers.parseOAuth2Token(core.allocator, response.body);
}

/// Refresh an access token
pub fn refreshToken(core: *ClientCore, refresh_token: []const u8) !OAuth2Token {
    const client_id = core.config.client_id orelse return DiscordError.MissingClientId;
    const client_secret = core.config.client_secret orelse {
        return DiscordError.MissingClientSecret;
    };

    const url = "https://discord.com/api/oauth2/token";

    var request = try async_http.HttpRequest.init(core.allocator, .post, url);
    defer request.deinit();

    try request.setHeader("Content-Type", "application/x-www-form-urlencoded");

    const body = try std.fmt.allocPrint(
        core.allocator,
        "grant_type=refresh_token&refresh_token={s}" ++
            "&client_id={s}&client_secret={s}",
        .{ refresh_token, client_id, client_secret },
    );
    defer core.allocator.free(body);
    try request.setBody(body);

    var response = try core.http.fetch(&request);
    defer response.deinit();

    if (!response.isSuccess()) {
        return DiscordError.ApiRequestFailed;
    }

    return try parsers.parseOAuth2Token(core.allocator, response.body);
}

test {
    std.testing.refAllDecls(@This());
}
