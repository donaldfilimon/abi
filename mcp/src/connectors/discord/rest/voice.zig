//! Discord REST API — Gateway and Voice Endpoints
//!
//! Gateway URL retrieval, bot info, and voice region listing.

const std = @import("std");
const types = @import("../types.zig");
const parsers = @import("../rest_parsers.zig");
const json_utils = @import("../../../foundation/mod.zig").utils.json;
const ClientCore = @import("core.zig").ClientCore;

const VoiceRegion = types.VoiceRegion;
const GatewayBotInfo = types.GatewayBotInfo;

/// Get the gateway URL
pub fn getGateway(core: *ClientCore) ![]const u8 {
    var request = try core.makeRequest(.get, "/gateway");
    defer request.deinit();

    var response = try core.doRequest(&request);
    defer response.deinit();

    const parsed = try std.json.parseFromSlice(
        std.json.Value,
        core.allocator,
        response.body,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();

    const object = try json_utils.getRequiredObject(parsed.value);
    return try json_utils.parseStringField(object, "url", core.allocator);
}

/// Get the gateway URL with bot info
pub fn getGatewayBot(core: *ClientCore) !GatewayBotInfo {
    var request = try core.makeRequest(.get, "/gateway/bot");
    defer request.deinit();

    var response = try core.doRequest(&request);
    defer response.deinit();

    const parsed = try std.json.parseFromSlice(
        std.json.Value,
        core.allocator,
        response.body,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();

    const object = try json_utils.getRequiredObject(parsed.value);

    return .{
        .url = try json_utils.parseStringField(object, "url", core.allocator),
        .shards = @intCast(try json_utils.parseIntField(object, "shards")),
        .session_start_limit = .{
            .total = 1000,
            .remaining = 1000,
            .reset_after = 0,
            .max_concurrency = 1,
        },
    };
}

/// Get voice regions
pub fn getVoiceRegions(core: *ClientCore) ![]VoiceRegion {
    var request = try core.makeRequest(.get, "/voice/regions");
    defer request.deinit();

    var response = try core.doRequest(&request);
    defer response.deinit();

    return try parsers.parseVoiceRegionArray(core.allocator, response.body);
}

test {
    std.testing.refAllDecls(@This());
}
