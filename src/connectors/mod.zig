const std = @import("std");

// Re-export shared types at the connectors top level
pub const connector = @import("connector.zig");
pub const http = @import("http.zig");
pub const json = @import("json.zig");
pub const openai = @import("openai.zig");
pub const anthropic = @import("anthropic.zig");
pub const discord = @import("discord.zig");
pub const discord_gateway = @import("discord_gateway.zig");
pub const discord_ws_client = @import("discord_ws_client.zig");
pub const discord_routing = @import("discord_routing.zig");
pub const twilio = @import("twilio.zig");
pub const twilio_relay = @import("twilio_relay.zig");
pub const grok = @import("grok.zig");
pub const fm = @import("fm.zig");
pub const local_bridge = @import("local_bridge.zig");

// Flatten shared types for backward compatibility
pub const ConnectorError = connector.ConnectorError;
pub const TransportMode = connector.TransportMode;
pub const ConnectorConfig = connector.ConnectorConfig;
pub const FmConfig = fm.FmConfig;
pub const transportModeName = connector.transportModeName;
pub const Response = connector.Response;

test {
    std.testing.refAllDecls(@This());
    _ = @import("tests.zig");
}
