const std = @import("std");

pub const ConnectorError = error{
    OutOfMemory,
    ConnectionFailed,
    AuthenticationError,
    RateLimited,
    InvalidResponse,
    Timeout,
    LiveTransportUnavailable,
};

pub const TransportMode = enum {
    local,
    live,
};

pub const ConnectorConfig = struct {
    api_key: []const u8,
    base_url: []const u8,
    timeout_ms: u32 = 30000,
    transport: TransportMode = .local,
};

pub fn transportModeName(mode: TransportMode) []const u8 {
    return switch (mode) {
        .local => "local",
        .live => "live",
    };
}

pub const Response = struct {
    status: u16,
    body: []u8,
    owned: bool = true,

    pub fn deinit(self: *Response, allocator: std.mem.Allocator) void {
        if (self.owned and self.body.len > 0) {
            allocator.free(self.body);
            self.body = "";
        }
    }
};

pub fn validateConnectorConfig(config: ConnectorConfig) ConnectorError!void {
    if (config.api_key.len == 0) return ConnectorError.AuthenticationError;
    if (config.base_url.len == 0) return ConnectorError.ConnectionFailed;
    if (config.timeout_ms == 0) return ConnectorError.Timeout;
}
