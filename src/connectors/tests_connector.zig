const std = @import("std");
const connector = @import("connector.zig");

const ConnectorError = connector.ConnectorError;

test "shared connector config validation rejects unsafe defaults" {
    try connector.validateConnectorConfig(.{ .api_key = "key", .base_url = "https://example.com" });
    try std.testing.expectError(ConnectorError.AuthenticationError, connector.validateConnectorConfig(.{ .api_key = "", .base_url = "https://example.com" }));
    try std.testing.expectError(ConnectorError.ConnectionFailed, connector.validateConnectorConfig(.{ .api_key = "key", .base_url = "" }));
    try std.testing.expectError(ConnectorError.Timeout, connector.validateConnectorConfig(.{ .api_key = "key", .base_url = "https://example.com", .timeout_ms = 0 }));
}

test {
    std.testing.refAllDecls(@This());
}
