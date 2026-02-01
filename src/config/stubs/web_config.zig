const std = @import("std");

pub const WebConfig = struct {
    bind_address: []const u8 = "127.0.0.1",
    port: u16 = 3000,
    cors_enabled: bool = false,
    timeout_ms: u64 = 30000,
    max_body_size: usize = 0,

    pub fn defaults() WebConfig {
        return .{};
    }
};
