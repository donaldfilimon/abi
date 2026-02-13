pub const AuthConfig = struct {
    jwt_secret: ?[]const u8 = null,
    session_timeout_ms: u64 = 3600_000,
    max_api_keys: u32 = 100,
    enable_rbac: bool = false,
    enable_rate_limit: bool = false,

    pub fn defaults() AuthConfig {
        return .{};
    }
};
