pub const Retry = struct {
    max_attempts: u8 = 3,
    base_ms: u32 = 500,
    factor: f32 = 2.0,
};

pub const Limits = struct {
    max_tokens_total: u64 = 5_000_000,
    per_provider_rps: u16 = 5,
    per_provider_parallel: u8 = 4,
};

pub const ProviderPolicy = struct {
    name: []const u8,
    allowed: bool = true,
};

pub const Policy = struct {
    retry: Retry = .{},
    limits: Limits = .{},
    providers: []const ProviderPolicy,
};
