pub const PagesConfig = struct {
    max_pages: u32 = 256,
    default_layout: []const u8 = "default",
    enable_template_cache: bool = true,
    template_cache_size: u32 = 64,
    default_cache_ttl_ms: u64 = 0, // 0 = no caching

    pub fn defaults() PagesConfig {
        return .{};
    }
};
