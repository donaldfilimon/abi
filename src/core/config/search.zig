pub const SearchConfig = struct {
    max_index_size_mb: u32 = 512,
    default_result_limit: u32 = 100,
    enable_stemming: bool = true,
    enable_fuzzy: bool = true,

    pub fn defaults() SearchConfig {
        return .{};
    }
};
