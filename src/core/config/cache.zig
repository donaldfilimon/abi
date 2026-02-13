pub const EvictionPolicy = enum { lru, lfu, fifo, random };

pub const CacheConfig = struct {
    max_entries: u32 = 10_000,
    max_memory_mb: u32 = 256,
    default_ttl_ms: u64 = 300_000,
    eviction_policy: EvictionPolicy = .lru,

    pub fn defaults() CacheConfig {
        return .{};
    }
};
