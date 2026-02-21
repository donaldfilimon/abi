const catalog = @import("catalog.zig");

pub const OptimizationHints = struct {
    default_local_size: u32,
    default_queue_depth: u32,
    prefer_unified_memory: bool,
    prefer_pinned_staging: bool,
    transfer_chunk_bytes: usize,
};

pub fn forPlatform(platform: catalog.PlatformClass) OptimizationHints {
    return switch (platform) {
        .macos => .{
            .default_local_size = 256,
            .default_queue_depth = 8,
            .prefer_unified_memory = true,
            .prefer_pinned_staging = false,
            .transfer_chunk_bytes = 8 * 1024 * 1024,
        },
        .linux => .{
            .default_local_size = 256,
            .default_queue_depth = 8,
            .prefer_unified_memory = false,
            .prefer_pinned_staging = true,
            .transfer_chunk_bytes = 16 * 1024 * 1024,
        },
        .windows => .{
            .default_local_size = 128,
            .default_queue_depth = 4,
            .prefer_unified_memory = false,
            .prefer_pinned_staging = true,
            .transfer_chunk_bytes = 8 * 1024 * 1024,
        },
        .ios => .{
            .default_local_size = 128,
            .default_queue_depth = 4,
            .prefer_unified_memory = true,
            .prefer_pinned_staging = false,
            .transfer_chunk_bytes = 4 * 1024 * 1024,
        },
        .android => .{
            .default_local_size = 128,
            .default_queue_depth = 4,
            .prefer_unified_memory = false,
            .prefer_pinned_staging = false,
            .transfer_chunk_bytes = 4 * 1024 * 1024,
        },
        .web => .{
            .default_local_size = 64,
            .default_queue_depth = 2,
            .prefer_unified_memory = false,
            .prefer_pinned_staging = false,
            .transfer_chunk_bytes = 1 * 1024 * 1024,
        },
        .freestanding, .other => .{
            .default_local_size = 64,
            .default_queue_depth = 2,
            .prefer_unified_memory = true,
            .prefer_pinned_staging = false,
            .transfer_chunk_bytes = 1 * 1024 * 1024,
        },
    };
}
