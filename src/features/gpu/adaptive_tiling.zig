//! Adaptive Matrix Tiling
//!
//! Selects optimal tile sizes for matrix operations based on
//! matrix dimensions and device capabilities.

const std = @import("std");

/// Adaptive tiling configuration for matrix operations.
/// Selects optimal tile sizes based on matrix dimensions and device capabilities.
pub const AdaptiveTiling = struct {
    device_info: DeviceInfo,

    /// Device capability information for tiling decisions.
    pub const DeviceInfo = struct {
        /// Maximum threads per block (e.g., 1024 for most CUDA devices).
        max_threads_per_block: u32 = 1024,
        /// Maximum shared memory per block in bytes (e.g., 48KB).
        max_shared_memory: u32 = 48 * 1024,
        /// Warp size (32 for NVIDIA, 64 for AMD).
        warp_size: u32 = 32,
        /// Compute capability for architecture-specific optimizations.
        compute_capability: ComputeCapability = .{ .major = 7, .minor = 0 },
    };

    /// GPU compute capability version.
    pub const ComputeCapability = struct {
        major: u32,
        minor: u32,

        /// Check if device supports tensor cores (Volta+).
        pub fn hasTensorCores(self: ComputeCapability) bool {
            return self.major >= 7;
        }

        /// Check if device is Ampere or newer.
        pub fn isAmpereOrNewer(self: ComputeCapability) bool {
            return self.major >= 8;
        }
    };

    /// Tile configuration for matrix operations.
    pub const TileConfig = struct {
        /// Tile height (M dimension).
        m: u32,
        /// Tile width (N dimension).
        n: u32,
        /// Reduction tile size (K dimension).
        k: u32,

        /// Calculate shared memory usage in bytes.
        pub fn sharedMemoryBytes(self: TileConfig, elem_size: u32) u32 {
            // A tile: m x k, B tile: k x n
            return (self.m * self.k + self.k * self.n) * elem_size;
        }

        /// Calculate thread block size.
        pub fn threadCount(self: TileConfig) u32 {
            return self.m * self.n;
        }
    };

    /// Element type for size calculation.
    pub const ElementType = enum {
        f16,
        f32,
        f64,
        i8,
        i32,

        /// Size in bytes.
        pub fn sizeOf(self: ElementType) u32 {
            return switch (self) {
                .f16, .i8 => 2,
                .f32, .i32 => 4,
                .f64 => 8,
            };
        }
    };

    /// Initialize with device info.
    pub fn init(device_info: DeviceInfo) AdaptiveTiling {
        return .{ .device_info = device_info };
    }

    /// Select optimal tile configuration for matrix multiplication C = A * B
    /// where A is (m x k) and B is (k x n), resulting in C (m x n).
    pub fn selectTile(self: AdaptiveTiling, m: u32, n: u32, k: u32, elem_type: ElementType) TileConfig {
        const elem_size = elem_type.sizeOf();
        const warp_size = self.device_info.warp_size;
        const max_shared = self.device_info.max_shared_memory;
        const max_threads = self.device_info.max_threads_per_block;
        const cc = self.device_info.compute_capability;

        // Start with default tile sizes
        var tile_m: u32 = 64;
        var tile_n: u32 = 64;
        var tile_k: u32 = 16;

        // Adjust for matrix shape - favor the larger dimension
        if (m > n * 4) {
            // Tall matrix: favor M dimension
            tile_m = 128;
            tile_n = 32;
        } else if (n > m * 4) {
            // Wide matrix: favor N dimension
            tile_m = 32;
            tile_n = 128;
        }

        // Adjust for compute capability
        if (cc.isAmpereOrNewer()) {
            // Ampere+: can use larger tiles with tensor cores
            tile_m = @min(tile_m * 2, 128);
            tile_n = @min(tile_n * 2, 128);
            tile_k = 32; // Larger K for better tensor core utilization
        } else if (cc.hasTensorCores()) {
            // Volta/Turing: moderate tile sizes
            tile_m = @min(tile_m, 96);
            tile_n = @min(tile_n, 96);
            tile_k = 16;
        }

        // Ensure thread count fits within limits first
        while (tile_m * tile_n > max_threads) {
            if (tile_m >= tile_n and tile_m > 16) {
                tile_m /= 2;
            } else if (tile_n > 16) {
                tile_n /= 2;
            } else {
                break;
            }
        }

        // Prefer warp-aligned tiles when possible, but respect thread limits
        if (tile_m * tile_n <= max_threads) {
            const aligned_m = alignToWarp(tile_m, warp_size);
            const aligned_n = alignToWarp(tile_n, warp_size);
            if (aligned_m * aligned_n <= max_threads) {
                tile_m = aligned_m;
                tile_n = aligned_n;
            }
        }

        // Ensure shared memory fits
        // Shared memory layout: A tile (m x k) + B tile (k x n)
        while (sharedMemUsage(tile_m, tile_n, tile_k, elem_size) > max_shared) {
            if (tile_k > 8) {
                tile_k /= 2;
            } else if (tile_m > tile_n and tile_m > warp_size) {
                tile_m /= 2;
            } else if (tile_n > warp_size) {
                tile_n /= 2;
            } else {
                break; // Can't reduce further
            }
        }

        // Clamp to actual matrix dimensions
        tile_m = @min(tile_m, m);
        tile_n = @min(tile_n, n);
        tile_k = @min(tile_k, k);

        // Ensure minimum sizes
        return .{
            .m = @max(tile_m, 16),
            .n = @max(tile_n, 16),
            .k = @max(tile_k, 8),
        };
    }

    /// Select tile for element-wise operations (simpler than matmul).
    pub fn selectElementwiseTile(self: AdaptiveTiling, total_elements: u32) u32 {
        const max_threads = self.device_info.max_threads_per_block;
        const warp_size = self.device_info.warp_size;

        // Use full block if enough elements, otherwise scale down
        var block_size = @min(total_elements, max_threads);

        // Round down to warp multiple for efficiency
        block_size = (block_size / warp_size) * warp_size;

        return @max(block_size, warp_size);
    }

    /// Calculate shared memory usage for matmul tiles.
    fn sharedMemUsage(tile_m: u32, tile_n: u32, tile_k: u32, elem_size: u32) u32 {
        return (tile_m * tile_k + tile_k * tile_n) * elem_size;
    }

    /// Align value up to warp boundary.
    fn alignToWarp(value: u32, warp_size: u32) u32 {
        if (value < warp_size) return warp_size;
        return ((value + warp_size - 1) / warp_size) * warp_size;
    }
};

// Tests
test "AdaptiveTiling selects optimal tile size for square matrices" {
    const tiling = AdaptiveTiling.init(.{
        .max_threads_per_block = 1024,
        .max_shared_memory = 48 * 1024,
        .warp_size = 32,
        .compute_capability = .{ .major = 8, .minor = 0 },
    });

    const tile = tiling.selectTile(1024, 1024, 1024, .f32);

    // Should select reasonable tile sizes
    try std.testing.expect(tile.m >= 16 and tile.m <= 128);
    try std.testing.expect(tile.n >= 16 and tile.n <= 128);
    try std.testing.expect(tile.k >= 8 and tile.k <= 32);

    // Thread count should not exceed max
    try std.testing.expect(tile.threadCount() <= 1024);
}

test "AdaptiveTiling handles non-square matrices" {
    const tiling = AdaptiveTiling.init(.{
        .max_threads_per_block = 1024,
        .max_shared_memory = 48 * 1024,
        .warp_size = 32,
        .compute_capability = .{ .major = 7, .minor = 5 },
    });

    // Tall skinny matrix (m >> n)
    const tile1 = tiling.selectTile(4096, 64, 256, .f32);
    try std.testing.expect(tile1.m >= tile1.n); // Should favor M dimension

    // Wide flat matrix (n >> m)
    const tile2 = tiling.selectTile(64, 4096, 256, .f32);
    try std.testing.expect(tile2.n >= tile2.m); // Should favor N dimension
}

test "AdaptiveTiling respects shared memory limits" {
    const tiling = AdaptiveTiling.init(.{
        .max_threads_per_block = 1024,
        .max_shared_memory = 16 * 1024, // Limited shared memory
        .warp_size = 32,
        .compute_capability = .{ .major = 6, .minor = 1 },
    });

    const tile = tiling.selectTile(2048, 2048, 2048, .f32);

    // Calculate shared memory usage: (tile_m * tile_k + tile_k * tile_n) * sizeof(f32)
    const shared_bytes = tile.sharedMemoryBytes(4);
    try std.testing.expect(shared_bytes <= 16 * 1024);
}

test "AdaptiveTiling respects thread count limits" {
    const tiling = AdaptiveTiling.init(.{
        .max_threads_per_block = 256, // Very limited
        .max_shared_memory = 48 * 1024,
        .warp_size = 32,
        .compute_capability = .{ .major = 7, .minor = 0 },
    });

    const tile = tiling.selectTile(1024, 1024, 1024, .f32);
    try std.testing.expect(tile.threadCount() <= 256);
}

test "AdaptiveTiling warp alignment" {
    const tiling = AdaptiveTiling.init(.{
        .max_threads_per_block = 1024,
        .max_shared_memory = 48 * 1024,
        .warp_size = 32,
        .compute_capability = .{ .major = 7, .minor = 0 },
    });

    const tile = tiling.selectTile(1000, 1000, 1000, .f32);

    // Tile dimensions should be warp-aligned for efficiency
    try std.testing.expect(tile.m % 32 == 0 or tile.m < 32);
    try std.testing.expect(tile.n % 32 == 0 or tile.n < 32);
}

test "AdaptiveTiling elementwise tile selection" {
    const tiling = AdaptiveTiling.init(.{
        .max_threads_per_block = 1024,
        .max_shared_memory = 48 * 1024,
        .warp_size = 32,
        .compute_capability = .{ .major = 7, .minor = 0 },
    });

    // Large array - should use full block
    const large_tile = tiling.selectElementwiseTile(1_000_000);
    try std.testing.expectEqual(@as(u32, 1024), large_tile);

    // Small array - should scale down
    const small_tile = tiling.selectElementwiseTile(64);
    try std.testing.expect(small_tile <= 64);
    try std.testing.expect(small_tile % 32 == 0); // Warp aligned
}

test {
    std.testing.refAllDecls(@This());
}
