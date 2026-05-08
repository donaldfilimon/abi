//! Data integrity primitives — CRC32 checksums and Bloom filters.

const std = @import("std");

// ============================================================================
// CRC32 Checksum
// ============================================================================

/// CRC32 implementation for data integrity
pub const Crc32 = struct {
    const TABLE: [256]u32 = blk: {
        @setEvalBranchQuota(4000);
        var table: [256]u32 = undefined;
        for (0..256) |i| {
            var crc: u32 = @intCast(i);
            for (0..8) |_| {
                if (crc & 1 == 1) {
                    crc = (crc >> 1) ^ 0xEDB88320;
                } else {
                    crc >>= 1;
                }
            }
            table[i] = crc;
        }
        break :blk table;
    };

    crc: u32 = 0xFFFFFFFF,

    pub fn update(self: *Crc32, data: []const u8) void {
        for (data) |byte| {
            const idx = (self.crc ^ byte) & 0xFF;
            self.crc = (self.crc >> 8) ^ TABLE[idx];
        }
    }

    pub fn finalize(self: *Crc32) u32 {
        return self.crc ^ 0xFFFFFFFF;
    }

    pub fn compute(data: []const u8) u32 {
        var crc = Crc32{};
        crc.update(data);
        return crc.finalize();
    }
};

// ============================================================================
// Bloom Filter
// ============================================================================

/// Bloom filter for fast negative ID lookups
pub const BloomFilter = struct {
    bits: []u8,
    num_hashes: u8,
    allocator: std.mem.Allocator,

    /// Create a bloom filter with optimal parameters for expected_items
    pub fn init(
        allocator: std.mem.Allocator,
        expected_items: usize,
        false_positive_rate: f64,
    ) !BloomFilter {
        // Calculate optimal size: m = -n * ln(p) / (ln(2)^2)
        const n = @as(f64, @floatFromInt(expected_items));
        const ln2 = @log(@as(f64, 2.0));
        const ln2_sq = ln2 * ln2;
        const m = @as(usize, @intFromFloat(-n * @log(false_positive_rate) / ln2_sq));

        // Round up to byte boundary
        const num_bytes = (m + 7) / 8;

        // Calculate optimal number of hashes: k = (m/n) * ln(2)
        const k = @as(u8, @intFromFloat(@as(f64, @floatFromInt(m)) / n * ln2));
        const num_hashes = @max(1, @min(k, 16));

        const bits = try allocator.alloc(u8, num_bytes);
        @memset(bits, 0);

        return .{
            .bits = bits,
            .num_hashes = num_hashes,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BloomFilter) void {
        self.allocator.free(self.bits);
    }

    /// Add an ID to the bloom filter
    pub fn add(self: *BloomFilter, id: u64) void {
        const hashes = self.getHashes(id);
        for (0..self.num_hashes) |i| {
            const bit_idx = hashes[i] % (self.bits.len * 8);
            const byte_idx = bit_idx / 8;
            const bit_offset: u3 = @intCast(bit_idx % 8);
            self.bits[byte_idx] |= @as(u8, 1) << bit_offset;
        }
    }

    /// Check if an ID might be in the set (may have false positives)
    pub fn mightContain(self: *const BloomFilter, id: u64) bool {
        const hashes = self.getHashes(id);
        for (0..self.num_hashes) |i| {
            const bit_idx = hashes[i] % (self.bits.len * 8);
            const byte_idx = bit_idx / 8;
            const bit_offset: u3 = @intCast(bit_idx % 8);
            if ((self.bits[byte_idx] >> bit_offset) & 1 == 0) {
                return false;
            }
        }
        return true;
    }

    fn getHashes(self: *const BloomFilter, id: u64) [16]usize {
        _ = self;
        // Use double hashing: h(i) = h1 + i * h2
        const h1 = std.hash.XxHash3.hash(0, std.mem.asBytes(&id));
        const h2 = std.hash.XxHash3.hash(1, std.mem.asBytes(&id));
        const max_usize_u64: u64 = std.math.maxInt(usize);

        var hashes: [16]usize = undefined;
        for (0..16) |i| {
            const step = @as(u64, @intCast(i)) *% h2;
            hashes[i] = @intCast((h1 +% step) % max_usize_u64);
        }
        return hashes;
    }

    pub fn serialize(self: *const BloomFilter, allocator: std.mem.Allocator) ![]u8 {
        var buf = std.ArrayListUnmanaged(u8).empty;
        errdefer buf.deinit(allocator);

        // Header: num_hashes (1) + size (4)
        try buf.append(allocator, self.num_hashes);
        try buf.appendSlice(allocator, &std.mem.toBytes(@as(u32, @intCast(self.bits.len))));
        try buf.appendSlice(allocator, self.bits);

        return buf.toOwnedSlice(allocator);
    }

    pub fn deserialize(allocator: std.mem.Allocator, data: []const u8) !BloomFilter {
        if (data.len < 5) return error.InvalidBloomFilter;

        const num_hashes = data[0];
        const size = std.mem.readInt(u32, data[1..5], .little);

        if (data.len < 5 + size) return error.InvalidBloomFilter;

        const bits = try allocator.dupe(u8, data[5..][0..size]);

        return .{
            .bits = bits,
            .num_hashes = num_hashes,
            .allocator = allocator,
        };
    }
};
