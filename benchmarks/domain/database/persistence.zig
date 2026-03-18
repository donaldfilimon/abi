//! Persistence Layer Benchmarks
//!
//! Measures performance of the ABI persistence subsystems:
//!
//! - **SegmentLog append throughput**: Append 10K blocks of varying sizes (64B, 1KB, 64KB).
//! - **WAL flush/recover cycle**: Flush 10K WAL entries to disk, recover. Entries/sec for both.
//! - **RLE compression ratio**: Compress 10K random f32 vectors (128 dims). Ratio + throughput.
//!
//! These benchmarks exercise the raw I/O and encoding paths that underpin
//! the vector database durability layer.

const std = @import("std");
const abi = @import("abi");
const core = @import("../../core/mod.zig");
const framework = @import("../../system/framework.zig");

// ============================================================================
// Timing helpers (Zig 0.16 compatible — std.time.Instant does NOT exist)
// ============================================================================

fn monotonicNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

// ============================================================================
// SegmentLog append benchmark
//
// We replicate the on-disk frame format (u32 LE length + payload) used by
// `src/core/database/block/segment_log.zig` to measure raw sequential
// append throughput without pulling in internal module dependencies.
// ============================================================================

fn benchSegmentLogAppend(allocator: std.mem.Allocator, block_size: usize, num_blocks: usize) !BenchStats {
    const tmp_path = "/tmp/abi_bench_segment_log.seg";
    std.posix.unlink(tmp_path) catch {};
    defer std.posix.unlink(tmp_path) catch {};

    // Generate payload data
    const payload = try allocator.alloc(u8, block_size);
    defer allocator.free(payload);
    // Fill with deterministic pattern
    for (payload, 0..) |*b, i| {
        b.* = @truncate(i);
    }

    // Encode a minimal block header (87 bytes) matching codec.zig format
    const header_size: usize = 32 + 4 + 4 + 32 + 8 + 4 + 2 + 1; // = 87
    const encoded_size = header_size + block_size;
    const encoded = try allocator.alloc(u8, encoded_size);
    defer allocator.free(encoded);

    // Fill header region with zeros (valid enough for I/O benchmarking)
    @memset(encoded[0..header_size], 0);
    // Copy payload
    @memcpy(encoded[header_size..], payload);

    const start = monotonicNs();

    // Open the file once and append all blocks
    const fd = std.posix.open(
        tmp_path,
        .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true },
        0o644,
    ) catch return error.OpenFailed;
    defer std.posix.close(fd);

    for (0..num_blocks) |_| {
        // Write length prefix (u32 LE)
        const len: u32 = @intCast(encoded.len);
        const len_bytes = std.mem.toBytes(len);
        try posixWriteAll(fd, &len_bytes);
        // Write encoded block
        try posixWriteAll(fd, encoded);
    }

    const elapsed = monotonicNs() - start;
    const total_bytes = num_blocks * (4 + encoded_size);

    return .{
        .elapsed_ns = elapsed,
        .ops = num_blocks,
        .total_bytes = total_bytes,
    };
}

// ============================================================================
// WAL flush/recover benchmark
//
// Mirrors the WalWriter serialize/flush + recover/deserialize cycle from
// `src/core/database/distributed/wal.zig`.
// ============================================================================

const WalEntryCompat = extern struct {
    sequence: u64,
    entry_type: u8,
    _pad0: [3]u8 = .{ 0, 0, 0 },
    timestamp: i64,
    vector_id: u64,
    dimension: u32,
    data_offset: u32,
    data_len: u32,
    crc32: u32,
};

const WalHeaderCompat = extern struct {
    magic: [4]u8 = .{ 'W', 'A', 'L', 'X' },
    version: u16 = 1,
    _pad0: [2]u8 = .{ 0, 0 },
    node_id: u64 = 0,
    created_at: i64 = 0,
    entry_count: u64 = 0,
    last_sequence: u64 = 0,
};

fn benchWalFlush(allocator: std.mem.Allocator, num_entries: usize, dimension: u32) !BenchStats {
    const tmp_path = "/tmp/abi_bench_wal_flush.wal";
    std.posix.unlink(tmp_path) catch {};
    defer std.posix.unlink(tmp_path) catch {};

    // Build entries and data buffer
    const entries = try allocator.alloc(WalEntryCompat, num_entries);
    defer allocator.free(entries);

    const data_per_entry = dimension * @sizeOf(f32);
    const data_buf = try allocator.alloc(u8, num_entries * data_per_entry);
    defer allocator.free(data_buf);

    // Fill with deterministic vector data
    const floats = std.mem.bytesAsSlice(f32, data_buf);
    for (floats, 0..) |*f, i| {
        f.* = @as(f32, @floatFromInt(i % 1000)) * 0.001;
    }

    for (entries, 0..) |*e, i| {
        e.* = .{
            .sequence = @intCast(i + 1),
            .entry_type = 0x01, // insert
            .timestamp = 1700000000,
            .vector_id = @intCast(i),
            .dimension = dimension,
            .data_offset = @intCast(i * data_per_entry),
            .data_len = @intCast(data_per_entry),
            .crc32 = std.hash.Crc32.hash(data_buf[i * data_per_entry ..][0..data_per_entry]),
        };
    }

    // Serialize: header + entries + data
    const header_size = @sizeOf(WalHeaderCompat);
    const entry_size = @sizeOf(WalEntryCompat);
    const entries_total = num_entries * entry_size;
    const total_size = header_size + entries_total + data_buf.len;

    const buffer = try allocator.alloc(u8, total_size);
    defer allocator.free(buffer);

    var hdr = WalHeaderCompat{
        .node_id = 1,
        .entry_count = @intCast(num_entries),
        .last_sequence = @intCast(num_entries),
    };

    // --- Flush benchmark ---
    const flush_start = monotonicNs();

    // Serialize header
    const hdr_bytes = std.mem.asBytes(&hdr);
    @memcpy(buffer[0..header_size], hdr_bytes);

    // Serialize entries
    var offset: usize = header_size;
    for (entries) |entry| {
        const e_bytes = std.mem.asBytes(&entry);
        @memcpy(buffer[offset..][0..entry_size], e_bytes);
        offset += entry_size;
    }

    // Serialize data
    @memcpy(buffer[offset..][0..data_buf.len], data_buf);

    // Write to disk
    const fd = std.posix.open(
        tmp_path,
        .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true },
        0o644,
    ) catch return error.OpenFailed;

    try posixWriteAll(fd, buffer);
    std.posix.close(fd);

    const flush_elapsed = monotonicNs() - flush_start;

    // --- Recover benchmark ---
    const recover_start = monotonicNs();

    const rfd = std.posix.open(
        tmp_path,
        .{ .ACCMODE = .RDONLY },
        0,
    ) catch return error.OpenFailed;

    // Determine file size
    const end_off = std.posix.lseek(rfd, 0, .END) catch return error.SeekFailed;
    const file_size: usize = @intCast(end_off);
    _ = std.posix.lseek(rfd, 0, .SET) catch return error.SeekFailed;

    const read_buf = try allocator.alloc(u8, file_size);
    defer allocator.free(read_buf);

    try posixReadAll(rfd, read_buf);
    std.posix.close(rfd);

    // Deserialize header
    const recovered_hdr = std.mem.bytesToValue(WalHeaderCompat, read_buf[0..header_size]);
    std.mem.doNotOptimizeAway(&recovered_hdr);

    // Deserialize entries
    var r_offset: usize = header_size;
    var recovered_count: usize = 0;
    while (recovered_count < recovered_hdr.entry_count and r_offset + entry_size <= read_buf.len) {
        const recovered_entry = std.mem.bytesToValue(WalEntryCompat, read_buf[r_offset..][0..entry_size]);
        std.mem.doNotOptimizeAway(&recovered_entry);
        r_offset += entry_size;
        recovered_count += 1;
    }

    const recover_elapsed = monotonicNs() - recover_start;

    // Return combined stats; caller distinguishes flush vs recover
    _ = recover_elapsed;
    _ = flush_elapsed;

    return .{
        .elapsed_ns = flush_elapsed,
        .ops = num_entries,
        .total_bytes = total_size,
        // Store recover time in a separate field
        .extra_ns = recover_elapsed,
    };
}

// ============================================================================
// RLE Compression benchmark
//
// Mirrors the RLE scheme from `src/core/database/block/compression.zig`:
//   - Runs of 3+ identical bytes: [0xFF, count, value]
//   - Literal 0xFF bytes escaped: [0xFF, 0x01, 0xFF]
//   - All other bytes emitted verbatim
// ============================================================================

const RLE_MARKER: u8 = 0xFF;

fn rleCompress(allocator: std.mem.Allocator, data: []const u8) ![]u8 {
    if (data.len == 0) {
        return allocator.alloc(u8, 0);
    }

    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    var i: usize = 0;
    while (i < data.len) {
        const byte = data[i];

        var run_len: usize = 1;
        while (i + run_len < data.len and data[i + run_len] == byte and run_len < 255) {
            run_len += 1;
        }

        if (byte == RLE_MARKER) {
            if (run_len >= 3) {
                try out.append(allocator, RLE_MARKER);
                try out.append(allocator, @intCast(run_len));
                try out.append(allocator, RLE_MARKER);
            } else {
                for (0..run_len) |_| {
                    try out.append(allocator, RLE_MARKER);
                    try out.append(allocator, 0x01);
                    try out.append(allocator, RLE_MARKER);
                }
            }
            i += run_len;
        } else if (run_len >= 3) {
            try out.append(allocator, RLE_MARKER);
            try out.append(allocator, @intCast(run_len));
            try out.append(allocator, byte);
            i += run_len;
        } else {
            for (0..run_len) |_| {
                try out.append(allocator, byte);
            }
            i += run_len;
        }
    }

    return out.toOwnedSlice(allocator);
}

fn rleDecompress(allocator: std.mem.Allocator, data: []const u8) ![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    var i: usize = 0;
    while (i < data.len) {
        if (data[i] == RLE_MARKER) {
            if (i + 2 >= data.len) return error.CorruptedData;
            const count = data[i + 1];
            const value = data[i + 2];
            if (count == 0) return error.CorruptedData;
            for (0..count) |_| {
                try out.append(allocator, value);
            }
            i += 3;
        } else {
            try out.append(allocator, data[i]);
            i += 1;
        }
    }

    return out.toOwnedSlice(allocator);
}

fn benchRleCompression(allocator: std.mem.Allocator, num_vectors: usize, dimension: usize) !CompressionStats {
    // Generate random f32 vectors as byte data
    const total_floats = num_vectors * dimension;
    const float_data = try allocator.alloc(f32, total_floats);
    defer allocator.free(float_data);

    // Use a deterministic PRNG for reproducible results
    var prng = std.Random.DefaultPrng.init(0xDEADBEEF);
    const random = prng.random();
    for (float_data) |*f| {
        f.* = random.float(f32) * 2.0 - 1.0; // [-1, 1]
    }

    const raw_bytes = std.mem.sliceAsBytes(float_data);
    const raw_size = raw_bytes.len;

    // Compress
    const compress_start = monotonicNs();
    const compressed = try rleCompress(allocator, raw_bytes);
    defer allocator.free(compressed);
    const compress_elapsed = monotonicNs() - compress_start;

    // Decompress
    const decompress_start = monotonicNs();
    const decompressed = try rleDecompress(allocator, compressed);
    defer allocator.free(decompressed);
    const decompress_elapsed = monotonicNs() - decompress_start;

    // Verify round-trip
    if (!std.mem.eql(u8, raw_bytes, decompressed)) {
        return error.RoundTripFailed;
    }

    return .{
        .raw_size = raw_size,
        .compressed_size = compressed.len,
        .compress_ns = compress_elapsed,
        .decompress_ns = decompress_elapsed,
        .num_vectors = num_vectors,
        .dimension = dimension,
    };
}

// ============================================================================
// Result types
// ============================================================================

const BenchStats = struct {
    elapsed_ns: u64,
    ops: usize,
    total_bytes: usize,
    extra_ns: u64 = 0,

    fn opsPerSecond(self: BenchStats) f64 {
        if (self.elapsed_ns == 0) return 0;
        return @as(f64, @floatFromInt(self.ops)) * 1_000_000_000.0 / @as(f64, @floatFromInt(self.elapsed_ns));
    }

    fn throughputMBps(self: BenchStats) f64 {
        if (self.elapsed_ns == 0) return 0;
        return @as(f64, @floatFromInt(self.total_bytes)) * 1_000.0 / @as(f64, @floatFromInt(self.elapsed_ns));
    }

    fn extraOpsPerSecond(self: BenchStats) f64 {
        if (self.extra_ns == 0) return 0;
        return @as(f64, @floatFromInt(self.ops)) * 1_000_000_000.0 / @as(f64, @floatFromInt(self.extra_ns));
    }
};

const CompressionStats = struct {
    raw_size: usize,
    compressed_size: usize,
    compress_ns: u64,
    decompress_ns: u64,
    num_vectors: usize,
    dimension: usize,

    fn ratio(self: CompressionStats) f64 {
        if (self.compressed_size == 0) return 0;
        return @as(f64, @floatFromInt(self.raw_size)) / @as(f64, @floatFromInt(self.compressed_size));
    }

    fn compressThroughputMBps(self: CompressionStats) f64 {
        if (self.compress_ns == 0) return 0;
        return @as(f64, @floatFromInt(self.raw_size)) * 1_000.0 / @as(f64, @floatFromInt(self.compress_ns));
    }

    fn decompressThroughputMBps(self: CompressionStats) f64 {
        if (self.decompress_ns == 0) return 0;
        return @as(f64, @floatFromInt(self.raw_size)) * 1_000.0 / @as(f64, @floatFromInt(self.decompress_ns));
    }
};

// ============================================================================
// POSIX I/O helpers
// ============================================================================

fn posixWriteAll(fd: std.posix.fd_t, buf: []const u8) !void {
    var written: usize = 0;
    while (written < buf.len) {
        const n = std.posix.write(fd, buf[written..]) catch return error.WriteFailed;
        if (n == 0) return error.WriteFailed;
        written += n;
    }
}

fn posixReadAll(fd: std.posix.fd_t, buf: []u8) !void {
    var total: usize = 0;
    while (total < buf.len) {
        const n = std.posix.read(fd, buf[total..]) catch return error.ReadFailed;
        if (n == 0) return error.ReadFailed;
        total += n;
    }
}

// ============================================================================
// Public entry point
// ============================================================================

pub fn runPersistenceBenchmarks(allocator: std.mem.Allocator, config: core.config.DatabaseBenchConfig) !void {
    _ = config;

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("                    PERSISTENCE LAYER BENCHMARKS\n", .{});
    std.debug.print("================================================================================\n\n", .{});

    // ── SegmentLog append throughput ────────────────────────────────────────
    std.debug.print("[persistence/segment_log]\n", .{});
    std.debug.print("-" ** 80 ++ "\n", .{});

    const block_sizes = [_]struct { size: usize, label: []const u8 }{
        .{ .size = 64, .label = "64B" },
        .{ .size = 1024, .label = "1KB" },
        .{ .size = 65536, .label = "64KB" },
    };

    for (block_sizes) |bs| {
        const stats = try benchSegmentLogAppend(allocator, bs.size, 10_000);
        std.debug.print(
            "  segment_log_append_{s:<6}       {d:>12.0} blocks/sec  {d:>8.1} MB/s  ({d}ns total)\n",
            .{ bs.label, stats.opsPerSecond(), stats.throughputMBps(), stats.elapsed_ns },
        );
    }

    // ── WAL flush/recover cycle ────────────────────────────────────────────
    std.debug.print("\n[persistence/wal]\n", .{});
    std.debug.print("-" ** 80 ++ "\n", .{});

    const wal_stats = try benchWalFlush(allocator, 10_000, 128);
    std.debug.print(
        "  wal_flush_10k                  {d:>12.0} entries/sec  {d:>8.1} MB/s\n",
        .{ wal_stats.opsPerSecond(), wal_stats.throughputMBps() },
    );
    std.debug.print(
        "  wal_recover_10k                {d:>12.0} entries/sec\n",
        .{wal_stats.extraOpsPerSecond()},
    );

    // ── RLE compression ────────────────────────────────────────────────────
    std.debug.print("\n[persistence/rle_compression]\n", .{});
    std.debug.print("-" ** 80 ++ "\n", .{});

    const compression_stats = try benchRleCompression(allocator, 10_000, 128);
    std.debug.print(
        "  rle_compress_10k_128d          ratio={d:>5.3}x  {d:>8.1} MB/s  raw={d} compressed={d}\n",
        .{
            compression_stats.ratio(),
            compression_stats.compressThroughputMBps(),
            compression_stats.raw_size,
            compression_stats.compressed_size,
        },
    );
    std.debug.print(
        "  rle_decompress_10k_128d        {d:>35.1} MB/s\n",
        .{compression_stats.decompressThroughputMBps()},
    );

    std.debug.print("\n", .{});
}

test "persistence benchmark smoke test" {
    const allocator = std.testing.allocator;

    // Small-scale smoke tests to verify the benchmarks don't crash
    const seg_stats = try benchSegmentLogAppend(allocator, 64, 10);
    try std.testing.expect(seg_stats.ops == 10);

    const wal_stats = try benchWalFlush(allocator, 10, 4);
    try std.testing.expect(wal_stats.ops == 10);

    const comp_stats = try benchRleCompression(allocator, 10, 16);
    try std.testing.expect(comp_stats.num_vectors == 10);
    try std.testing.expect(comp_stats.dimension == 16);
}
