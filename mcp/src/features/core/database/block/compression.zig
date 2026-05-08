//! Block compression strategies for the database layer.
//!
//! - `.none` — passthrough (no compression).
//! - `.lz4`  — real LZ4 block format compression (token/literal/match sequences).
//! - `.zstd` — simple RLE fallback (not real zstd; placeholder for future work).

const std = @import("std");

pub const CompressionStrategy = enum {
    none,
    lz4,
    zstd,
};

pub const CompressionError = error{
    NotImplemented,
    OutOfMemory,
    CorruptedData,
};

// ============================================================================
// Public entry points
// ============================================================================

pub fn compress(allocator: std.mem.Allocator, data: []const u8, strategy: CompressionStrategy) CompressionError![]u8 {
    return switch (strategy) {
        .none => allocator.dupe(u8, data) catch return error.OutOfMemory,
        .lz4 => lz4Compress(allocator, data),
        .zstd => compressRle(allocator, data),
    };
}

pub fn decompress(allocator: std.mem.Allocator, data: []const u8, strategy: CompressionStrategy) CompressionError![]u8 {
    return switch (strategy) {
        .none => allocator.dupe(u8, data) catch return error.OutOfMemory,
        .lz4 => lz4Decompress(allocator, data),
        .zstd => decompressRle(allocator, data),
    };
}

// ============================================================================
// LZ4 block format implementation
// ============================================================================

const LZ4_HASH_BITS = 16;
const LZ4_HASH_SIZE = 1 << LZ4_HASH_BITS; // 65536
const LZ4_MIN_MATCH = 4;
const LZ4_MAX_OFFSET = 65535;

/// Hash 4 bytes into a 16-bit index for the hash table.
fn lz4Hash(data: []const u8) u16 {
    const v = std.mem.readInt(u32, data[0..4], .little);
    // Knuth multiplicative hash, take top 16 bits.
    return @truncate((v *% 2654435761) >> (32 - LZ4_HASH_BITS));
}

/// Write an LZ4 variable-length field: if `length` >= 15, the token already
/// holds 15 and we emit additional bytes (255 each until remainder < 255).
fn writeVarLen(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, length: usize) CompressionError!void {
    if (length < 15) return; // already encoded in the token nibble
    var remaining = length - 15;
    while (remaining >= 255) {
        out.append(allocator, 255) catch return error.OutOfMemory;
        remaining -= 255;
    }
    out.append(allocator, @intCast(remaining)) catch return error.OutOfMemory;
}

fn lz4Compress(allocator: std.mem.Allocator, data: []const u8) CompressionError![]u8 {
    if (data.len == 0) {
        return allocator.alloc(u8, 0) catch return error.OutOfMemory;
    }

    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    // Reserve generous space — worst case is slightly larger than input.
    out.ensureTotalCapacity(allocator, data.len + (data.len / 255) + 16) catch return error.OutOfMemory;

    // Hash table: maps 4-byte sequence hash -> position in input.
    // Initialised to 0; we treat position 0 specially (always miss on first check).
    var hash_table: [LZ4_HASH_SIZE]u32 = undefined;
    @memset(&hash_table, 0);

    var anchor: usize = 0; // start of pending literals
    var ip: usize = 0; // current scan position

    // Need at least 4 bytes remaining to attempt a match + 5 bytes for last-literal safety.
    const limit = if (data.len > 12) data.len - 5 else 0;

    while (ip < limit) {
        if (ip + LZ4_MIN_MATCH > data.len) break;

        const h = lz4Hash(data[ip..]);
        const ref = hash_table[h];
        hash_table[h] = @intCast(ip);

        // Check for a valid match: same 4 bytes, offset in range, not position 0 on first use.
        if (ref > 0 and ip - ref <= LZ4_MAX_OFFSET and ip - ref > 0 and
            std.mem.eql(u8, data[ref .. ref + LZ4_MIN_MATCH], data[ip .. ip + LZ4_MIN_MATCH]))
        {
            // Extend the match forwards.
            var match_len: usize = LZ4_MIN_MATCH;
            while (ip + match_len < data.len and data[ref + match_len] == data[ip + match_len]) {
                match_len += 1;
            }

            const lit_len = ip - anchor;
            const offset: u16 = @intCast(ip - ref);

            // Token byte: high nibble = literal length (capped 15), low nibble = match length - 4 (capped 15).
            const lit_token: u8 = if (lit_len >= 15) 15 else @intCast(lit_len);
            const match_extra = match_len - LZ4_MIN_MATCH;
            const match_token: u8 = if (match_extra >= 15) 15 else @intCast(match_extra);
            const token: u8 = (lit_token << 4) | match_token;

            out.append(allocator, token) catch return error.OutOfMemory;

            // Literal length extension.
            try writeVarLen(&out, allocator, lit_len);

            // Literal bytes.
            out.appendSlice(allocator, data[anchor .. anchor + lit_len]) catch return error.OutOfMemory;

            // Match offset (2 bytes LE).
            out.append(allocator, @intCast(offset & 0xFF)) catch return error.OutOfMemory;
            out.append(allocator, @intCast(offset >> 8)) catch return error.OutOfMemory;

            // Match length extension.
            try writeVarLen(&out, allocator, match_extra);

            ip += match_len;
            anchor = ip;
        } else {
            ip += 1;
        }
    }

    // Emit remaining literals (the last sequence has no match).
    {
        const lit_len = data.len - anchor;
        const lit_token: u8 = if (lit_len >= 15) 15 else @intCast(lit_len);
        const token: u8 = lit_token << 4; // match nibble = 0 (no match)
        out.append(allocator, token) catch return error.OutOfMemory;
        try writeVarLen(&out, allocator, lit_len);
        out.appendSlice(allocator, data[anchor..]) catch return error.OutOfMemory;
    }

    return out.toOwnedSlice(allocator) catch return error.OutOfMemory;
}

fn lz4Decompress(allocator: std.mem.Allocator, data: []const u8) CompressionError![]u8 {
    if (data.len == 0) {
        return allocator.alloc(u8, 0) catch return error.OutOfMemory;
    }

    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    var i: usize = 0;

    while (i < data.len) {
        // Read token.
        if (i >= data.len) return error.CorruptedData;
        const token = data[i];
        i += 1;

        // --- Literals ---
        var lit_len: usize = token >> 4;
        if (lit_len == 15) {
            while (true) {
                if (i >= data.len) return error.CorruptedData;
                const extra = data[i];
                i += 1;
                lit_len += extra;
                if (extra != 255) break;
            }
        }

        if (i + lit_len > data.len) return error.CorruptedData;
        out.appendSlice(allocator, data[i .. i + lit_len]) catch return error.OutOfMemory;
        i += lit_len;

        // If we've consumed all input, this was the last (literal-only) sequence.
        if (i >= data.len) break;

        // --- Match ---
        if (i + 1 >= data.len) return error.CorruptedData;
        const offset: u16 = @as(u16, data[i]) | (@as(u16, data[i + 1]) << 8);
        i += 2;

        if (offset == 0) return error.CorruptedData;
        if (offset > out.items.len) return error.CorruptedData;

        var match_len: usize = (token & 0x0F) + LZ4_MIN_MATCH;
        if ((token & 0x0F) == 15) {
            while (true) {
                if (i >= data.len) return error.CorruptedData;
                const extra = data[i];
                i += 1;
                match_len += extra;
                if (extra != 255) break;
            }
        }

        // Copy from output buffer — byte-by-byte to handle overlapping matches.
        const match_start = out.items.len - offset;
        out.ensureTotalCapacity(allocator, out.items.len + match_len) catch return error.OutOfMemory;
        for (0..match_len) |j| {
            out.append(allocator, out.items[match_start + j]) catch return error.OutOfMemory;
        }
    }

    return out.toOwnedSlice(allocator) catch return error.OutOfMemory;
}

// ============================================================================
// RLE implementation (used as .zstd placeholder)
// ============================================================================

/// Marker byte used by the RLE scheme.
const RLE_MARKER: u8 = 0xFF;

fn compressRle(allocator: std.mem.Allocator, data: []const u8) CompressionError![]u8 {
    if (data.len == 0) {
        return allocator.alloc(u8, 0) catch return error.OutOfMemory;
    }

    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    var i: usize = 0;
    while (i < data.len) {
        const byte = data[i];

        // Count the run length starting at i.
        var run_len: usize = 1;
        while (i + run_len < data.len and data[i + run_len] == byte and run_len < 255) {
            run_len += 1;
        }

        if (byte == RLE_MARKER) {
            if (run_len >= 3) {
                out.append(allocator, RLE_MARKER) catch return error.OutOfMemory;
                out.append(allocator, @intCast(run_len)) catch return error.OutOfMemory;
                out.append(allocator, RLE_MARKER) catch return error.OutOfMemory;
            } else {
                for (0..run_len) |_| {
                    out.append(allocator, RLE_MARKER) catch return error.OutOfMemory;
                    out.append(allocator, 0x01) catch return error.OutOfMemory;
                    out.append(allocator, RLE_MARKER) catch return error.OutOfMemory;
                }
            }
            i += run_len;
        } else if (run_len >= 3) {
            out.append(allocator, RLE_MARKER) catch return error.OutOfMemory;
            out.append(allocator, @intCast(run_len)) catch return error.OutOfMemory;
            out.append(allocator, byte) catch return error.OutOfMemory;
            i += run_len;
        } else {
            for (0..run_len) |_| {
                out.append(allocator, byte) catch return error.OutOfMemory;
            }
            i += run_len;
        }
    }

    return out.toOwnedSlice(allocator) catch return error.OutOfMemory;
}

fn decompressRle(allocator: std.mem.Allocator, data: []const u8) CompressionError![]u8 {
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
                out.append(allocator, value) catch return error.OutOfMemory;
            }
            i += 3;
        } else {
            out.append(allocator, data[i]) catch return error.OutOfMemory;
            i += 1;
        }
    }

    return out.toOwnedSlice(allocator) catch return error.OutOfMemory;
}

// ============================================================================
// Tests
// ============================================================================

test "LZ4 round-trip" {
    const allocator = std.testing.allocator;

    const input = "Hello world! Hello world! Hello world! Hello world!";
    const compressed = try compress(allocator, input, .lz4);
    defer allocator.free(compressed);

    const decompressed = try decompress(allocator, compressed, .lz4);
    defer allocator.free(decompressed);

    try std.testing.expectEqualSlices(u8, input, decompressed);
}

test "LZ4 compresses repetitive data" {
    const allocator = std.testing.allocator;

    // Highly repetitive: 200 copies of "ABCD".
    var input: [800]u8 = undefined;
    for (0..200) |i| {
        input[i * 4 + 0] = 'A';
        input[i * 4 + 1] = 'B';
        input[i * 4 + 2] = 'C';
        input[i * 4 + 3] = 'D';
    }

    const compressed = try compress(allocator, &input, .lz4);
    defer allocator.free(compressed);

    // Compressed output should be smaller than the 800-byte input.
    try std.testing.expect(compressed.len < input.len);

    const decompressed = try decompress(allocator, compressed, .lz4);
    defer allocator.free(decompressed);

    try std.testing.expectEqualSlices(u8, &input, decompressed);
}

test "LZ4 handles incompressible data" {
    const allocator = std.testing.allocator;

    // Pseudo-random data — unlikely to compress well but must round-trip.
    var input: [256]u8 = undefined;
    var state: u32 = 0xDEADBEEF;
    for (&input) |*b| {
        state = state *% 1103515245 +% 12345;
        b.* = @truncate(state >> 16);
    }

    const compressed = try compress(allocator, &input, .lz4);
    defer allocator.free(compressed);

    const decompressed = try decompress(allocator, compressed, .lz4);
    defer allocator.free(decompressed);

    try std.testing.expectEqualSlices(u8, &input, decompressed);
}

test "LZ4 empty data" {
    const allocator = std.testing.allocator;

    const compressed = try compress(allocator, &[_]u8{}, .lz4);
    defer allocator.free(compressed);

    const decompressed = try decompress(allocator, compressed, .lz4);
    defer allocator.free(decompressed);

    try std.testing.expectEqual(@as(usize, 0), decompressed.len);
}

test "RLE round-trip" {
    const allocator = std.testing.allocator;

    const input = "AAABBBCCDDDDDD";
    const compressed = try compress(allocator, input, .zstd);
    defer allocator.free(compressed);

    const decompressed = try decompress(allocator, compressed, .zstd);
    defer allocator.free(decompressed);

    try std.testing.expectEqualSlices(u8, input, decompressed);
}

test "RLE marker escape round-trip" {
    const allocator = std.testing.allocator;

    const input = &[_]u8{ 0x01, 0xFF, 0x02, 0xFF, 0xFF, 0xFF, 0xFF, 0x03 };
    const compressed = try compress(allocator, input, .zstd);
    defer allocator.free(compressed);

    const decompressed = try decompress(allocator, compressed, .zstd);
    defer allocator.free(decompressed);

    try std.testing.expectEqualSlices(u8, input, decompressed);
}

test "RLE empty data" {
    const allocator = std.testing.allocator;

    const compressed = try compress(allocator, &[_]u8{}, .zstd);
    defer allocator.free(compressed);

    const decompressed = try decompress(allocator, compressed, .zstd);
    defer allocator.free(decompressed);

    try std.testing.expectEqual(@as(usize, 0), decompressed.len);
}

test "compression none passthrough" {
    const allocator = std.testing.allocator;
    const input = "passthrough";

    const out = try compress(allocator, input, .none);
    defer allocator.free(out);

    try std.testing.expectEqualSlices(u8, input, out);
}
