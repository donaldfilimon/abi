//! Compression utilities for the unified storage format.
//!
//! Supports multiple algorithms optimized for different use cases:
//! - None: Maximum speed, no compression
//! - LZ4: Very fast compression/decompression, moderate ratio
//! - ZSTD: Excellent compression ratio, still fast
//! - RLE: Run-length encoding for sparse data

const std = @import("std");

pub const CompressionType = enum(u8) {
    none = 0,
    lz4 = 1,
    zstd = 2,
    rle = 3,
};

pub const CompressionError = error{
    InvalidInput,
    OutputTooSmall,
    CorruptedData,
    UnsupportedAlgorithm,
    OutOfMemory,
};

/// Compress data using the specified algorithm
pub fn compress(
    allocator: std.mem.Allocator,
    input: []const u8,
    comp_type: CompressionType,
) CompressionError![]u8 {
    return switch (comp_type) {
        .none => {
            const output = allocator.alloc(u8, input.len) catch return error.OutOfMemory;
            @memcpy(output, input);
            return output;
        },
        .lz4 => compressLz4(allocator, input),
        .zstd => compressZstd(allocator, input),
        .rle => compressRle(allocator, input),
    };
}

/// Decompress data using the specified algorithm
pub fn decompress(
    allocator: std.mem.Allocator,
    input: []const u8,
    expected_size: usize,
    comp_type: CompressionType,
) CompressionError![]u8 {
    return switch (comp_type) {
        .none => {
            const output = allocator.alloc(u8, input.len) catch return error.OutOfMemory;
            @memcpy(output, input);
            return output;
        },
        .lz4 => decompressLz4(allocator, input, expected_size),
        .zstd => decompressZstd(allocator, input, expected_size),
        .rle => decompressRle(allocator, input, expected_size),
    };
}

/// LZ4-style fast compression
/// Format: sequences of (literal_len, match_offset, match_len)
fn compressLz4(allocator: std.mem.Allocator, input: []const u8) CompressionError![]u8 {
    if (input.len == 0) {
        const output = allocator.alloc(u8, 0) catch return error.OutOfMemory;
        return output;
    }

    // Worst case: no compression + overhead
    var output = std.ArrayListUnmanaged(u8).empty;
    errdefer output.deinit(allocator);

    // Use hash table for fast matching
    var hash_table: [4096]u32 = undefined;
    @memset(&hash_table, 0);

    var src_pos: usize = 0;
    var anchor: usize = 0;

    while (src_pos + 4 <= input.len) {
        // Hash current position
        const hash_val = hash4(input[src_pos..][0..4]);
        const match_pos = hash_table[hash_val];
        hash_table[hash_val] = @intCast(src_pos);

        // Check for match
        if (match_pos > 0 and src_pos - match_pos < 65535 and
            src_pos >= match_pos and
            std.mem.eql(u8, input[match_pos..][0..4], input[src_pos..][0..4]))
        {
            // Extend match first to find total length
            var match_len: usize = 4;
            while (src_pos + match_len < input.len and
                match_pos + match_len < src_pos and
                input[src_pos + match_len] == input[match_pos + match_len])
            {
                match_len += 1;
            }

            // Encode literals + match (match_len-4 is stored in token, minimum match is 4)
            try encodeLz4Sequence(&output, allocator, input[anchor..src_pos], @intCast(src_pos - match_pos), @intCast(match_len - 4));

            src_pos += match_len;
            anchor = src_pos;
        } else {
            src_pos += 1;
        }
    }

    // Encode remaining literals
    if (anchor < input.len) {
        try encodeLz4Sequence(&output, allocator, input[anchor..], 0, 0);
    }

    return output.toOwnedSlice(allocator) catch return error.OutOfMemory;
}

fn hash4(data: *const [4]u8) usize {
    const val = std.mem.readInt(u32, data, .little);
    return @intCast((val *% 2654435761) >> 20);
}

fn encodeLz4Sequence(
    output: *std.ArrayListUnmanaged(u8),
    allocator: std.mem.Allocator,
    literals: []const u8,
    offset: u16,
    match_len: u8,
) !void {
    // Token byte: high 4 bits = literal length, low 4 bits = match length
    var lit_len = literals.len;
    var token: u8 = 0;

    if (lit_len >= 15) {
        token = 0xF0;
    } else {
        token = @as(u8, @intCast(lit_len)) << 4;
    }

    if (match_len >= 15) {
        token |= 0x0F;
    } else {
        token |= match_len;
    }

    try output.append(allocator, token);

    // Extended literal length
    if (lit_len >= 15) {
        lit_len -= 15;
        while (lit_len >= 255) {
            try output.append(allocator, 255);
            lit_len -= 255;
        }
        try output.append(allocator, @intCast(lit_len));
    }

    // Literals
    try output.appendSlice(allocator, literals);

    // Match offset (little-endian)
    if (offset > 0) {
        try output.append(allocator, @truncate(offset));
        try output.append(allocator, @truncate(offset >> 8));
    }
}

fn decompressLz4(allocator: std.mem.Allocator, input: []const u8, expected_size: usize) CompressionError![]u8 {
    var output = allocator.alloc(u8, expected_size) catch return error.OutOfMemory;
    errdefer allocator.free(output);

    var src_pos: usize = 0;
    var dst_pos: usize = 0;

    while (src_pos < input.len and dst_pos < expected_size) {
        if (src_pos >= input.len) break;

        const token = input[src_pos];
        src_pos += 1;

        // Decode literal length
        var lit_len: usize = (token >> 4);
        if (lit_len == 15) {
            while (src_pos < input.len) {
                const extra = input[src_pos];
                src_pos += 1;
                lit_len += extra;
                if (extra != 255) break;
            }
        }

        // Copy literals
        if (lit_len > 0) {
            if (src_pos + lit_len > input.len or dst_pos + lit_len > expected_size) {
                return error.CorruptedData;
            }
            @memcpy(output[dst_pos..][0..lit_len], input[src_pos..][0..lit_len]);
            src_pos += lit_len;
            dst_pos += lit_len;
        }

        // If we've filled the output, we're done (last sequence was literals only)
        if (dst_pos >= expected_size) break;

        // Check if we have match data (offset bytes)
        if (src_pos + 2 > input.len) break;

        // Decode match offset
        const offset = std.mem.readInt(u16, input[src_pos..][0..2], .little);
        src_pos += 2;

        if (offset == 0) break; // End marker (shouldn't happen with valid data)

        // Decode match length (token low nibble is match_len - 4)
        var match_len: usize = (token & 0x0F) + 4;
        if ((token & 0x0F) == 15) {
            while (src_pos < input.len) {
                const extra = input[src_pos];
                src_pos += 1;
                match_len += extra;
                if (extra != 255) break;
            }
        }

        // Copy match (may overlap, so copy byte by byte)
        if (dst_pos < offset or dst_pos + match_len > expected_size) {
            return error.CorruptedData;
        }

        const match_start = dst_pos - offset;
        for (0..match_len) |i| {
            output[dst_pos + i] = output[match_start + i];
        }
        dst_pos += match_len;
    }

    if (dst_pos != expected_size) {
        return error.CorruptedData;
    }

    return output;
}

/// ZSTD-style compression (simplified)
fn compressZstd(allocator: std.mem.Allocator, input: []const u8) CompressionError![]u8 {
    // For now, use the same algorithm as LZ4 but with larger window
    // A full ZSTD implementation would use Huffman coding and FSE
    return compressLz4(allocator, input);
}

fn decompressZstd(allocator: std.mem.Allocator, input: []const u8, expected_size: usize) CompressionError![]u8 {
    return decompressLz4(allocator, input, expected_size);
}

/// Run-length encoding for sparse data
fn compressRle(allocator: std.mem.Allocator, input: []const u8) CompressionError![]u8 {
    if (input.len == 0) {
        return allocator.alloc(u8, 0) catch return error.OutOfMemory;
    }

    var output = std.ArrayListUnmanaged(u8).empty;
    errdefer output.deinit(allocator);

    var i: usize = 0;
    while (i < input.len) {
        const val = input[i];
        var count: usize = 1;

        // Count consecutive identical bytes
        while (i + count < input.len and input[i + count] == val and count < 255) {
            count += 1;
        }

        if (count >= 4 or val == 0xFF) {
            // Encode as run
            try output.append(allocator, 0xFF);
            try output.append(allocator, @intCast(count));
            try output.append(allocator, val);
        } else {
            // Encode literal
            for (0..count) |_| {
                try output.append(allocator, val);
            }
        }

        i += count;
    }

    return output.toOwnedSlice(allocator) catch return error.OutOfMemory;
}

fn decompressRle(allocator: std.mem.Allocator, input: []const u8, expected_size: usize) CompressionError![]u8 {
    var output = allocator.alloc(u8, expected_size) catch return error.OutOfMemory;
    errdefer allocator.free(output);

    var src_pos: usize = 0;
    var dst_pos: usize = 0;

    while (src_pos < input.len and dst_pos < expected_size) {
        const val = input[src_pos];
        src_pos += 1;

        if (val == 0xFF and src_pos + 2 <= input.len) {
            const count = input[src_pos];
            const byte = input[src_pos + 1];
            src_pos += 2;

            if (dst_pos + count > expected_size) {
                return error.CorruptedData;
            }

            @memset(output[dst_pos..][0..count], byte);
            dst_pos += count;
        } else {
            if (dst_pos >= expected_size) {
                return error.CorruptedData;
            }
            output[dst_pos] = val;
            dst_pos += 1;
        }
    }

    if (dst_pos != expected_size) {
        return error.CorruptedData;
    }

    return output;
}

/// Estimate compressed size for a given algorithm
pub fn estimateCompressedSize(input: []const u8, comp_type: CompressionType) usize {
    return switch (comp_type) {
        .none => input.len,
        .lz4 => input.len + input.len / 255 + 16, // Worst case
        .zstd => input.len + input.len / 255 + 16,
        .rle => input.len * 3 / 2, // Worst case with escape sequences
    };
}

test "compression roundtrip none" {
    const allocator = std.testing.allocator;
    const input = "Hello, World! This is a test of the compression system.";

    const compressed = try compress(allocator, input, .none);
    defer allocator.free(compressed);

    const decompressed = try decompress(allocator, compressed, input.len, .none);
    defer allocator.free(decompressed);

    try std.testing.expectEqualStrings(input, decompressed);
}

test "compression roundtrip rle" {
    const allocator = std.testing.allocator;

    // Sparse data with runs
    var input: [256]u8 = undefined;
    @memset(input[0..100], 0);
    @memset(input[100..150], 0xAA);
    @memset(input[150..256], 0);

    const compressed = try compress(allocator, &input, .rle);
    defer allocator.free(compressed);

    const decompressed = try decompress(allocator, compressed, input.len, .rle);
    defer allocator.free(decompressed);

    try std.testing.expectEqualSlices(u8, &input, decompressed);
}

test "compression roundtrip lz4" {
    const allocator = std.testing.allocator;

    // Repetitive data that compresses well
    const pattern = "ABCD";
    var input: [256]u8 = undefined;
    for (0..64) |i| {
        @memcpy(input[i * 4 ..][0..4], pattern);
    }

    const compressed = try compress(allocator, &input, .lz4);
    defer allocator.free(compressed);

    // Should be smaller than input
    try std.testing.expect(compressed.len < input.len);

    const decompressed = try decompress(allocator, compressed, input.len, .lz4);
    defer allocator.free(decompressed);

    try std.testing.expectEqualSlices(u8, &input, decompressed);
}

test {
    std.testing.refAllDecls(@This());
}
