//! Optional compression strategies.
//!
//! The `.lz4` variant uses a simple RLE (run-length encoding) scheme:
//!   - Runs of 3+ identical bytes are encoded as [0xFF, count, value].
//!   - Literal 0xFF bytes are escaped as [0xFF, 0x01, 0xFF].
//!   - All other bytes are emitted verbatim.

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

/// Marker byte used by the RLE scheme.
const RLE_MARKER: u8 = 0xFF;

pub fn compress(allocator: std.mem.Allocator, data: []const u8, strategy: CompressionStrategy) CompressionError![]u8 {
    return switch (strategy) {
        .none => allocator.dupe(u8, data) catch return error.OutOfMemory,
        .lz4 => rleCompress(allocator, data),
        .zstd => error.NotImplemented,
    };
}

pub fn decompress(allocator: std.mem.Allocator, data: []const u8, strategy: CompressionStrategy) CompressionError![]u8 {
    return switch (strategy) {
        .none => allocator.dupe(u8, data) catch return error.OutOfMemory,
        .lz4 => rleDecompress(allocator, data),
        .zstd => error.NotImplemented,
    };
}

// ============================================================================
// RLE implementation
// ============================================================================

fn rleCompress(allocator: std.mem.Allocator, data: []const u8) CompressionError![]u8 {
    if (data.len == 0) {
        return allocator.alloc(u8, 0) catch return error.OutOfMemory;
    }

    // Worst case: every byte is 0xFF and needs escaping (3x expansion).
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
            // Escape every occurrence of the marker byte, regardless of run length.
            // Each marker byte -> [0xFF, 0x01, 0xFF].  Runs of markers are still
            // encoded as a single RLE triple when >= 3.
            if (run_len >= 3) {
                out.append(allocator, RLE_MARKER) catch return error.OutOfMemory;
                out.append(allocator, @intCast(run_len)) catch return error.OutOfMemory;
                out.append(allocator, RLE_MARKER) catch return error.OutOfMemory;
            } else {
                // Escape each marker individually.
                for (0..run_len) |_| {
                    out.append(allocator, RLE_MARKER) catch return error.OutOfMemory;
                    out.append(allocator, 0x01) catch return error.OutOfMemory;
                    out.append(allocator, RLE_MARKER) catch return error.OutOfMemory;
                }
            }
            i += run_len;
        } else if (run_len >= 3) {
            // Encode a non-marker run.
            out.append(allocator, RLE_MARKER) catch return error.OutOfMemory;
            out.append(allocator, @intCast(run_len)) catch return error.OutOfMemory;
            out.append(allocator, byte) catch return error.OutOfMemory;
            i += run_len;
        } else {
            // Emit literal bytes.
            for (0..run_len) |_| {
                out.append(allocator, byte) catch return error.OutOfMemory;
            }
            i += run_len;
        }
    }

    return out.toOwnedSlice(allocator) catch return error.OutOfMemory;
}

fn rleDecompress(allocator: std.mem.Allocator, data: []const u8) CompressionError![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    var i: usize = 0;
    while (i < data.len) {
        if (data[i] == RLE_MARKER) {
            // Must have at least 2 more bytes: count + value.
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

test "RLE round-trip" {
    const allocator = std.testing.allocator;

    // Plain data with runs.
    const input = "AAABBBCCDDDDDD";
    const compressed = try compress(allocator, input, .lz4);
    defer allocator.free(compressed);

    const decompressed = try decompress(allocator, compressed, .lz4);
    defer allocator.free(decompressed);

    try std.testing.expectEqualSlices(u8, input, decompressed);
}

test "RLE marker escape round-trip" {
    const allocator = std.testing.allocator;

    // Data containing literal 0xFF bytes.
    const input = &[_]u8{ 0x01, 0xFF, 0x02, 0xFF, 0xFF, 0xFF, 0xFF, 0x03 };
    const compressed = try compress(allocator, input, .lz4);
    defer allocator.free(compressed);

    const decompressed = try decompress(allocator, compressed, .lz4);
    defer allocator.free(decompressed);

    try std.testing.expectEqualSlices(u8, input, decompressed);
}

test "RLE empty data" {
    const allocator = std.testing.allocator;

    const compressed = try compress(allocator, &[_]u8{}, .lz4);
    defer allocator.free(compressed);

    const decompressed = try decompress(allocator, compressed, .lz4);
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
