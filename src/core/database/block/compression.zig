//! Optional compression strategies.

const std = @import("std");

pub const CompressionStrategy = enum {
    none,
    lz4,
    zstd,
};

pub fn compress(allocator: std.mem.Allocator, data: []const u8, strategy: CompressionStrategy) ![]u8 {
    return switch (strategy) {
        .none => allocator.dupe(u8, data),
        .lz4 => error.NotImplemented,
        .zstd => error.NotImplemented,
    };
}

pub fn decompress(allocator: std.mem.Allocator, data: []const u8, strategy: CompressionStrategy) ![]u8 {
    return switch (strategy) {
        .none => allocator.dupe(u8, data),
        .lz4 => error.NotImplemented,
        .zstd => error.NotImplemented,
    };
}
