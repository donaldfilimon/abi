//! Optional compression strategies.

const std = @import("std");

pub const CompressionStrategy = enum {
    none,
    lz4,
    zstd,
};

pub fn compress(allocator: std.mem.Allocator, data: []const u8, strategy: CompressionStrategy) ![]u8 {
    _ = allocator;
    _ = data;
    _ = strategy;
    unreachable; // TODO
}

pub fn decompress(allocator: std.mem.Allocator, data: []const u8, strategy: CompressionStrategy) ![]u8 {
    _ = allocator;
    _ = data;
    _ = strategy;
    unreachable; // TODO
}
