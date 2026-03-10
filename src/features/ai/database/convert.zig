//! Dataset Conversion Utilities
//!
//! Converts between raw formats (TokenBin, JSONL) and WDBX.

const std = @import("std");
const dataset = @import("wdbx.zig");

pub const ConversionError = error{
    FileNotFound,
    InvalidFormat,
    WriteError,
    ReadError,
};

// Re-export IO helpers from the dataset implementation.
pub const readTokenBinFile = dataset.readTokenBinFile;
pub const writeTokenBinFile = dataset.writeTokenBinFile;

/// Convert a raw TokenBin file (u32 array) to WDBX.
pub fn tokenBinToWdbx(
    allocator: std.mem.Allocator,
    input_path: []const u8,
    output_path: []const u8,
    block_size: usize,
) !void {
    const tokens = try readTokenBinFile(allocator, input_path);
    defer allocator.free(tokens);

    // Create WDBX
    var token_dataset = try dataset.WdbxTokenDataset.init(allocator, output_path);
    defer token_dataset.deinit();

    if (block_size > std.math.maxInt(u32)) return error.InvalidFormat;
    try token_dataset.importTokenBin(tokens, @intCast(block_size));
    try token_dataset.save();
}

/// Convert WDBX dataset to TokenBin.
pub fn wdbxToTokenBin(
    allocator: std.mem.Allocator,
    input_path: []const u8,
    output_path: []const u8,
) !void {
    var token_dataset = try dataset.WdbxTokenDataset.init(allocator, input_path);
    defer token_dataset.deinit();

    // The dataset collector uses its internal allocator and expects only max_tokens.
    const tokens = try token_dataset.collectTokens(0); // 0 = all
    defer allocator.free(tokens);

    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    const file = std.Io.Dir.cwd().createFile(io, output_path, .{}) catch return error.WriteError;
    defer file.close(io);

    const bytes = std.mem.sliceAsBytes(tokens);
    try file.writeStreamingAll(io, bytes);
}

test {
    std.testing.refAllDecls(@This());
}
