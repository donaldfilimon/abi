//! Dataset Conversion Utilities
//!
//! Converts between raw formats (TokenBin, JSONL) and WDBX.

const std = @import("std");
const wdbx = @import("wdbx.zig"); // Import the newly implemented wdbx.zig

pub const ConversionError = error{
    FileNotFound,
    InvalidFormat,
    WriteError,
    ReadError,
};

// Re-export IO helpers from wdbx (which has the robust implementation)
pub const readTokenBinFile = wdbx.readTokenBinFile;
pub const writeTokenBinFile = wdbx.writeTokenBinFile;

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
    var dataset = try wdbx.WdbxTokenDataset.init(allocator, output_path);
    defer dataset.deinit();

    if (block_size > std.math.maxInt(u32)) return error.InvalidFormat;
    try dataset.importTokenBin(tokens, @intCast(block_size));
    try dataset.save();
}

/// Convert WDBX dataset to TokenBin.
pub fn wdbxToTokenBin(
    allocator: std.mem.Allocator,
    input_path: []const u8,
    output_path: []const u8,
) !void {
    var dataset = try wdbx.WdbxTokenDataset.init(allocator, input_path);
    defer dataset.deinit();

    // wdbx.collectTokens now uses self.allocator internally and expects only max_tokens
    const tokens = try dataset.collectTokens(0); // 0 = all
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
