//! Model Export Utilities
//!
//! Exports trained models to GGUF format.

const std = @import("std");
const db = @import("../../database/mod.zig");
const training = @import("../training/mod.zig");

pub fn exportGguf(
    allocator: std.mem.Allocator,
    model: *const training.TrainableModel,
    path: []const u8,
) !void {
    // 1. Create Unified Format
    var builder = db.UnifiedFormatBuilder.init(allocator);
    defer builder.deinit();

    // Add weights
    for (model.weights.items) |layer| {
        // Assume layer.name is compatible
        const shape = [4]u64{ layer.shape[0], layer.shape[1], layer.shape[2], layer.shape[3] };
        const data = std.mem.sliceAsBytes(layer.data);

        // Map types (f32 -> f32)
        try builder.addTensor(layer.name, data, .f32, &shape);
    }

    // Add metadata
    try builder.addMetadata("general.architecture", "llama");
    try builder.addMetadata("general.name", model.config.name);

    // Build unified format
    const unified_bytes = try builder.build();
    defer allocator.free(unified_bytes);

    var format = try db.UnifiedFormat.fromMemory(allocator, unified_bytes);
    defer format.deinit();

    // 2. Convert to GGUF
    const gguf_bytes = try db.toGguf(allocator, &format);
    defer allocator.free(gguf_bytes);

    // 3. Write to disk
    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();
    var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    defer file.close(io);
    var writer = file.writer(io);
    try writer.writeAll(gguf_bytes);
}
