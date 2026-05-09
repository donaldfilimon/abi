//! Model Export Utilities
//!
//! Exports trained models to GGUF format. Works with ModelState (flat weight
//! vectors) rather than TrainableModel (which has structured layer weights
//! requiring a different serialization path).

const std = @import("std");
const build_options = @import("build_options");
const db = if (build_options.feat_database)
    @import("../../database/mod.zig")
else
    @import("../../database/stub.zig");
const training = if (build_options.feat_training) @import("../training/mod.zig") else @import("../training/stub.zig");

/// Export a ModelState's flat weight vector to GGUF format.
///
/// This is the primary export path used by brain_export.zig. It takes raw
/// weights from ModelState rather than structured TrainableModel layers.
pub fn exportGgufFromState(
    allocator: std.mem.Allocator,
    model: *const training.ModelState,
    model_name: []const u8,
    path: []const u8,
) !void {
    // 1. Create Unified Format
    var builder = db.retrieval.formats.UnifiedFormatBuilder.init(allocator);
    defer builder.deinit();

    // Add weights as a single flattened tensor
    const weight_data = std.mem.sliceAsBytes(model.weights);
    const shape = [4]u64{ @intCast(model.weights.len), 1, 1, 1 };
    _ = try builder.addTensor("model.weights", weight_data, .f32, &shape);

    // Add metadata
    _ = try builder.addMetadata("general.architecture", "llama");
    _ = try builder.addMetadata("general.name", model_name);

    // Build unified format
    const unified_bytes = try builder.build();
    defer allocator.free(unified_bytes);

    var format = try db.retrieval.formats.UnifiedFormat.fromMemory(allocator, unified_bytes);
    defer format.deinit();

    // 2. Convert to GGUF
    const gguf_bytes = try db.retrieval.formats.toGguf(allocator, &format);
    defer allocator.free(gguf_bytes);

    // 3. Write to disk using std.Io backend
    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();
    var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    defer file.close(io);
    try file.writeStreamingAll(io, gguf_bytes);
}

test {
    std.testing.refAllDecls(@This());
}
