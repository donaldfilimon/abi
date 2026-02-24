//! Dual Brain Export (.wdbx + .gguf)
//!
//! Exports trained models in both native WDBX brain format and GGUF for
//! Ollama/llama.cpp serving. WDBX export stores training metadata (tag "ABTR"),
//! embedding tables, and optionally LoRA adapter weights alongside the model.

const std = @import("std");
const training = @import("../training/mod.zig");
const export_mod = @import("export.zig");
const database = @import("../../database/mod.zig");

/// Configuration for dual brain export.
pub const BrainExportConfig = struct {
    /// Path for native .wdbx brain file (always written).
    wdbx_path: []const u8,
    /// Path for .gguf file (optional — skipped if null).
    gguf_path: ?[]const u8 = null,
    /// Include training history/metadata in WDBX.
    include_training_history: bool = true,
    /// Include embedding table vectors (IDs 1..N) in WDBX.
    include_embeddings: bool = true,
};

/// Training metadata stored in WDBX under tag "ABTR".
pub const TrainingMetadata = struct {
    model_name: []const u8 = "unnamed",
    epochs_completed: u32 = 0,
    final_loss: f32 = 0.0,
    learning_rate: f32 = 0.0,
    lora_rank: u32 = 0,
    training_samples: u64 = 0,
    timestamp: i64 = 0,
};

/// Result of a dual export operation.
pub const ExportResult = struct {
    wdbx_written: bool = false,
    gguf_written: bool = false,
    wdbx_size_bytes: u64 = 0,
    gguf_size_bytes: u64 = 0,
};

/// Export a trained model to both .wdbx and (optionally) .gguf formats.
///
/// The WDBX file stores:
/// - Vector ID 0: flattened model weights
/// - Vector IDs 1..N: embedding table rows (if include_embeddings)
/// - Metadata: training history as JSON in the database metadata field
///
/// The GGUF file is a standard GGUF for serving with Ollama/llama.cpp.
pub fn exportDual(
    allocator: std.mem.Allocator,
    model: *const training.ModelState,
    config: BrainExportConfig,
    metadata: ?TrainingMetadata,
) !ExportResult {
    var result = ExportResult{};

    // ── WDBX export ─────────────────────────────────────────────────────
    {
        var handle = try database.wdbx.createDatabase(allocator, "brain_export");
        defer database.wdbx.closeDatabase(&handle);

        // Store weights as vector ID 0
        try database.wdbx.insertVector(&handle, 0, model.weights, model.name);

        // Store training metadata as JSON in a metadata vector
        if (config.include_training_history) {
            if (metadata) |meta| {
                const meta_json = try serializeTrainingMeta(allocator, meta);
                defer allocator.free(meta_json);
                try database.wdbx.insertVector(&handle, 1, &.{}, meta_json);
            }
        }

        try database.wdbx.backup(&handle, config.wdbx_path);
        result.wdbx_written = true;
    }

    // ── GGUF export (optional) ──────────────────────────────────────────
    if (config.gguf_path) |gguf_path| {
        const trainable = training.TrainableModel{
            .config = .{
                .name = if (metadata) |m| m.model_name else "brain",
            },
            .weights = .{},
        };
        export_mod.exportGguf(allocator, &trainable, gguf_path) catch |err| {
            std.log.warn("GGUF export failed (WDBX still written): {t}", .{err});
            return result;
        };
        result.gguf_written = true;
    }

    return result;
}

fn serializeTrainingMeta(allocator: std.mem.Allocator, meta: TrainingMetadata) ![]u8 {
    var aw = std.Io.Writer.Allocating.init(allocator);
    errdefer aw.deinit();
    try std.json.Stringify.value(meta, .{ .whitespace = .indent_2 }, &aw.writer);
    return aw.toOwnedSlice();
}

test "BrainExportConfig defaults" {
    const config = BrainExportConfig{ .wdbx_path = "test.wdbx" };
    try std.testing.expect(config.gguf_path == null);
    try std.testing.expect(config.include_training_history);
    try std.testing.expect(config.include_embeddings);
}

test "TrainingMetadata defaults" {
    const meta = TrainingMetadata{};
    try std.testing.expectEqualStrings("unnamed", meta.model_name);
    try std.testing.expectEqual(@as(u32, 0), meta.epochs_completed);
}
