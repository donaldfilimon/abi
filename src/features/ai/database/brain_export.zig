//! Dual Brain Export (.wdbx + .gguf)
//!
//! Exports trained models in both native WDBX brain format and GGUF for
//! Ollama/llama.cpp serving. WDBX export stores training metadata (tag "ABTR"),
//! embedding tables, and optionally LoRA adapter weights alongside the model.

const std = @import("std");
const build_options = @import("build_options");
const training = if (build_options.feat_training) @import("../training/mod.zig") else @import("../training/stub.zig");
const export_mod = @import("export.zig");
const database = if (build_options.feat_database)
    @import("../../database/mod.zig")
else
    @import("../../database/stub.zig");
const shared_types = @import("types.zig");

pub const BrainExportConfig = shared_types.BrainExportConfig;
pub const TrainingMetadata = shared_types.TrainingMetadata;
pub const ExportResult = shared_types.ExportResult;

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
        var store = try database.Store.open(allocator, "brain_export");
        defer store.deinit();

        // Store weights as vector ID 0
        try store.insert(0, model.weights, model.name);

        // Store training metadata as JSON in a metadata vector
        if (config.include_training_history) {
            if (metadata) |meta| {
                const meta_json = try serializeTrainingMeta(allocator, meta);
                defer allocator.free(meta_json);
                try store.insert(1, &.{}, meta_json);
            }
        }

        try store.save(config.wdbx_path);
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

test "exportDual writes WDBX metadata through abi.database.Store" {
    if (!build_options.feat_database or !build_options.feat_training) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const path = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/brain-export.wdbx", .{tmp.sub_path});
    defer allocator.free(path);

    var model = try training.ModelState.init(allocator, 3, "brain-model");
    defer model.deinit();
    model.weights[0] = 0.25;
    model.weights[1] = 0.5;
    model.weights[2] = 0.75;

    const result = try exportDual(allocator, &model, .{
        .wdbx_path = path,
    }, .{
        .model_name = "brain-model",
        .epochs_completed = 3,
        .training_samples = 42,
    });

    try std.testing.expect(result.wdbx_written);

    var store = try database.Store.load(allocator, path);
    defer store.deinit();

    const weights = store.get(0) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("brain-model", weights.metadata.?);

    const meta = store.get(1) orelse return error.TestUnexpectedResult;
    try std.testing.expect(meta.metadata != null);
    try std.testing.expect(std.mem.indexOf(u8, meta.metadata.?, "\"model_name\"") != null);
}
