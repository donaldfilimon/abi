//! End-to-end pipeline for the `nn` character-level trainer: write a JSONL
//! corpus to disk, load + extract + train from it (`trainOnJsonl`), separately
//! train a `Model`, persist it to a checkpoint *file* (`saveModelPath`, not
//! just in-memory bytes), reload it (`loadModelPath`), and greedily generate
//! (sample/decode) from both the original and reloaded model to confirm the
//! whole file-backed chain is deterministic end to end.
//!
//! This is a small, from-scratch, pure-Zig demo trainer (see
//! `src/features/nn/types.zig`) — not a production or distributed LLM.

const std = @import("std");
const nn = @import("../features/nn/mod.zig");
const train_mod = @import("../features/nn/train.zig");
const persist = @import("../features/nn/persist.zig");

const testing = std.testing;

fn deleteIfExists(path: []const u8) void {
    std.Io.Dir.cwd().deleteFile(testing.io, path) catch |err| switch (err) {
        error.FileNotFound => {},
        else => std.debug.print("e2e llm cleanup: failed to delete '{s}': {s}\n", .{ path, @errorName(err) }),
    };
}

/// `zig-out` only exists after an install step has run, so a bare
/// `zig build test-integration` on a clean tree would otherwise fail with
/// FileNotFound before reaching a single assertion.
fn ensureOutDir() !void {
    std.Io.Dir.createDirPath(.cwd(), testing.io, "zig-out") catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };
}

test "e2e: llm pipeline (jsonl load -> train -> save -> load -> generate -> decode)" {
    const a = testing.allocator;
    try ensureOutDir();

    const corpus_path = "zig-out/e2e-llm-corpus.jsonl";
    const checkpoint_path = "zig-out/e2e-llm-checkpoint.bin";
    defer {
        deleteIfExists(corpus_path);
        deleteIfExists(checkpoint_path);
    }
    deleteIfExists(corpus_path);
    deleteIfExists(checkpoint_path);

    // --- Phase 1: "load" — write a JSONL corpus to disk and extract+train
    // from the file, the same shape as `abi nn train --jsonl`. ---
    const jsonl =
        \\{"text": "hello world hello world "}
        \\{"text": ""}
        \\{"other": "ignored"}
        \\{"text": "hello world hello world "}
        \\
    ;
    {
        var dir = std.Io.Dir.cwd();
        try dir.writeFile(testing.io, .{ .sub_path = corpus_path, .data = jsonl });
    }

    const config = nn.TrainConfig{
        .seq_len = 2,
        .hidden = 8,
        .embed_dim = 4,
        .epochs = 60,
        .lr = 0.5,
        .seed = 7,
    };

    const report = try train_mod.trainOnJsonl(a, corpus_path, "text", config);
    try testing.expect(report.improved);
    try testing.expect(report.final_loss < report.initial_loss);

    // --- Phase 2: train a model directly (same corpus content) so we hold
    // the live `Model` needed to save/sample. ---
    const corpus_bytes =
        \\hello world hello world
        \\hello world hello world
    ;
    var model = try train_mod.trainModel(a, corpus_bytes, config);
    defer model.deinit();
    try testing.expect(model.report.improved);

    // --- Phase 3: "save" the trained model to a checkpoint *file*. ---
    try persist.saveModelPath(testing.io, a, &model, checkpoint_path);

    // --- Phase 4: "load" the checkpoint back from disk into a fresh Model. ---
    var loaded = try persist.loadModelPath(testing.io, a, checkpoint_path);
    defer loaded.deinit();

    try testing.expectEqual(model.vocab_size, loaded.vocab_size);
    try testing.expectEqual(model.seq_len, loaded.seq_len);
    try testing.expectEqualSlices(f32, model.w1.data, loaded.w1.data);

    // --- Phase 5: "generate -> decode" — greedy-sample bytes from both the
    // original and the reloaded model; the checkpoint round trip must not
    // change generation output. ---
    const original_output = try nn.sample(a, &model, 'h', 24);
    defer a.free(original_output);
    const reloaded_output = try nn.sample(a, &loaded, 'h', 24);
    defer a.free(reloaded_output);

    try testing.expectEqualStrings(original_output, reloaded_output);
    // Decoded bytes must be printable corpus characters, not garbage — the
    // "decode" step of the pipeline.
    for (original_output) |byte| {
        try testing.expect(std.ascii.isPrint(byte));
    }
}

test {
    std.testing.refAllDecls(@This());
}
