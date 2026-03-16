//! Abbey Fine-Tuning Orchestrator
//!
//! End-to-end pipeline: reads lilex-generated Alpaca JSONL, tokenizes with
//! the base model's BPE tokenizer, fine-tunes via LoRA, merges adapters into
//! base weights, and exports the result as a GGUF file for Ollama serving.
//!
//! Usage (from abi CLI):
//!   abi abbey-train --base-gguf gemma-3-4b-it.gguf \
//!                   --jsonl out/abbey_train.jsonl   \
//!                   --output abbey_brain.gguf        \
//!                   --epochs 3 --batch-size 4

const std = @import("std");
const training = @import("../training/mod.zig");
const lora_mod = @import("../training/lora.zig");
const bpe = @import("../llm/tokenizer/bpe.zig");
const gguf_reader = @import("../llm/io/gguf.zig");
const brain_export = @import("../database/brain_export.zig");

/// Configuration for Abbey fine-tuning.
pub const AbbyTrainConfig = struct {
    /// Path to base model GGUF (e.g. gemma-3-4b-it.gguf)
    base_gguf_path: []const u8,
    /// Path to lilex-generated Alpaca JSONL
    jsonl_path: []const u8,
    /// Output GGUF path
    output_path: []const u8 = "abbey_brain.gguf",
    /// Training epochs
    epochs: u32 = 3,
    /// Batch size (sequences per batch)
    batch_size: u32 = 4,
    /// Max sequence length
    max_seq_len: u32 = 512,
    /// Learning rate
    learning_rate: f32 = 2e-5,
    /// Gradient accumulation steps (effective batch = batch_size * grad_accum)
    grad_accum_steps: u32 = 8,
    /// LoRA rank
    lora_rank: u32 = 16,
    /// LoRA alpha
    lora_alpha: f32 = 32.0,
    /// LoRA dropout
    lora_dropout: f32 = 0.05,
    /// Save LoRA adapter weights separately
    save_adapter_path: ?[]const u8 = null,
    /// Checkpoint directory
    checkpoint_path: ?[]const u8 = null,
    /// Output path for native .wdbx brain file (dual export)
    wdbx_output_path: ?[]const u8 = null,
};

/// Format an instruction sample into the Alpaca prompt template for tokenization.
fn formatInstructionPrompt(
    allocator: std.mem.Allocator,
    sample: training.InstructionSample,
) ![]u8 {
    var buf = std.ArrayListUnmanaged(u8).empty;
    errdefer buf.deinit(allocator);

    const writer = buf.writer(allocator);

    try writer.writeAll("### Instruction:\n");
    try writer.writeAll(sample.instruction);
    try writer.writeAll("\n");

    if (sample.input) |input| {
        if (input.len > 0) {
            try writer.writeAll("### Input:\n");
            try writer.writeAll(input);
            try writer.writeAll("\n");
        }
    }

    try writer.writeAll("### Response:\n");
    try writer.writeAll(sample.output);
    try writer.writeAll("\n");

    return buf.toOwnedSlice(allocator);
}

/// Run the full Abbey fine-tuning pipeline.
pub fn run(allocator: std.mem.Allocator, config: AbbyTrainConfig) !void {
    // ── Step 1: Load base model from GGUF ──────────────────────────────
    std.log.info("Loading base model from: {s}", .{config.base_gguf_path});
    var model = try training.TrainableModel.fromGguf(allocator, config.base_gguf_path);
    defer model.deinit();

    const model_cfg = model.config;
    std.log.info("Model: {d} layers, hidden_dim={d}, vocab={d}", .{
        model_cfg.num_layers,
        model_cfg.hidden_dim,
        model_cfg.vocab_size,
    });

    // ── Step 2: Initialize I/O backend ─────────────────────────────────
    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    // ── Step 3: Load BPE tokenizer from the same GGUF ──────────────────
    std.log.info("Loading tokenizer from GGUF metadata...", .{});
    var tokenizer = try loadTokenizerFromGguf(allocator, io, config.base_gguf_path);
    defer tokenizer.deinit();

    // ── Step 4: Read and parse JSONL ────────────────────────────────────
    std.log.info("Reading JSONL from: {s}", .{config.jsonl_path});

    const jsonl_data = std.Io.Dir.cwd().readFileAlloc(
        io,
        config.jsonl_path,
        allocator,
        .limited(256 * 1024 * 1024), // 256 MB max
    ) catch |err| {
        std.log.err("Failed to read JSONL: {t}", .{err});
        return err;
    };
    defer allocator.free(jsonl_data);

    var samples = try training.parseInstructionDataset(allocator, jsonl_data);
    defer samples.deinit(allocator);

    std.log.info("Parsed {d} instruction samples", .{samples.items.len});
    if (samples.items.len == 0) {
        std.log.err("No samples found in JSONL. Nothing to train on.", .{});
        return error.NoTrainingData;
    }

    // ── Step 4: Tokenize all samples into a flat token array ───────────
    std.log.info("Tokenizing samples...", .{});
    var all_tokens = std.ArrayListUnmanaged(u32).empty;
    defer all_tokens.deinit(allocator);

    for (samples.items) |sample| {
        const prompt = try formatInstructionPrompt(allocator, sample);
        defer allocator.free(prompt);

        const tokens = try tokenizer.encode(allocator, prompt);
        defer allocator.free(tokens);

        // Truncate to max_seq_len per sample
        const seq_len = @min(tokens.len, config.max_seq_len);
        try all_tokens.appendSlice(allocator, tokens[0..seq_len]);
    }

    std.log.info("Total tokens: {d}", .{all_tokens.items.len});

    // ── Step 5: Initialize LoRA adapters ────────────────────────────────
    std.log.info("Initializing LoRA (rank={d}, alpha={d:.1})...", .{
        config.lora_rank,
        config.lora_alpha,
    });
    var lora_model = try lora_mod.LoraModel.init(
        allocator,
        model_cfg.num_layers,
        model_cfg.hidden_dim,
        model_cfg.num_heads,
        model_cfg.num_kv_heads,
        model_cfg.intermediate_dim,
        .{
            .rank = config.lora_rank,
            .alpha = config.lora_alpha,
            .dropout = config.lora_dropout,
            .target_modules = .{
                .q_proj = true,
                .k_proj = true,
                .v_proj = true,
            },
        },
    );
    defer lora_model.deinit();

    std.log.info("LoRA trainable params: {d}", .{lora_model.numParams()});

    // ── Step 6: Train ───────────────────────────────────────────────────
    const train_config = training.LlmTrainingConfig{
        .epochs = config.epochs,
        .batch_size = config.batch_size,
        .max_seq_len = config.max_seq_len,
        .learning_rate = config.learning_rate,
        .lr_schedule = .warmup_cosine,
        .warmup_steps = 100,
        .grad_accum_steps = config.grad_accum_steps,
        .max_grad_norm = 1.0,
        .weight_decay = 0.01,
        .optimizer = .adamw,
        .log_interval = 10,
        .checkpoint_interval = if (config.checkpoint_path != null) 500 else 0,
        .checkpoint_path = config.checkpoint_path,
        .export_gguf_path = null, // We handle GGUF export manually after merge
        .export_name = "abbey",
    };

    try train_config.validate();

    var trainer = try training.LlamaTrainer.init(allocator, &model, train_config);
    defer trainer.deinit();

    try model.prepareForTraining(config.max_seq_len);

    std.log.info("Starting training: {d} epochs, batch_size={d}, grad_accum={d}", .{
        config.epochs,
        config.batch_size,
        config.grad_accum_steps,
    });

    for (0..config.epochs) |epoch| {
        const loss = try trainer.trainEpoch(all_tokens.items, samples.items.len);
        std.log.info("Epoch {d}/{d} — loss: {d:.4}", .{
            epoch + 1,
            config.epochs,
            loss,
        });
    }

    // ── Step 7: Merge LoRA weights into base model ─────────────────────
    std.log.info("Merging LoRA adapters into base weights...", .{});
    lora_model.mergeWeights(&model.weights);

    // Optionally save LoRA adapter weights separately
    if (config.save_adapter_path) |adapter_path| {
        std.log.info("Saving LoRA adapter to: {s}", .{adapter_path});
        try lora_model.save(allocator, adapter_path);
    }

    // ── Step 8: Export merged model as GGUF ─────────────────────────────
    std.log.info("Exporting GGUF to: {s}", .{config.output_path});
    const gguf_export_config = @import("../training/trainable_model/checkpoint.zig").GgufExportConfig{
        .name = "abbey",
    };
    try model.exportToGguf(allocator, config.output_path, gguf_export_config);

    // ── Step 9: Dual brain export (.wdbx + .gguf) ────────────────────────
    if (config.wdbx_output_path) |wdbx_path| {
        std.log.info("Exporting WDBX brain to: {s}", .{wdbx_path});

        var ts: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.REALTIME, &ts);

        const brain_cfg = brain_export.BrainExportConfig{
            .wdbx_path = wdbx_path,
            .gguf_path = null, // GGUF already exported above
            .include_training_history = true,
            .include_embeddings = true,
        };
        const meta = brain_export.TrainingMetadata{
            .model_name = "abbey",
            .epochs_completed = config.epochs,
            .learning_rate = config.learning_rate,
            .lora_rank = config.lora_rank,
            .training_samples = samples.items.len,
            .timestamp = @intCast(ts.sec),
        };

        if (brain_export.exportDual(allocator, &model.state, brain_cfg, meta)) |_| {
            std.log.info("WDBX brain exported to: {s}", .{wdbx_path});
        } else |err| {
            std.log.warn("WDBX brain export failed: {t}", .{err});
        }
    }

    std.log.info("Done! Model exported to: {s}", .{config.output_path});
    if (config.wdbx_output_path) |wdbx_path| {
        std.log.info("  WDBX brain: {s}", .{wdbx_path});
    }
    std.log.info("Next steps:", .{});
    std.log.info("  1. Create Modelfile: FROM ./{s}", .{config.output_path});
    std.log.info("  2. ollama create abbey -f Modelfile", .{});
    std.log.info("  3. ollama run abbey \"Hello!\"", .{});
}

/// Load a BPE tokenizer from GGUF metadata (vocab + merges).
fn loadTokenizerFromGguf(
    allocator: std.mem.Allocator,
    io: std.Io,
    gguf_path: []const u8,
) !bpe.BpeTokenizer {
    var gguf = try gguf_reader.GgufFile.open(allocator, io, gguf_path);
    defer gguf.deinit();

    var tokenizer = bpe.BpeTokenizer.init(allocator);
    errdefer tokenizer.deinit();

    // Load vocab tokens
    if (gguf.getTokensArray()) |tokens_data| {
        try tokenizer.loadVocab(tokens_data);
    }

    // Load merges
    if (gguf.getMergesArray()) |merges_data| {
        try tokenizer.loadMergesFromGguf(merges_data.data, merges_data.count);
    }

    // Load special token IDs
    if (gguf.getMetadataValue("tokenizer.ggml.bos_token_id")) |bos| {
        tokenizer.special.bos_id = @intCast(bos.uint32);
    }
    if (gguf.getMetadataValue("tokenizer.ggml.eos_token_id")) |eos| {
        tokenizer.special.eos_id = @intCast(eos.uint32);
    }

    return tokenizer;
}

/// Error set for abbey training.
pub const AbbyTrainError = error{
    NoTrainingData,
    InvalidModel,
    TokenizerLoadFailed,
};
