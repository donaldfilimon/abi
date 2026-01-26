//! Training pipeline CLI command.
//!
//! Commands:
//! - train run [options] - Run training pipeline
//! - train llm <model.gguf> [options] - Train LLM model
//! - train resume <checkpoint> - Resume training from checkpoint
//! - train monitor [run-id] - Monitor training progress (TUI dashboard)
//! - train info - Show default training configuration
//! - train help - Show help message

const std = @import("std");
const abi = @import("abi");
const utils = @import("../utils/mod.zig");
const tui = @import("../tui/mod.zig");

const train_subcommands = [_][]const u8{
    "run",
    "new",
    "llm",
    "vision",
    "clip",
    "resume",
    "monitor",
    "info",
    "help",
};

/// Run the train command with the provided arguments.
pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (args.len == 0 or utils.args.matchesAny(args[0], &[_][]const u8{ "help", "--help", "-h" })) {
        printHelp();
        return;
    }

    const command = std.mem.sliceTo(args[0], 0);

    if (std.mem.eql(u8, command, "run")) {
        try runTrain(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "new")) {
        try runNewModel(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "llm")) {
        try runLlmTrain(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "vision")) {
        try runVisionTrain(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "clip")) {
        try runClipTrain(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "resume")) {
        try runResume(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "monitor")) {
        try runMonitor(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "info")) {
        runInfo();
        return;
    }

    std.debug.print("Unknown train command: {s}\n", .{command});
    if (utils.args.suggestCommand(command, &train_subcommands)) |suggestion| {
        std.debug.print("Did you mean: {s}\n", .{suggestion});
    }
    printHelp();
}

fn runTrain(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (utils.args.containsHelpArgs(args)) {
        printHelp();
        return;
    }

    var config = abi.ai.TrainingConfig{};

    var i: usize = 0;
    while (i < args.len) {
        const arg = args[i];
        i += 1;

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--epochs", "-e" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.epochs = std.fmt.parseInt(u32, val, 10) catch 10;
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--batch-size", "-b" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.batch_size = std.fmt.parseInt(u32, val, 10) catch 32;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--model-size")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.model_size = std.fmt.parseInt(u32, val, 10) catch 512;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--sample-count")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.sample_count = std.fmt.parseInt(usize, val, 10) catch 1024;
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--learning-rate", "--lr" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.learning_rate = std.fmt.parseFloat(f32, val) catch 0.001;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--optimizer")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.optimizer = parseOptimizer(val);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--lr-schedule")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.learning_rate_schedule = parseLrSchedule(val);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--warmup-steps")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.warmup_steps = std.fmt.parseInt(u32, val, 10) catch 100;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--decay-steps")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.decay_steps = std.fmt.parseInt(u32, val, 10) catch 1000;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--weight-decay")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.weight_decay = std.fmt.parseFloat(f32, val) catch 0.01;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--gradient-clip")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.gradient_clip_norm = std.fmt.parseFloat(f32, val) catch 1.0;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--gradient-accumulation")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.gradient_accumulation_steps = std.fmt.parseInt(u32, val, 10) catch 1;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--checkpoint-interval")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.checkpoint_interval = std.fmt.parseInt(u32, val, 10) catch 0;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--checkpoint-path")) {
            if (i < args.len) {
                config.checkpoint_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--max-checkpoints")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.max_checkpoints = std.fmt.parseInt(u32, val, 10) catch 5;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--early-stopping-patience")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.early_stopping_patience = std.fmt.parseInt(u32, val, 10) catch 5;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--mixed-precision")) {
            config.mixed_precision = true;
            continue;
        }
    }

    // Validate configuration
    config.validate() catch |err| {
        std.debug.print("Invalid configuration: {t}\n", .{err});
        return;
    };

    // Print configuration summary
    std.debug.print("Training Configuration\n", .{});
    std.debug.print("======================\n", .{});
    std.debug.print("Epochs:           {d}\n", .{config.epochs});
    std.debug.print("Batch size:       {d}\n", .{config.batch_size});
    std.debug.print("Sample count:     {d}\n", .{config.sample_count});
    std.debug.print("Model size:       {d}\n", .{config.model_size});
    std.debug.print("Learning rate:    {d:.6}\n", .{config.learning_rate});
    std.debug.print("Optimizer:        {t}\n", .{config.optimizer});
    std.debug.print("LR schedule:      {t}\n", .{config.learning_rate_schedule});
    std.debug.print("Warmup steps:     {d}\n", .{config.warmup_steps});
    std.debug.print("Decay steps:      {d}\n", .{config.decay_steps});
    std.debug.print("Weight decay:     {d:.6}\n", .{config.weight_decay});
    std.debug.print("Gradient clip:    {d:.2}\n", .{config.gradient_clip_norm});
    std.debug.print("Grad accumulation:{d}\n", .{config.gradient_accumulation_steps});
    if (config.checkpoint_interval > 0) {
        std.debug.print("Checkpoint interval: {d}\n", .{config.checkpoint_interval});
    }
    if (config.checkpoint_path) |path| {
        std.debug.print("Checkpoint path:  {s}\n", .{path});
    }
    std.debug.print("Mixed precision:  {}\n", .{config.mixed_precision});
    std.debug.print("\n", .{});

    // Run training
    std.debug.print("Starting training...\n\n", .{});

    var timer = std.time.Timer.start() catch {
        std.debug.print("Error: Failed to start timer\n", .{});
        return;
    };

    var result = abi.ai.trainWithResult(allocator, config) catch |err| {
        std.debug.print("Training failed: {t}\n", .{err});
        return;
    };
    defer result.deinit();

    const elapsed_ns = timer.read();
    const elapsed_ms = elapsed_ns / std.time.ns_per_ms;

    // Print results
    std.debug.print("Training Complete\n", .{});
    std.debug.print("=================\n", .{});
    std.debug.print("Epochs completed: {d}\n", .{result.report.epochs});
    std.debug.print("Batches/epoch:    {d}\n", .{result.report.batches});
    std.debug.print("Gradient updates: {d}\n", .{result.report.gradient_updates});
    std.debug.print("Final loss:       {d:.6}\n", .{result.report.final_loss});
    std.debug.print("Final accuracy:   {d:.2}%\n", .{result.report.final_accuracy * 100});
    std.debug.print("Best loss:        {d:.6}\n", .{result.report.best_loss});
    std.debug.print("Final LR:         {d:.8}\n", .{result.report.learning_rate});
    std.debug.print("Checkpoints saved:{d}\n", .{result.report.checkpoints_saved});
    if (result.report.early_stopped) {
        std.debug.print("Early stopped:    yes\n", .{});
    }
    std.debug.print("Total time:       {d}ms\n", .{elapsed_ms});

    // Print loss history (abbreviated)
    if (result.loss_history.len > 0) {
        std.debug.print("\nLoss history (sampled):\n", .{});
        const step_size = @max(1, result.loss_history.len / 5);
        var idx: usize = 0;
        while (idx < result.loss_history.len) : (idx += step_size) {
            std.debug.print("  Epoch {d}: loss={d:.6}, acc={d:.2}%\n", .{
                idx + 1,
                result.loss_history[idx],
                result.accuracy_history[idx] * 100,
            });
        }
    }
}

fn runNewModel(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (utils.args.containsHelpArgs(args)) {
        printNewHelp();
        return;
    }

    // Check if LLM feature is enabled
    if (!abi.ai.llm.isEnabled()) {
        std.debug.print("Error: LLM feature is not enabled. Build with -Denable-llm=true\n", .{});
        return;
    }

    // Default model configuration (tiny model for training from scratch)
    var hidden_dim: u32 = 256;
    var num_layers: u32 = 4;
    var num_heads: u32 = 4;
    var intermediate_dim: u32 = 512;
    var vocab_size: u32 = 32000;
    var max_seq_len: u32 = 128;

    // Training config
    var epochs: u32 = 1;
    var batch_size: u32 = 4;
    var learning_rate: f32 = 1e-4;
    var warmup_steps: u32 = 100;
    var weight_decay: f32 = 0.01;
    var gradient_clip: f32 = 1.0;
    var grad_accum_steps: u32 = 1;
    var checkpoint_interval: u32 = 0;
    var checkpoint_path: ?[]const u8 = null;
    var max_checkpoints: u32 = 3;
    var log_interval: u32 = 10;
    var export_gguf_path: ?[]const u8 = null;
    var export_gguf_name: ?[]const u8 = null;
    var dataset_path: ?[]const u8 = null;
    var dataset_format: DatasetFormat = .text;
    var dataset_max_tokens: usize = 0;

    var i: usize = 0;
    while (i < args.len) {
        const arg = args[i];
        i += 1;

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--hidden-dim")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                hidden_dim = std.fmt.parseInt(u32, val, 10) catch 256;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--num-layers")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                num_layers = std.fmt.parseInt(u32, val, 10) catch 4;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--num-heads")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                num_heads = std.fmt.parseInt(u32, val, 10) catch 4;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--intermediate-dim")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                intermediate_dim = std.fmt.parseInt(u32, val, 10) catch 512;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--vocab-size")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                vocab_size = std.fmt.parseInt(u32, val, 10) catch 32000;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--max-seq-len")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                max_seq_len = std.fmt.parseInt(u32, val, 10) catch 128;
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--epochs", "-e" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                epochs = std.fmt.parseInt(u32, val, 10) catch 1;
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--batch-size", "-b" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                batch_size = std.fmt.parseInt(u32, val, 10) catch 4;
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--learning-rate", "--lr" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                learning_rate = std.fmt.parseFloat(f32, val) catch 1e-4;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--warmup-steps")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                warmup_steps = std.fmt.parseInt(u32, val, 10) catch 100;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--weight-decay")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                weight_decay = std.fmt.parseFloat(f32, val) catch 0.01;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--gradient-clip")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                gradient_clip = std.fmt.parseFloat(f32, val) catch 1.0;
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--grad-accum", "--gradient-accumulation" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                grad_accum_steps = std.fmt.parseInt(u32, val, 10) catch 1;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--checkpoint-interval")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                checkpoint_interval = std.fmt.parseInt(u32, val, 10) catch 0;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--checkpoint-path")) {
            if (i < args.len) {
                checkpoint_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--max-checkpoints")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                max_checkpoints = std.fmt.parseInt(u32, val, 10) catch 3;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--log-interval")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                log_interval = std.fmt.parseInt(u32, val, 10) catch 10;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--export-gguf")) {
            if (i < args.len) {
                export_gguf_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--export-name")) {
            if (i < args.len) {
                export_gguf_name = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--dataset-path")) {
            if (i < args.len) {
                dataset_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--dataset-format")) {
            if (i < args.len) {
                dataset_format = parseDatasetFormat(std.mem.sliceTo(args[i], 0));
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--dataset-max-tokens")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                dataset_max_tokens = std.fmt.parseInt(usize, val, 10) catch 0;
                i += 1;
            }
            continue;
        }
    }

    // Create model config
    const model_config = abi.ai.TrainableModelConfig{
        .hidden_dim = hidden_dim,
        .num_layers = num_layers,
        .num_heads = num_heads,
        .num_kv_heads = num_heads, // No GQA for simplicity
        .intermediate_dim = intermediate_dim,
        .vocab_size = vocab_size,
        .max_seq_len = max_seq_len,
    };

    const num_params = model_config.numParams();

    // Print configuration
    std.debug.print("New Transformer Model Configuration\n", .{});
    std.debug.print("====================================\n", .{});
    std.debug.print("Architecture:\n", .{});
    std.debug.print("  Hidden dim:       {d}\n", .{hidden_dim});
    std.debug.print("  Num layers:       {d}\n", .{num_layers});
    std.debug.print("  Num heads:        {d}\n", .{num_heads});
    std.debug.print("  Intermediate dim: {d}\n", .{intermediate_dim});
    std.debug.print("  Vocab size:       {d}\n", .{vocab_size});
    std.debug.print("  Max seq len:      {d}\n", .{max_seq_len});
    std.debug.print("  Parameters:       {d} ({d:.2} MB)\n", .{
        num_params,
        @as(f64, @floatFromInt(num_params * 4)) / (1024 * 1024),
    });
    std.debug.print("\nTraining:\n", .{});
    std.debug.print("  Epochs:           {d}\n", .{epochs});
    std.debug.print("  Batch size:       {d}\n", .{batch_size});
    std.debug.print("  Learning rate:    {e:.2}\n", .{learning_rate});
    std.debug.print("  Warmup steps:     {d}\n", .{warmup_steps});
    std.debug.print("  Weight decay:     {d:.4}\n", .{weight_decay});
    std.debug.print("  Gradient clip:    {d:.2}\n", .{gradient_clip});
    std.debug.print("  Grad accumulation:{d}\n", .{grad_accum_steps});
    if (dataset_path) |path| {
        std.debug.print("  Dataset:          {s}\n", .{path});
        std.debug.print("  Dataset format:   {t}\n", .{dataset_format});
    } else {
        std.debug.print("  Dataset:          (synthetic)\n", .{});
    }
    if (export_gguf_path) |path| {
        std.debug.print("  Export GGUF:      {s}\n", .{path});
    }
    std.debug.print("\n", .{});

    // Create model from scratch
    std.debug.print("Initializing model with random weights...\n", .{});
    var model = abi.ai.TrainableModel.init(allocator, model_config) catch |err| {
        std.debug.print("Error initializing model: {t}\n", .{err});
        return;
    };
    defer model.deinit();

    std.debug.print("Model initialized: {d} parameters\n\n", .{num_params});

    // Prepare training tokens
    var train_tokens: []u32 = &.{};
    defer if (train_tokens.len > 0) allocator.free(train_tokens);

    if (dataset_path) |path| {
        // Load dataset if provided
        std.debug.print("Loading dataset from {s}...\n", .{path});

        // For text format without tokenizer, we need to create synthetic tokens
        // based on byte values (simple character-level encoding)
        if (dataset_format == .text) {
            const text = readTextFile(allocator, path) catch |err| {
                std.debug.print("Error reading dataset: {t}\n", .{err});
                return;
            };
            defer allocator.free(text);

            // Simple byte-level tokenization
            train_tokens = allocator.alloc(u32, text.len) catch |err| {
                std.debug.print("Error allocating tokens: {t}\n", .{err});
                return;
            };
            for (text, 0..) |byte, idx| {
                train_tokens[idx] = @as(u32, byte) % vocab_size;
            }

            if (dataset_max_tokens > 0 and train_tokens.len > dataset_max_tokens) {
                const trimmed = allocator.alloc(u32, dataset_max_tokens) catch |err| {
                    std.debug.print("Error trimming tokens: {t}\n", .{err});
                    return;
                };
                @memcpy(trimmed, train_tokens[0..dataset_max_tokens]);
                allocator.free(train_tokens);
                train_tokens = trimmed;
            }
        } else if (dataset_format == .tokenbin) {
            train_tokens = abi.ai.readTokenBinFile(allocator, path) catch |err| {
                std.debug.print("Error reading tokenbin: {t}\n", .{err});
                return;
            };
            if (dataset_max_tokens > 0 and train_tokens.len > dataset_max_tokens) {
                const trimmed = allocator.alloc(u32, dataset_max_tokens) catch |err| {
                    std.debug.print("Error trimming tokens: {t}\n", .{err});
                    return;
                };
                @memcpy(trimmed, train_tokens[0..dataset_max_tokens]);
                allocator.free(train_tokens);
                train_tokens = trimmed;
            }
        } else {
            std.debug.print("Error: JSONL format requires a tokenizer (use --dataset-format text or tokenbin)\n", .{});
            return;
        }

        std.debug.print("Loaded {d} tokens\n\n", .{train_tokens.len});
    } else {
        // Generate synthetic training data
        const num_tokens = @as(usize, max_seq_len) * batch_size * 10;
        train_tokens = allocator.alloc(u32, num_tokens) catch |err| {
            std.debug.print("Error allocating synthetic tokens: {t}\n", .{err});
            return;
        };

        var rng = std.Random.DefaultPrng.init(42);
        for (train_tokens) |*t| {
            t.* = rng.random().intRangeLessThan(u32, 0, vocab_size);
        }
        std.debug.print("Generated {d} synthetic tokens for training\n\n", .{num_tokens});
    }

    // Clamp tokens to vocab size
    clampTokens(train_tokens, vocab_size);

    // Create LLM training config
    var llm_config = abi.ai.LlmTrainingConfig{
        .epochs = epochs,
        .batch_size = batch_size,
        .max_seq_len = max_seq_len,
        .learning_rate = learning_rate,
        .warmup_steps = warmup_steps,
        .weight_decay = weight_decay,
        .max_grad_norm = gradient_clip,
        .grad_accum_steps = grad_accum_steps,
        .checkpoint_interval = checkpoint_interval,
        .checkpoint_path = checkpoint_path,
        .max_checkpoints = max_checkpoints,
        .log_interval = log_interval,
        .log_dir = "logs",
        .enable_metrics_stream = true,
    };

    if (export_gguf_path) |path| {
        llm_config.export_gguf_path = path;
    }
    if (export_gguf_name) |name| {
        llm_config.export_name = name;
    }

    std.debug.print("Starting training from scratch...\n", .{});
    var timer = std.time.Timer.start() catch {
        std.debug.print("Error: Failed to start timer\n", .{});
        return;
    };

    const report = abi.ai.training.llm_trainer.trainLlm(allocator, &model, llm_config, train_tokens) catch |err| {
        std.debug.print("Training failed: {t}\n", .{err});
        return;
    };

    const elapsed_ns = timer.read();
    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;

    std.debug.print("\nTraining Complete\n", .{});
    std.debug.print("=================\n", .{});
    std.debug.print("Final loss:       {d:.6}\n", .{report.final_loss});
    std.debug.print("Final accuracy:   {d:.2}%\n", .{report.final_accuracy * 100});
    std.debug.print("Total steps:      {d}\n", .{report.total_steps});
    std.debug.print("Tokens processed: {d}\n", .{report.total_tokens});
    std.debug.print("Wall time:        {d:.2}s\n", .{elapsed_s});
    std.debug.print("Checkpoints saved:{d}\n", .{report.checkpoints_saved});
    if (export_gguf_path) |path| {
        std.debug.print("Model exported to:{s}\n", .{path});
    }
}

fn printNewHelp() void {
    const help_text =
        \\Usage: abi train new [options]
        \\
        \\Create and train a new transformer model from scratch.
        \\
        \\Architecture options:
        \\  --hidden-dim <n>       Hidden dimension (default: 256)
        \\  --num-layers <n>       Number of layers (default: 4)
        \\  --num-heads <n>        Number of attention heads (default: 4)
        \\  --intermediate-dim <n> FFN intermediate dim (default: 512)
        \\  --vocab-size <n>       Vocabulary size (default: 32000)
        \\  --max-seq-len <n>      Maximum sequence length (default: 128)
        \\
        \\Training options:
        \\  -e, --epochs <n>       Number of epochs (default: 1)
        \\  -b, --batch-size <n>   Batch size (default: 4)
        \\  --lr, --learning-rate  Learning rate (default: 1e-4)
        \\  --warmup-steps <n>     Warmup steps (default: 100)
        \\  --weight-decay <f>     Weight decay (default: 0.01)
        \\  --gradient-clip <f>    Gradient clip norm (default: 1.0)
        \\  --grad-accum <n>       Gradient accumulation (default: 1)
        \\  --log-interval <n>     Log every N steps (default: 10)
        \\
        \\Checkpointing:
        \\  --checkpoint-interval  Steps between checkpoints
        \\  --checkpoint-path      Path to save checkpoints
        \\  --max-checkpoints      Max checkpoints to keep (default: 3)
        \\
        \\Dataset:
        \\  --dataset-path <path>  Local dataset file (text or tokenbin)
        \\  --dataset-format       text, tokenbin (default: text)
        \\  --dataset-max-tokens   Limit tokens for training
        \\
        \\Export:
        \\  --export-gguf <path>   Export trained model as GGUF
        \\  --export-name <name>   Model name in GGUF metadata
        \\
        \\Examples:
        \\  abi train new --epochs 5
        \\  abi train new --hidden-dim 512 --num-layers 8 --num-heads 8
        \\  abi train new --dataset-path data.txt --epochs 10
        \\  abi train new --export-gguf model.gguf --export-name my-model
        \\
    ;
    std.debug.print("{s}", .{help_text});
}

fn runLlmTrain(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (utils.args.containsHelpArgs(args)) {
        printHelp();
        return;
    }

    // Check if LLM feature is enabled
    if (!abi.ai.llm.isEnabled()) {
        std.debug.print("Error: LLM feature is not enabled. Build with -Denable-llm=true\n", .{});
        return;
    }

    if (args.len == 0) {
        std.debug.print("Usage: abi train llm <model.gguf> [options]\n", .{});
        std.debug.print("\nUse 'abi train help' for full options list.\n", .{});
        return;
    }

    // First argument is the model path
    const model_path = std.mem.sliceTo(args[0], 0);

    // Parse LLM training options
    var config = abi.ai.LlmTrainingConfig{};
    var use_gpu: bool = false;
    var dataset_url: ?[]const u8 = null;
    var dataset_path: ?[]const u8 = null;
    var dataset_cache: ?[]const u8 = null;
    var dataset_wdbx: ?[]const u8 = null;
    var dataset_format: DatasetFormat = .tokenbin;
    var dataset_block_tokens: u32 = 2048;
    var dataset_max_tokens: usize = 0;
    var dataset_max_bytes: usize = 64 * 1024 * 1024;
    var export_gguf_path: ?[]const u8 = null;
    var export_gguf_name: ?[]const u8 = null;
    var log_dir: ?[]const u8 = null;
    var i: usize = 1;

    while (i < args.len) {
        const arg = args[i];
        i += 1;

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--epochs", "-e" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.epochs = std.fmt.parseInt(u32, val, 10) catch 1;
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--batch-size", "-b" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.batch_size = std.fmt.parseInt(u32, val, 10) catch 4;
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--learning-rate", "--lr" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.learning_rate = std.fmt.parseFloat(f32, val) catch 1e-5;
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--grad-accum", "--gradient-accumulation" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.grad_accum_steps = std.fmt.parseInt(u32, val, 10) catch 1;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--max-seq-len")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.max_seq_len = std.fmt.parseInt(u32, val, 10) catch 512;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--warmup-steps")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.warmup_steps = std.fmt.parseInt(u32, val, 10) catch 100;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--weight-decay")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.weight_decay = std.fmt.parseFloat(f32, val) catch 0.01;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--gradient-clip")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.max_grad_norm = std.fmt.parseFloat(f32, val) catch 1.0;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--label-smoothing")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.label_smoothing = std.fmt.parseFloat(f32, val) catch 0.0;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--checkpoint-interval")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.checkpoint_interval = std.fmt.parseInt(u32, val, 10) catch 0;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--checkpoint-path")) {
            if (i < args.len) {
                config.checkpoint_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--max-checkpoints")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.max_checkpoints = std.fmt.parseInt(u32, val, 10) catch 3;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--optimizer")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.optimizer = parseOptimizer(val);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--lr-schedule")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.lr_schedule = parseLrSchedule(val);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--use-gpu")) {
            use_gpu = true;
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--mixed-precision")) {
            config.mixed_precision = true;
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--log-interval")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.log_interval = std.fmt.parseInt(u32, val, 10) catch 10;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--dataset-url")) {
            if (i < args.len) {
                dataset_url = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--dataset-path")) {
            if (i < args.len) {
                dataset_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--dataset-cache")) {
            if (i < args.len) {
                dataset_cache = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--dataset-wdbx")) {
            if (i < args.len) {
                dataset_wdbx = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--dataset-format")) {
            if (i < args.len) {
                dataset_format = parseDatasetFormat(std.mem.sliceTo(args[i], 0));
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--dataset-block-tokens")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                dataset_block_tokens = std.fmt.parseInt(u32, val, 10) catch 2048;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--dataset-max-tokens")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                dataset_max_tokens = std.fmt.parseInt(usize, val, 10) catch 0;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--dataset-max-bytes")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                dataset_max_bytes = std.fmt.parseInt(usize, val, 10) catch dataset_max_bytes;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--log-dir")) {
            if (i < args.len) {
                log_dir = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--export-gguf")) {
            if (i < args.len) {
                export_gguf_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--export-name")) {
            if (i < args.len) {
                export_gguf_name = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }
    }

    // Print configuration
    std.debug.print("LLM Training Configuration\n", .{});
    std.debug.print("==========================\n", .{});
    std.debug.print("Model:            {s}\n", .{model_path});
    std.debug.print("Epochs:           {d}\n", .{config.epochs});
    std.debug.print("Batch size:       {d}\n", .{config.batch_size});
    std.debug.print("Max seq len:      {d}\n", .{config.max_seq_len});
    std.debug.print("Learning rate:    {e:.2}\n", .{config.learning_rate});
    std.debug.print("Optimizer:        {t}\n", .{config.optimizer});
    std.debug.print("LR schedule:      {t}\n", .{config.lr_schedule});
    std.debug.print("Warmup steps:     {d}\n", .{config.warmup_steps});
    std.debug.print("Weight decay:     {d:.4}\n", .{config.weight_decay});
    std.debug.print("Gradient clip:    {d:.2}\n", .{config.max_grad_norm});
    std.debug.print("Grad accumulation:{d}\n", .{config.grad_accum_steps});
    if (config.label_smoothing > 0) {
        std.debug.print("Label smoothing:  {d:.2}\n", .{config.label_smoothing});
    }
    if (config.checkpoint_interval > 0) {
        std.debug.print("Checkpoint interval: {d}\n", .{config.checkpoint_interval});
    }
    if (config.checkpoint_path) |path| {
        std.debug.print("Checkpoint path:  {s}\n", .{path});
    }
    std.debug.print("Use GPU:          {}\n", .{use_gpu});
    std.debug.print("Dataset format:   {t}\n", .{dataset_format});
    if (dataset_url) |url| {
        std.debug.print("Dataset URL:      {s}\n", .{url});
    }
    if (dataset_path) |path| {
        std.debug.print("Dataset path:     {s}\n", .{path});
    }
    if (dataset_wdbx) |path| {
        std.debug.print("Dataset WDBX:     {s}\n", .{path});
    }
    if (dataset_max_tokens > 0) {
        std.debug.print("Max tokens:       {d}\n", .{dataset_max_tokens});
    }
    std.debug.print("Mixed precision:  {}\n", .{config.mixed_precision});
    std.debug.print("\n", .{});

    // Load model
    std.debug.print("Loading model from {s}...\n", .{model_path});
    var model = abi.ai.TrainableModel.fromGguf(allocator, model_path) catch |err| {
        std.debug.print("Error loading GGUF model: {t}\n", .{err});
        return;
    };
    defer model.deinit();

    const num_params = model.numParams();
    std.debug.print("Model initialized: {d} parameters ({d:.2} MB)\n\n", .{
        num_params,
        @as(f64, @floatFromInt(num_params * 4)) / (1024 * 1024),
    });

    if (export_gguf_path) |path| {
        config.export_gguf_path = path;
    }
    if (export_gguf_name) |name| {
        config.export_name = name;
    }

    // Default log directory and metrics stream for dashboards
    if (log_dir == null) {
        log_dir = "logs";
    }
    config.log_dir = log_dir;
    config.enable_metrics_stream = true;

    var tokenizer: ?abi.ai.llm.tokenizer.Tokenizer = null;
    defer if (tokenizer) |*tok| tok.deinit();

    if (dataset_format != .tokenbin) {
        var gguf_file = abi.ai.llm.io.gguf.GgufFile.open(allocator, model_path) catch |err| {
            std.debug.print("Error opening GGUF for tokenizer: {t}\n", .{err});
            return;
        };
        defer gguf_file.deinit();

        const tok = abi.ai.llm.tokenizer.loadFromGguf(allocator, &gguf_file) catch |err| {
            std.debug.print("Error loading tokenizer from GGUF: {t}\n", .{err});
            return;
        };
        tokenizer = tok;
    }

    var dataset = try resolveDatasetPath(allocator, dataset_url, dataset_path, dataset_cache, dataset_max_bytes);
    defer if (dataset.owned and dataset.path.len > 0) allocator.free(dataset.path);

    var train_tokens: []u32 = &.{};
    const tokenizer_ptr: ?*abi.ai.llm.tokenizer.Tokenizer = if (tokenizer) |*tok| tok else null;
    if (dataset_wdbx) |db_path| {
        var wdbx_dataset = abi.ai.WdbxTokenDataset.init(allocator, db_path) catch |err| {
            std.debug.print("Error opening WDBX dataset: {t}\n", .{err});
            return;
        };
        defer wdbx_dataset.deinit();

        if (dataset.path.len > 0) {
            const ingest_tokens = try loadTokensFromPath(
                allocator,
                dataset_format,
                dataset.path,
                tokenizer_ptr,
                dataset_max_tokens,
            );
            defer allocator.free(ingest_tokens);

            try wdbx_dataset.importTokenBin(ingest_tokens, dataset_block_tokens);
            try wdbx_dataset.save();
        }

        train_tokens = try wdbx_dataset.collectTokens(dataset_max_tokens);
    } else {
        if (dataset.path.len == 0) {
            std.debug.print("Error: dataset path or URL required when --dataset-wdbx is not provided.\n", .{});
            return;
        }
        train_tokens = try loadTokensFromPath(
            allocator,
            dataset_format,
            dataset.path,
            tokenizer_ptr,
            dataset_max_tokens,
        );
    }
    defer allocator.free(train_tokens);

    if (train_tokens.len == 0) {
        std.debug.print("Error: dataset yielded no tokens.\n", .{});
        return;
    }

    clampTokens(train_tokens, model.config.vocab_size);

    std.debug.print("Starting LLM training...\n", .{});

    const report = abi.ai.training.llm_trainer.trainLlm(allocator, &model, config, train_tokens) catch |err| {
        std.debug.print("Training failed: {t}\n", .{err});
        return;
    };

    std.debug.print("\nTraining Complete\n", .{});
    std.debug.print("=================\n", .{});
    std.debug.print("Final loss:       {d:.6}\n", .{report.final_loss});
    std.debug.print("Final accuracy:   {d:.2}%\n", .{report.final_accuracy * 100});
    std.debug.print("Total steps:      {d}\n", .{report.total_steps});
    std.debug.print("Tokens processed: {d}\n", .{report.total_tokens});
    std.debug.print("Total time:       {d:.2}s\n", .{@as(f64, @floatFromInt(report.total_time_ns)) / 1e9});
    std.debug.print("Checkpoints saved:{d}\n", .{report.checkpoints_saved});
}

fn runVisionTrain(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (utils.args.containsHelpArgs(args)) {
        printVisionHelp();
        return;
    }

    // Check if Vision feature is enabled
    if (!abi.ai.vision.isEnabled()) {
        std.debug.print("Error: Vision feature is not enabled. Build with -Denable-vision=true\n", .{});
        return;
    }

    // Default ViT configuration (tiny model for training from scratch)
    var image_size: u32 = 224;
    var patch_size: u32 = 16;
    var hidden_size: u32 = 384;
    var num_layers: u32 = 12;
    var num_heads: u32 = 6;
    var mlp_dim: u32 = 1536;
    var num_classes: u32 = 1000;
    var dropout: f32 = 0.1;

    // Training config
    var epochs: u32 = 10;
    var batch_size: u32 = 32;
    var learning_rate: f32 = 1e-4;
    var warmup_steps: u32 = 500;
    var weight_decay: f32 = 0.01;
    var gradient_clip: f32 = 1.0;
    var log_interval: u32 = 10;
    var dataset_path: ?[]const u8 = null;

    var i: usize = 0;
    while (i < args.len) {
        const arg = args[i];
        i += 1;

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--image-size")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                image_size = std.fmt.parseInt(u32, val, 10) catch 224;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--patch-size")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                patch_size = std.fmt.parseInt(u32, val, 10) catch 16;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--hidden-size")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                hidden_size = std.fmt.parseInt(u32, val, 10) catch 384;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--num-layers")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                num_layers = std.fmt.parseInt(u32, val, 10) catch 12;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--num-heads")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                num_heads = std.fmt.parseInt(u32, val, 10) catch 6;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--mlp-dim")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                mlp_dim = std.fmt.parseInt(u32, val, 10) catch 1536;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--num-classes")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                num_classes = std.fmt.parseInt(u32, val, 10) catch 1000;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--dropout")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                dropout = std.fmt.parseFloat(f32, val) catch 0.1;
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--epochs", "-e" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                epochs = std.fmt.parseInt(u32, val, 10) catch 10;
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--batch-size", "-b" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                batch_size = std.fmt.parseInt(u32, val, 10) catch 32;
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--learning-rate", "--lr" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                learning_rate = std.fmt.parseFloat(f32, val) catch 1e-4;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--warmup-steps")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                warmup_steps = std.fmt.parseInt(u32, val, 10) catch 500;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--weight-decay")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                weight_decay = std.fmt.parseFloat(f32, val) catch 0.01;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--gradient-clip")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                gradient_clip = std.fmt.parseFloat(f32, val) catch 1.0;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--log-interval")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                log_interval = std.fmt.parseInt(u32, val, 10) catch 10;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--dataset-path")) {
            if (i < args.len) {
                dataset_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }
    }

    // Create ViT config
    const vit_config = abi.ai.vision.ViTConfig{
        .image_size = image_size,
        .patch_size = patch_size,
        .hidden_size = hidden_size,
        .num_layers = num_layers,
        .num_heads = num_heads,
        .mlp_dim = mlp_dim,
        .in_channels = 3,
        .use_class_token = true,
    };

    const trainable_vit_config = abi.ai.TrainableViTConfig{
        .vit_config = vit_config,
        .num_classes = num_classes,
        .dropout = dropout,
    };

    const num_params = trainable_vit_config.numParams();

    // Print configuration
    std.debug.print("Vision Transformer (ViT) Training Configuration\n", .{});
    std.debug.print("================================================\n", .{});
    std.debug.print("Architecture:\n", .{});
    std.debug.print("  Image size:    {d}x{d}\n", .{ image_size, image_size });
    std.debug.print("  Patch size:    {d}x{d}\n", .{ patch_size, patch_size });
    std.debug.print("  Hidden size:   {d}\n", .{hidden_size});
    std.debug.print("  Num layers:    {d}\n", .{num_layers});
    std.debug.print("  Num heads:     {d}\n", .{num_heads});
    std.debug.print("  MLP dim:       {d}\n", .{mlp_dim});
    std.debug.print("  Num classes:   {d}\n", .{num_classes});
    std.debug.print("  Parameters:    {d} ({d:.2} MB)\n", .{
        num_params,
        @as(f64, @floatFromInt(num_params * 4)) / (1024 * 1024),
    });
    std.debug.print("\nTraining:\n", .{});
    std.debug.print("  Epochs:        {d}\n", .{epochs});
    std.debug.print("  Batch size:    {d}\n", .{batch_size});
    std.debug.print("  Learning rate: {e:.2}\n", .{learning_rate});
    std.debug.print("  Warmup steps:  {d}\n", .{warmup_steps});
    std.debug.print("  Weight decay:  {d:.4}\n", .{weight_decay});
    std.debug.print("  Gradient clip: {d:.2}\n", .{gradient_clip});
    std.debug.print("  Dropout:       {d:.2}\n", .{dropout});
    if (dataset_path) |path| {
        std.debug.print("  Dataset:       {s}\n", .{path});
    } else {
        std.debug.print("  Dataset:       (synthetic)\n", .{});
    }
    std.debug.print("\n", .{});

    // Initialize model
    std.debug.print("Initializing ViT model with random weights...\n", .{});
    var model = abi.ai.TrainableViTModel.init(allocator, trainable_vit_config) catch |err| {
        std.debug.print("Error initializing model: {t}\n", .{err});
        return;
    };
    defer model.deinit();

    std.debug.print("Model initialized: {d} parameters\n\n", .{num_params});

    // Generate synthetic training data (images)
    const image_dim = image_size * image_size * 3;
    const num_samples = batch_size * 10;
    var train_images = allocator.alloc(f32, num_samples * image_dim) catch |err| {
        std.debug.print("Error allocating training data: {t}\n", .{err});
        return;
    };
    defer allocator.free(train_images);

    var train_labels = allocator.alloc(u32, num_samples) catch |err| {
        std.debug.print("Error allocating labels: {t}\n", .{err});
        return;
    };
    defer allocator.free(train_labels);

    // Initialize with random data
    var rng = std.Random.DefaultPrng.init(42);
    for (train_images) |*p| {
        p.* = rng.random().float(f32);
    }
    for (train_labels) |*l| {
        l.* = rng.random().intRangeLessThan(u32, 0, num_classes);
    }

    std.debug.print("Generated {d} synthetic images for training\n\n", .{num_samples});
    std.debug.print("Starting Vision training...\n", .{});

    var timer = std.time.Timer.start() catch {
        std.debug.print("Error: Failed to start timer\n", .{});
        return;
    };

    // Training loop
    const batches_per_epoch = num_samples / batch_size;
    var total_loss: f32 = 0;
    var step: u32 = 0;

    for (0..epochs) |epoch| {
        var epoch_loss: f32 = 0;

        for (0..batches_per_epoch) |batch_idx| {
            const batch_start = batch_idx * batch_size * image_dim;
            const batch_images = train_images[batch_start .. batch_start + batch_size * image_dim];

            // Forward pass
            var logits = allocator.alloc(f32, batch_size * num_classes) catch continue;
            defer allocator.free(logits);

            model.forward(batch_images, batch_size, logits) catch continue;

            // Compute cross-entropy loss (simplified)
            var batch_loss: f32 = 0;
            for (0..batch_size) |b| {
                const label = train_labels[batch_idx * batch_size + b];
                const logit_offset = b * num_classes;

                // Softmax + cross-entropy
                var max_logit: f32 = logits[logit_offset];
                for (0..num_classes) |c| {
                    if (logits[logit_offset + c] > max_logit) {
                        max_logit = logits[logit_offset + c];
                    }
                }

                var sum_exp: f32 = 0;
                for (0..num_classes) |c| {
                    sum_exp += @exp(logits[logit_offset + c] - max_logit);
                }

                const log_prob = logits[logit_offset + label] - max_logit - @log(sum_exp);
                batch_loss -= log_prob;
            }
            batch_loss /= @as(f32, @floatFromInt(batch_size));
            epoch_loss += batch_loss;

            // Apply gradient update (SGD)
            _ = model.clipGradients(gradient_clip);
            model.applySgdUpdate(learning_rate);
            model.zeroGradients();

            step += 1;

            if (step % log_interval == 0) {
                std.debug.print("  Step {d}: loss={d:.4}\n", .{ step, batch_loss });
            }
        }

        epoch_loss /= @as(f32, @floatFromInt(batches_per_epoch));
        total_loss = epoch_loss;

        std.debug.print("Epoch {d}/{d}: avg_loss={d:.4}\n", .{ epoch + 1, epochs, epoch_loss });
    }

    const elapsed_ns = timer.read();
    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;

    std.debug.print("\nVision Training Complete\n", .{});
    std.debug.print("========================\n", .{});
    std.debug.print("Final loss:  {d:.6}\n", .{total_loss});
    std.debug.print("Total steps: {d}\n", .{step});
    std.debug.print("Wall time:   {d:.2}s\n", .{elapsed_s});
}

fn runClipTrain(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (utils.args.containsHelpArgs(args)) {
        printClipHelp();
        return;
    }

    // Check if Vision feature is enabled
    if (!abi.ai.vision.isEnabled()) {
        std.debug.print("Error: Vision feature is not enabled. Build with -Denable-vision=true\n", .{});
        return;
    }

    // Default CLIP configuration
    var image_size: u32 = 224;
    var patch_size: u32 = 16;
    var vision_hidden: u32 = 768;
    var vision_layers: u32 = 12;
    const vision_heads: u32 = 12;
    var text_hidden: u32 = 512;
    var text_layers: u32 = 12;
    const text_heads: u32 = 8;
    var projection_dim: u32 = 512;
    var temperature: f32 = 0.07;

    // Training config
    var epochs: u32 = 10;
    var batch_size: u32 = 64;
    var learning_rate: f32 = 1e-4;
    var warmup_steps: u32 = 2000;
    var weight_decay: f32 = 0.1;
    var gradient_clip: f32 = 1.0;
    var log_interval: u32 = 10;
    var dataset_path: ?[]const u8 = null;

    var i: usize = 0;
    while (i < args.len) {
        const arg = args[i];
        i += 1;

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--image-size")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                image_size = std.fmt.parseInt(u32, val, 10) catch 224;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--patch-size")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                patch_size = std.fmt.parseInt(u32, val, 10) catch 16;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--vision-hidden")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                vision_hidden = std.fmt.parseInt(u32, val, 10) catch 768;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--vision-layers")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                vision_layers = std.fmt.parseInt(u32, val, 10) catch 12;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--text-hidden")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                text_hidden = std.fmt.parseInt(u32, val, 10) catch 512;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--text-layers")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                text_layers = std.fmt.parseInt(u32, val, 10) catch 12;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--projection-dim")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                projection_dim = std.fmt.parseInt(u32, val, 10) catch 512;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--temperature")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                temperature = std.fmt.parseFloat(f32, val) catch 0.07;
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--epochs", "-e" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                epochs = std.fmt.parseInt(u32, val, 10) catch 10;
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--batch-size", "-b" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                batch_size = std.fmt.parseInt(u32, val, 10) catch 64;
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--learning-rate", "--lr" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                learning_rate = std.fmt.parseFloat(f32, val) catch 1e-4;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--warmup-steps")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                warmup_steps = std.fmt.parseInt(u32, val, 10) catch 2000;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--weight-decay")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                weight_decay = std.fmt.parseFloat(f32, val) catch 0.1;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--gradient-clip")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                gradient_clip = std.fmt.parseFloat(f32, val) catch 1.0;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--log-interval")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                log_interval = std.fmt.parseInt(u32, val, 10) catch 10;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--dataset-path")) {
            if (i < args.len) {
                dataset_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }
    }

    // Create CLIP config
    const vit_config = abi.ai.vision.ViTConfig{
        .image_size = image_size,
        .patch_size = patch_size,
        .hidden_size = vision_hidden,
        .num_layers = vision_layers,
        .num_heads = vision_heads,
        .mlp_dim = vision_hidden * 4,
        .in_channels = 3,
        .use_class_token = true,
    };

    const vision_config = abi.ai.TrainableViTConfig{
        .vit_config = vit_config,
        .num_classes = 0, // CLIP uses projection
        .projection_dim = projection_dim,
    };

    const clip_config = abi.ai.CLIPTrainingConfig{
        .vision_config = vision_config,
        .text_hidden_size = text_hidden,
        .text_num_layers = text_layers,
        .text_num_heads = text_heads,
        .projection_dim = projection_dim,
        .temperature = temperature,
        .learnable_temperature = true,
    };

    const num_params = clip_config.numParams();

    // Print configuration
    std.debug.print("CLIP (Contrastive Language-Image Pretraining) Configuration\n", .{});
    std.debug.print("============================================================\n", .{});
    std.debug.print("Vision Encoder:\n", .{});
    std.debug.print("  Image size:    {d}x{d}\n", .{ image_size, image_size });
    std.debug.print("  Patch size:    {d}x{d}\n", .{ patch_size, patch_size });
    std.debug.print("  Hidden size:   {d}\n", .{vision_hidden});
    std.debug.print("  Num layers:    {d}\n", .{vision_layers});
    std.debug.print("  Num heads:     {d}\n", .{vision_heads});
    std.debug.print("\nText Encoder:\n", .{});
    std.debug.print("  Hidden size:   {d}\n", .{text_hidden});
    std.debug.print("  Num layers:    {d}\n", .{text_layers});
    std.debug.print("  Num heads:     {d}\n", .{text_heads});
    std.debug.print("\nContrastive:\n", .{});
    std.debug.print("  Projection dim:{d}\n", .{projection_dim});
    std.debug.print("  Temperature:   {d:.4}\n", .{temperature});
    std.debug.print("  Parameters:    {d} ({d:.2} MB)\n", .{
        num_params,
        @as(f64, @floatFromInt(num_params * 4)) / (1024 * 1024),
    });
    std.debug.print("\nTraining:\n", .{});
    std.debug.print("  Epochs:        {d}\n", .{epochs});
    std.debug.print("  Batch size:    {d}\n", .{batch_size});
    std.debug.print("  Learning rate: {e:.2}\n", .{learning_rate});
    std.debug.print("  Warmup steps:  {d}\n", .{warmup_steps});
    std.debug.print("  Weight decay:  {d:.4}\n", .{weight_decay});
    std.debug.print("  Gradient clip: {d:.2}\n", .{gradient_clip});
    if (dataset_path) |path| {
        std.debug.print("  Dataset:       {s}\n", .{path});
    } else {
        std.debug.print("  Dataset:       (synthetic)\n", .{});
    }
    std.debug.print("\n", .{});

    // Initialize model
    std.debug.print("Initializing CLIP model with random weights...\n", .{});
    var model = abi.ai.TrainableCLIPModel.init(allocator, clip_config) catch |err| {
        std.debug.print("Error initializing model: {t}\n", .{err});
        return;
    };
    defer model.deinit();

    std.debug.print("Model initialized: {d} parameters\n\n", .{num_params});

    // Generate synthetic training data (image-text pairs)
    const image_dim = image_size * image_size * 3;
    const text_max_len: u32 = 77;
    const num_samples = batch_size * 10;

    var train_images = allocator.alloc(f32, num_samples * image_dim) catch |err| {
        std.debug.print("Error allocating training images: {t}\n", .{err});
        return;
    };
    defer allocator.free(train_images);

    var train_tokens = allocator.alloc(u32, num_samples * text_max_len) catch |err| {
        std.debug.print("Error allocating training tokens: {t}\n", .{err});
        return;
    };
    defer allocator.free(train_tokens);

    // Initialize with random data
    var rng = std.Random.DefaultPrng.init(42);
    for (train_images) |*p| {
        p.* = rng.random().float(f32);
    }
    for (train_tokens) |*t| {
        t.* = rng.random().intRangeLessThan(u32, 0, clip_config.text_vocab_size);
    }

    std.debug.print("Generated {d} synthetic image-text pairs for training\n\n", .{num_samples});
    std.debug.print("Starting CLIP contrastive training...\n", .{});

    var timer = std.time.Timer.start() catch {
        std.debug.print("Error: Failed to start timer\n", .{});
        return;
    };

    // Training loop
    const batches_per_epoch = num_samples / batch_size;
    var total_loss: f32 = 0;
    var step: u32 = 0;

    // Allocate embedding buffers
    const image_embeddings = allocator.alloc(f32, batch_size * projection_dim) catch |err| {
        std.debug.print("Error allocating embeddings: {t}\n", .{err});
        return;
    };
    defer allocator.free(image_embeddings);

    const text_embeddings = allocator.alloc(f32, batch_size * projection_dim) catch |err| {
        std.debug.print("Error allocating embeddings: {t}\n", .{err});
        return;
    };
    defer allocator.free(text_embeddings);

    const d_image_emb = allocator.alloc(f32, batch_size * projection_dim) catch |err| {
        std.debug.print("Error allocating gradients: {t}\n", .{err});
        return;
    };
    defer allocator.free(d_image_emb);

    const d_text_emb = allocator.alloc(f32, batch_size * projection_dim) catch |err| {
        std.debug.print("Error allocating gradients: {t}\n", .{err});
        return;
    };
    defer allocator.free(d_text_emb);

    for (0..epochs) |epoch| {
        var epoch_loss: f32 = 0;

        for (0..batches_per_epoch) |batch_idx| {
            const img_start = batch_idx * batch_size * image_dim;
            const txt_start = batch_idx * batch_size * text_max_len;

            const batch_images = train_images[img_start .. img_start + batch_size * image_dim];
            const batch_tokens = train_tokens[txt_start .. txt_start + batch_size * text_max_len];

            // Encode images and text
            model.encodeImages(batch_images, batch_size, image_embeddings) catch continue;
            model.encodeText(batch_tokens, batch_size, text_embeddings) catch continue;

            // Compute contrastive loss
            const loss = model.computeContrastiveLoss(
                image_embeddings,
                text_embeddings,
                batch_size,
                d_image_emb,
                d_text_emb,
            );
            epoch_loss += loss;

            // Apply gradient update (SGD)
            model.applySgdUpdate(learning_rate);
            model.zeroGradients();

            step += 1;

            if (step % log_interval == 0) {
                std.debug.print("  Step {d}: loss={d:.4}\n", .{ step, loss });
            }
        }

        epoch_loss /= @as(f32, @floatFromInt(batches_per_epoch));
        total_loss = epoch_loss;

        std.debug.print("Epoch {d}/{d}: avg_loss={d:.4}\n", .{ epoch + 1, epochs, epoch_loss });
    }

    const elapsed_ns = timer.read();
    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;

    std.debug.print("\nCLIP Training Complete\n", .{});
    std.debug.print("======================\n", .{});
    std.debug.print("Final loss:  {d:.6}\n", .{total_loss});
    std.debug.print("Total steps: {d}\n", .{step});
    std.debug.print("Wall time:   {d:.2}s\n", .{elapsed_s});
}

fn printVisionHelp() void {
    const help_text =
        \\Usage: abi train vision [options]
        \\
        \\Train a Vision Transformer (ViT) model for image classification.
        \\
        \\Architecture options:
        \\  --image-size <n>     Input image size (default: 224)
        \\  --patch-size <n>     Patch size (default: 16)
        \\  --hidden-size <n>    Hidden dimension (default: 384)
        \\  --num-layers <n>     Number of transformer layers (default: 12)
        \\  --num-heads <n>      Number of attention heads (default: 6)
        \\  --mlp-dim <n>        MLP hidden dimension (default: 1536)
        \\  --num-classes <n>    Number of output classes (default: 1000)
        \\  --dropout <f>        Dropout rate (default: 0.1)
        \\
        \\Training options:
        \\  -e, --epochs <n>     Number of epochs (default: 10)
        \\  -b, --batch-size <n> Batch size (default: 32)
        \\  --lr, --learning-rate <f> Learning rate (default: 1e-4)
        \\  --warmup-steps <n>   Warmup steps (default: 500)
        \\  --weight-decay <f>   Weight decay (default: 0.01)
        \\  --gradient-clip <f>  Gradient clip norm (default: 1.0)
        \\  --log-interval <n>   Log every N steps (default: 10)
        \\  --dataset-path <path> Dataset directory
        \\
        \\Examples:
        \\  abi train vision --epochs 10 --batch-size 64
        \\  abi train vision --hidden-size 768 --num-layers 12 --num-heads 12
        \\  abi train vision --dataset-path ./imagenet --epochs 90
        \\
    ;
    std.debug.print("{s}", .{help_text});
}

fn printClipHelp() void {
    const help_text =
        \\Usage: abi train clip [options]
        \\
        \\Train a CLIP (Contrastive Language-Image Pretraining) model.
        \\
        \\Vision encoder options:
        \\  --image-size <n>     Input image size (default: 224)
        \\  --patch-size <n>     Patch size (default: 16)
        \\  --vision-hidden <n>  Vision hidden dimension (default: 768)
        \\  --vision-layers <n>  Vision transformer layers (default: 12)
        \\
        \\Text encoder options:
        \\  --text-hidden <n>    Text hidden dimension (default: 512)
        \\  --text-layers <n>    Text transformer layers (default: 12)
        \\
        \\Contrastive options:
        \\  --projection-dim <n> Shared projection dimension (default: 512)
        \\  --temperature <f>    Temperature for InfoNCE loss (default: 0.07)
        \\
        \\Training options:
        \\  -e, --epochs <n>     Number of epochs (default: 10)
        \\  -b, --batch-size <n> Batch size (default: 64)
        \\  --lr, --learning-rate <f> Learning rate (default: 1e-4)
        \\  --warmup-steps <n>   Warmup steps (default: 2000)
        \\  --weight-decay <f>   Weight decay (default: 0.1)
        \\  --gradient-clip <f>  Gradient clip norm (default: 1.0)
        \\  --log-interval <n>   Log every N steps (default: 10)
        \\  --dataset-path <path> Image-text pairs dataset
        \\
        \\Examples:
        \\  abi train clip --epochs 10 --batch-size 256
        \\  abi train clip --vision-hidden 768 --text-hidden 512 --projection-dim 512
        \\  abi train clip --dataset-path ./laion --epochs 32
        \\
    ;
    std.debug.print("{s}", .{help_text});
}

fn runResume(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (utils.args.containsHelpArgs(args)) {
        printHelp();
        return;
    }

    if (args.len == 0) {
        std.debug.print("Usage: abi train resume <checkpoint-path>\n", .{});
        return;
    }

    const checkpoint_path = std.mem.sliceTo(args[0], 0);
    std.debug.print("Loading checkpoint: {s}\n", .{checkpoint_path});

    // Load checkpoint
    var ckpt = abi.ai.loadCheckpoint(allocator, checkpoint_path) catch |err| {
        std.debug.print("Error loading checkpoint: {t}\n", .{err});
        std.debug.print("\nNote: Resume functionality loads model weights from a saved checkpoint.\n", .{});
        std.debug.print("Use 'abi train run --checkpoint-path <path>' to save checkpoints during training.\n", .{});
        return;
    };
    defer ckpt.deinit(allocator);

    std.debug.print("\nCheckpoint Info:\n", .{});
    std.debug.print("  Step:      {d}\n", .{ckpt.step});
    std.debug.print("  Timestamp: {d}\n", .{ckpt.timestamp});
    std.debug.print("  Weights:   {d} parameters\n", .{ckpt.weights.len});
    std.debug.print("\nNote: Full resume training not yet implemented.\n", .{});
    std.debug.print("Checkpoint loaded successfully.\n", .{});
}

fn runMonitor(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (utils.args.containsHelpArgs(args)) {
        printMonitorHelp();
        return;
    }

    // Parse optional run-id argument
    var run_id: ?[]const u8 = null;
    var log_dir: []const u8 = "logs";
    var non_interactive = false;

    var i: usize = 0;
    while (i < args.len) {
        const arg = args[i];
        i += 1;

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--log-dir")) {
            if (i < args.len) {
                log_dir = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--no-tui") or
            std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--non-interactive"))
        {
            non_interactive = true;
            continue;
        }

        // First non-option argument is run-id
        if (run_id == null and arg[0] != '-') {
            run_id = std.mem.sliceTo(arg, 0);
        }
    }

    // Use the default theme from the theme system
    const theme = &tui.themes.themes.default;

    var panel = tui.TrainingPanel.init(allocator, theme, .{
        .log_dir = log_dir,
        .run_id = run_id,
    });
    defer panel.deinit();

    // Try interactive mode if terminal is supported
    if (!non_interactive and tui.Terminal.isSupported()) {
        var terminal = tui.Terminal.init(allocator);
        defer terminal.deinit();

        panel.runInteractive(&terminal) catch |err| {
            // Fall back to non-interactive mode on error
            std.debug.print("Interactive mode failed ({t}), falling back to snapshot mode.\n\n", .{err});
            non_interactive = true;
        };

        if (!non_interactive) return;
    }

    // Non-interactive fallback: render single snapshot
    const DebugWriter = struct {
        pub const Error = error{};
        pub fn print(_: @This(), comptime fmt: []const u8, print_args: anytype) Error!void {
            std.debug.print(fmt, print_args);
        }
    };

    // Load metrics before rendering
    panel.loadMetricsFile(panel.buildMetricsPath()) catch {};

    panel.render(DebugWriter{}) catch |err| {
        std.debug.print("Error rendering panel: {t}\n", .{err});
        return;
    };

    std.debug.print("\nTraining Monitor (snapshot mode)\n", .{});
    std.debug.print("Log directory: {s}\n", .{log_dir});
    if (run_id) |id| {
        std.debug.print("Run ID: {s}\n", .{id});
    } else {
        std.debug.print("Monitoring: current/latest run\n", .{});
    }
    std.debug.print("\nRun without --no-tui for interactive mode.\n", .{});
}

fn printMonitorHelp() void {
    const help_text =
        \\Usage: abi train monitor [run-id] [options]
        \\
        \\Monitor training progress with a TUI dashboard.
        \\
        \\Options:
        \\  --log-dir <path>    Log directory (default: logs)
        \\
        \\Arguments:
        \\  run-id              Optional run ID to monitor (default: latest)
        \\
        \\Keyboard controls:
        \\  r       Refresh display
        \\  h       Toggle history mode
        \\  q       Quit
        \\  ?       Show help
        \\  ←/→     Switch between runs (history mode)
        \\
        \\Examples:
        \\  abi train monitor                    # Monitor latest run
        \\  abi train monitor run-2026-01-24     # Monitor specific run
        \\  abi train monitor --log-dir ./logs   # Custom log directory
        \\
    ;
    std.debug.print("{s}", .{help_text});
}

fn runInfo() void {
    const default_config = abi.ai.TrainingConfig{};

    std.debug.print("Default Training Configuration\n", .{});
    std.debug.print("==============================\n\n", .{});
    std.debug.print("Recommended Model: GPT-2 Small (124M parameters)\n", .{});
    std.debug.print("  - Open source, no authentication required\n", .{});
    std.debug.print("  - Download: https://huggingface.co/TheBloke/gpt2-GGUF\n", .{});
    std.debug.print("  - Training: abi train llm models/gpt2.gguf --epochs 1\n\n", .{});
    std.debug.print("Basic parameters:\n", .{});
    std.debug.print("  epochs:                    {d}\n", .{default_config.epochs});
    std.debug.print("  batch_size:                {d}\n", .{default_config.batch_size});
    std.debug.print("  sample_count:              {d}\n", .{default_config.sample_count});
    std.debug.print("  model_size:                {d}\n", .{default_config.model_size});
    std.debug.print("\nOptimizer settings:\n", .{});
    std.debug.print("  learning_rate:             {d:.6}\n", .{default_config.learning_rate});
    std.debug.print("  optimizer:                 {t}\n", .{default_config.optimizer});
    std.debug.print("  learning_rate_schedule:    {t}\n", .{default_config.learning_rate_schedule});
    std.debug.print("  warmup_steps:              {d}\n", .{default_config.warmup_steps});
    std.debug.print("  decay_steps:               {d}\n", .{default_config.decay_steps});
    std.debug.print("  min_learning_rate:         {d:.6}\n", .{default_config.min_learning_rate});
    std.debug.print("  weight_decay:              {d:.4}\n", .{default_config.weight_decay});
    std.debug.print("\nGradient settings:\n", .{});
    std.debug.print("  gradient_accumulation_steps:{d}\n", .{default_config.gradient_accumulation_steps});
    std.debug.print("  gradient_clip_norm:        {d:.2}\n", .{default_config.gradient_clip_norm});
    std.debug.print("\nCheckpointing:\n", .{});
    std.debug.print("  checkpoint_interval:       {d}\n", .{default_config.checkpoint_interval});
    std.debug.print("  max_checkpoints:           {d}\n", .{default_config.max_checkpoints});
    std.debug.print("\nEarly stopping:\n", .{});
    std.debug.print("  early_stopping_patience:   {d}\n", .{default_config.early_stopping_patience});
    std.debug.print("  early_stopping_threshold:  {d:.6}\n", .{default_config.early_stopping_threshold});
    std.debug.print("\nOther:\n", .{});
    std.debug.print("  mixed_precision:           {}\n", .{default_config.mixed_precision});
    std.debug.print("\nAvailable optimizers: sgd, adam, adamw\n", .{});
    std.debug.print("Available LR schedules: constant, cosine, warmup_cosine, step, polynomial\n", .{});
}

fn parseOptimizer(val: []const u8) abi.ai.OptimizerType {
    if (std.mem.eql(u8, val, "sgd")) return .sgd;
    if (std.mem.eql(u8, val, "adam")) return .adam;
    if (std.mem.eql(u8, val, "adamw")) return .adamw;
    return .adamw; // default
}

fn parseLrSchedule(val: []const u8) abi.ai.LearningRateSchedule {
    if (std.mem.eql(u8, val, "constant")) return .constant;
    if (std.mem.eql(u8, val, "cosine")) return .cosine;
    if (std.mem.eql(u8, val, "warmup_cosine")) return .warmup_cosine;
    if (std.mem.eql(u8, val, "step")) return .step;
    if (std.mem.eql(u8, val, "polynomial")) return .polynomial;
    return .warmup_cosine; // default
}

const DatasetFormat = enum {
    tokenbin,
    text,
    jsonl,
};

const DatasetPath = struct {
    path: []const u8,
    owned: bool,
};

fn parseDatasetFormat(val: []const u8) DatasetFormat {
    if (std.mem.eql(u8, val, "text")) return .text;
    if (std.mem.eql(u8, val, "jsonl")) return .jsonl;
    return .tokenbin;
}

fn resolveDatasetPath(
    allocator: std.mem.Allocator,
    dataset_url: ?[]const u8,
    dataset_path: ?[]const u8,
    dataset_cache: ?[]const u8,
    max_bytes: usize,
) !DatasetPath {
    if (dataset_path) |path| {
        return .{ .path = path, .owned = false };
    }
    if (dataset_url == null) {
        return .{ .path = "", .owned = false };
    }

    const url = dataset_url.?;
    const cache_path = if (dataset_cache) |path|
        path
    else
        try defaultDatasetCachePath(allocator, url);
    const owned = dataset_cache == null;

    if (!fileExists(cache_path)) {
        try downloadToFile(allocator, url, cache_path, max_bytes);
    }

    return .{ .path = cache_path, .owned = owned };
}

fn defaultDatasetCachePath(allocator: std.mem.Allocator, url: []const u8) ![]const u8 {
    var name: []const u8 = "dataset.bin";
    if (std.mem.lastIndexOfScalar(u8, url, '/')) |idx| {
        const tail = url[idx + 1 ..];
        if (tail.len > 0) name = tail;
    }

    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();
    std.Io.Dir.cwd().createDirPath(io, "datasets") catch {};

    return std.fs.path.join(allocator, &.{ "datasets", name });
}

fn downloadToFile(allocator: std.mem.Allocator, url: []const u8, path: []const u8, max_bytes: usize) !void {
    var client = try abi.utils.async_http.AsyncHttpClient.init(allocator);
    defer client.deinit();

    var request = try abi.utils.async_http.HttpRequest.init(allocator, .get, url);
    defer request.deinit();

    var response = try client.fetch(&request);
    defer response.deinit();

    if (!response.isSuccess()) {
        return error.DownloadFailed;
    }
    if (max_bytes > 0 and response.body.len > max_bytes) {
        return error.PayloadTooLarge;
    }

    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    defer file.close(io);
    try file.writeStreamingAll(io, response.body);
}

fn loadTokensFromPath(
    allocator: std.mem.Allocator,
    format: DatasetFormat,
    path: []const u8,
    tokenizer: ?*abi.ai.llm.tokenizer.Tokenizer,
    max_tokens: usize,
) ![]u32 {
    switch (format) {
        .tokenbin => {
            var tokens = try abi.ai.readTokenBinFile(allocator, path);
            if (max_tokens > 0 and tokens.len > max_tokens) {
                const trimmed = try allocator.alloc(u32, max_tokens);
                @memcpy(trimmed, tokens[0..max_tokens]);
                allocator.free(tokens);
                tokens = trimmed;
            }
            return tokens;
        },
        .text => {
            const text = try readTextFile(allocator, path);
            defer allocator.free(text);
            if (tokenizer == null) return error.InvalidTokenizer;
            var tokens = try tokenizer.?.encode(allocator, text);
            if (max_tokens > 0 and tokens.len > max_tokens) {
                const trimmed = try allocator.alloc(u32, max_tokens);
                @memcpy(trimmed, tokens[0..max_tokens]);
                allocator.free(tokens);
                tokens = trimmed;
            }
            return tokens;
        },
        .jsonl => {
            const text = try readTextFile(allocator, path);
            defer allocator.free(text);
            if (tokenizer == null) return error.InvalidTokenizer;
            return try tokenizeJsonl(allocator, tokenizer.?, text, max_tokens);
        },
    }
}

fn readTextFile(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    return std.Io.Dir.cwd().readFileAlloc(
        io,
        path,
        allocator,
        .limited(256 * 1024 * 1024),
    );
}

fn tokenizeJsonl(
    allocator: std.mem.Allocator,
    tokenizer: *abi.ai.llm.tokenizer.Tokenizer,
    data: []const u8,
    max_tokens: usize,
) ![]u32 {
    var tokens = std.ArrayListUnmanaged(u32).empty;
    errdefer tokens.deinit(allocator);

    var lines = std.mem.splitScalar(u8, data, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        var text = line;

        const parsed = std.json.parseFromSlice(
            struct {
                text: ?[]const u8 = null,
                instruction: ?[]const u8 = null,
                input: ?[]const u8 = null,
                output: ?[]const u8 = null,
            },
            allocator,
            line,
            .{},
        ) catch null;

        if (parsed) |p| {
            defer p.deinit();
            if (p.value.text) |t| {
                text = t;
            } else if (p.value.instruction != null or p.value.output != null) {
                var buf = std.ArrayListUnmanaged(u8).empty;
                defer buf.deinit(allocator);
                if (p.value.instruction) |instr| {
                    try buf.appendSlice(allocator, instr);
                }
                if (p.value.input) |inp| {
                    if (buf.items.len > 0) try buf.appendSlice(allocator, "\n");
                    try buf.appendSlice(allocator, inp);
                }
                if (p.value.output) |out| {
                    if (buf.items.len > 0) try buf.appendSlice(allocator, "\n");
                    try buf.appendSlice(allocator, out);
                }
                text = try buf.toOwnedSlice(allocator);
                defer allocator.free(@constCast(text));
            }
        }

        const line_tokens = try tokenizer.encode(allocator, text);
        defer allocator.free(line_tokens);
        try appendTokensWithLimit(allocator, &tokens, line_tokens, max_tokens);
        if (max_tokens > 0 and tokens.items.len >= max_tokens) break;
    }

    return tokens.toOwnedSlice(allocator);
}

fn appendTokensWithLimit(
    allocator: std.mem.Allocator,
    out: *std.ArrayListUnmanaged(u32),
    tokens: []const u32,
    max_tokens: usize,
) !void {
    var idx: usize = 0;
    while (idx < tokens.len) : (idx += 1) {
        if (max_tokens > 0 and out.items.len >= max_tokens) break;
        try out.append(allocator, tokens[idx]);
    }
}

fn clampTokens(tokens: []u32, vocab_size: u32) void {
    if (vocab_size == 0) return;
    const max_id = vocab_size - 1;
    for (tokens) |*t| {
        if (t.* > max_id) t.* = max_id;
    }
}

fn fileExists(path: []const u8) bool {
    var io_backend = std.Io.Threaded.init(std.heap.page_allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return false;
    file.close(io);
    return true;
}

fn printHelp() void {
    const help_text =
        \\Usage: abi train <command> [options]
        \\
        \\Run the training pipeline for neural network models.
        \\
        \\Commands:
        \\  run [options]           Run basic training with specified configuration
        \\  new [options]           Create and train a new transformer from scratch
        \\  llm <model> [options]   Train LLM from GGUF model file
        \\  vision [options]        Train Vision Transformer (ViT) for image classification
        \\  clip [options]          Train CLIP multimodal model (vision + text)
        \\  resume <checkpoint>     Resume training from a checkpoint file
        \\  monitor [run-id]        Monitor training progress (TUI dashboard)
        \\  info                    Show default training configuration
        \\  help                    Show this help message
        \\
        \\Basic training options (for 'run'):
        \\  -e, --epochs <n>           Number of epochs (default: 10)
        \\  -b, --batch-size <n>       Batch size (default: 32)
        \\  --model-size <n>           Model parameters (default: 512)
        \\  --sample-count <n>         Training samples (default: 1024)
        \\  --lr, --learning-rate <f>  Learning rate (default: 0.001)
        \\  --optimizer <type>         sgd, adam, adamw (default: adamw)
        \\  --lr-schedule <type>       LR schedule (default: warmup_cosine)
        \\  --warmup-steps <n>         Warmup steps (default: 100)
        \\  --decay-steps <n>          Decay steps (default: 1000)
        \\  --weight-decay <f>         Weight decay (default: 0.01)
        \\  --gradient-clip <f>        Gradient clip norm (default: 1.0)
        \\  --gradient-accumulation <n> Gradient accumulation steps (default: 1)
        \\  --checkpoint-interval <n>  Steps between checkpoints (default: 0)
        \\  --checkpoint-path <path>   Path to save checkpoints
        \\  --max-checkpoints <n>      Max checkpoints to retain (default: 5)
        \\  --early-stopping-patience <n> Early stopping patience (default: 5)
        \\  --mixed-precision          Enable mixed precision training
        \\
        \\LLM training options (for 'llm'):
        \\  -e, --epochs <n>           Number of epochs (default: 1)
        \\  -b, --batch-size <n>       Batch size (default: 4)
        \\  --lr, --learning-rate <f>  Learning rate (default: 1e-5)
        \\  --grad-accum <n>           Gradient accumulation steps (default: 1)
        \\  --max-seq-len <n>          Maximum sequence length (default: 512)
        \\  --warmup-steps <n>         Warmup steps (default: 100)
        \\  --weight-decay <f>         Weight decay (default: 0.01)
        \\  --gradient-clip <f>        Gradient clip norm (default: 1.0)
        \\  --label-smoothing <f>      Label smoothing (default: 0.0)
        \\  --optimizer <type>         sgd, adam, adamw (default: adamw)
        \\  --lr-schedule <type>       LR schedule (default: warmup_cosine)
        \\  --checkpoint-interval <n>  Steps between checkpoints (default: 0)
        \\  --checkpoint-path <path>   Path to save checkpoints
        \\  --max-checkpoints <n>      Max checkpoints to retain (default: 3)
        \\  --use-gpu                  Enable GPU acceleration (cuBLAS)
        \\  --mixed-precision          Enable mixed precision training
        \\  --log-interval <n>         Steps between log outputs (default: 10)
        \\  --export-gguf <path>       Export GGUF weights after training
        \\  --export-name <name>       GGUF model name metadata (default: abi-llama)
        \\  --dataset-url <url>        Download dataset from URL
        \\  --dataset-path <path>      Load dataset from local path
        \\  --dataset-cache <path>     Cache downloaded dataset
        \\  --dataset-wdbx <path>      Store or read dataset from WDBX
        \\  --dataset-format <type>    tokenbin, text, jsonl (default: tokenbin)
        \\  --dataset-block-tokens <n> Tokens per WDBX block (default: 2048)
        \\  --dataset-max-tokens <n>   Limit tokens used for training
        \\  --dataset-max-bytes <n>    Limit download size in bytes
        \\
        \\LR schedules: constant, cosine, warmup_cosine, step, polynomial
        \\Optimizers: sgd, adam, adamw
        \\
        \\Examples:
        \\  abi train run --epochs 10 --batch-size 32
        \\  abi train run -e 5 -b 16 --optimizer adam --lr 0.0001
        \\  abi train new --hidden-dim 256 --num-layers 4 --epochs 5
        \\  abi train new --dataset-path data.txt --export-gguf model.gguf
        \\  abi train llm model.gguf --epochs 1 --batch-size 4 --lr 1e-5
        \\  abi train llm model.gguf --grad-accum 8 --max-seq-len 512 --use-gpu
        \\  abi train llm model.gguf --checkpoint-interval 100 --checkpoint-path ./ckpt
        \\  abi train llm model.gguf --dataset-url https://example.com/data.bin --dataset-format tokenbin
        \\  abi train llm model.gguf --dataset-path data.jsonl --dataset-format jsonl
        \\  abi train llm model.gguf --dataset-url https://example.com/text.txt --dataset-format text --dataset-wdbx ./data/train.wdbx
        \\  abi train vision --epochs 10 --batch-size 64 --hidden-size 384
        \\  abi train vision --image-size 224 --num-layers 12 --num-heads 6
        \\  abi train clip --epochs 10 --batch-size 256 --projection-dim 512
        \\  abi train clip --vision-hidden 768 --text-hidden 512 --temperature 0.07
        \\  abi train resume ./checkpoints/model.ckpt
        \\  abi train info
        \\
    ;
    std.debug.print("{s}", .{help_text});
}
