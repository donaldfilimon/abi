//! LLM training handler.
//!
//! Handles the `abi train llm` subcommand which fine-tunes an existing
//! GGUF model with configurable training parameters, dataset loading,
//! and optional GGUF export.

const std = @import("std");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");
const common = @import("common.zig");
const mod = @import("mod.zig");

pub fn runLlmTrain(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (utils.args.containsHelpArgs(args)) {
        mod.printHelp();
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
    var use_gpu: bool = true;
    var dataset_url: ?[]const u8 = null;
    var dataset_path: ?[]const u8 = null;
    var dataset_cache: ?[]const u8 = null;
    var dataset_wdbx: ?[]const u8 = null;
    var dataset_format: common.DatasetFormat = .tokenbin;
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
                config.optimizer = common.parseOptimizer(val);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--lr-schedule")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.lr_schedule = common.parseLrSchedule(val);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--use-gpu")) {
            use_gpu = true;
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--cpu-only") or
            std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--no-gpu"))
        {
            use_gpu = false;
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
                dataset_format = common.parseDatasetFormat(std.mem.sliceTo(args[i], 0));
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
    config.use_gpu = use_gpu;
    std.debug.print("Use GPU:          {}\n", .{config.use_gpu});
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

    const dataset = try common.resolveDatasetPath(allocator, dataset_url, dataset_path, dataset_cache, dataset_max_bytes);
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
            const ingest_tokens = try common.loadTokensFromPath(
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
        train_tokens = try common.loadTokensFromPath(
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

    common.clampTokens(train_tokens, model.config.vocab_size);

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
