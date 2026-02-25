//! Create and train a new transformer model from scratch.
//!
//! Handles the `abi train new` subcommand which creates a new model with
//! configurable architecture and trains it. Also handles GGUF export and
//! external quantization via llama-quantize.

const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");
const common = @import("common.zig");

const cli_io = common.cli_io;
const gguf_writer = common.gguf_writer;

pub const ByteTokenizerBuild = struct {
    tokens: []const []const u8,
    backing: []u8,
    extra: std.ArrayListUnmanaged([]const u8),

    pub fn init(allocator: std.mem.Allocator, vocab_size: u32) !ByteTokenizerBuild {
        const count: usize = @intCast(vocab_size);
        var tokens = try allocator.alloc([]const u8, count);
        errdefer allocator.free(tokens);

        const byte_count = @min(count, 256);
        var backing = try allocator.alloc(u8, byte_count);
        errdefer allocator.free(backing);

        for (0..byte_count) |i| {
            backing[i] = @intCast(i);
            tokens[i] = backing[i .. i + 1];
        }

        var extra = std.ArrayListUnmanaged([]const u8).empty;
        errdefer extra.deinit(allocator);

        if (count > byte_count) {
            for (byte_count..count) |i| {
                const label = try std.fmt.allocPrint(allocator, "<tok_{d}>", .{i});
                try extra.append(allocator, label);
                tokens[i] = label;
            }
        }

        return .{
            .tokens = tokens,
            .backing = backing,
            .extra = extra,
        };
    }

    pub fn deinit(self: *ByteTokenizerBuild, allocator: std.mem.Allocator) void {
        for (self.extra.items) |entry| {
            allocator.free(entry);
        }
        self.extra.deinit(allocator);
        allocator.free(self.backing);
        allocator.free(self.tokens);
        self.* = undefined;
    }

    pub fn toConfig(self: *const ByteTokenizerBuild) gguf_writer.TokenizerConfig {
        const unk_id: u32 = 0;
        return .{
            .model = "gpt2",
            .tokens = self.tokens,
            .bos_token_id = 1,
            .eos_token_id = 2,
            .unknown_token_id = unk_id,
            .padding_token_id = 0,
            .add_bos_token = false,
            .add_eos_token = false,
        };
    }
};

pub fn runNewModel(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (utils.args.containsHelpArgs(args)) {
        printNewHelp();
        return;
    }

    // Check if LLM feature is enabled
    if (!abi.ai.llm.isEnabled()) {
        utils.output.printError("LLM feature is not enabled. Build with -Denable-llm=true", .{});
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
    var external_quantize: ?[]const u8 = null;
    var dataset_path: ?[]const u8 = null;
    var dataset_format: common.DatasetFormat = .text;
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

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--external-quantize")) {
            if (i < args.len) {
                external_quantize = std.mem.sliceTo(args[i], 0);
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
                dataset_format = common.parseDatasetFormat(std.mem.sliceTo(args[i], 0));
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
    const model_config = abi.ai.training.trainable_model.TrainableModelConfig{
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
    utils.output.printHeader("New Transformer Model Configuration");
    utils.output.println("Architecture:", .{});
    utils.output.printKeyValueFmt("Hidden dim", "{d}", .{hidden_dim});
    utils.output.printKeyValueFmt("Num layers", "{d}", .{num_layers});
    utils.output.printKeyValueFmt("Num heads", "{d}", .{num_heads});
    utils.output.printKeyValueFmt("Intermediate dim", "{d}", .{intermediate_dim});
    utils.output.printKeyValueFmt("Vocab size", "{d}", .{vocab_size});
    utils.output.printKeyValueFmt("Max seq len", "{d}", .{max_seq_len});
    utils.output.printKeyValueFmt("Parameters", "{d} ({d:.2} MB)", .{
        num_params,
        @as(f64, @floatFromInt(num_params * 4)) / (1024 * 1024),
    });
    utils.output.println("", .{});
    utils.output.println("Training:", .{});
    utils.output.printKeyValueFmt("Epochs", "{d}", .{epochs});
    utils.output.printKeyValueFmt("Batch size", "{d}", .{batch_size});
    utils.output.printKeyValueFmt("Learning rate", "{e:.2}", .{learning_rate});
    utils.output.printKeyValueFmt("Warmup steps", "{d}", .{warmup_steps});
    utils.output.printKeyValueFmt("Weight decay", "{d:.4}", .{weight_decay});
    utils.output.printKeyValueFmt("Gradient clip", "{d:.2}", .{gradient_clip});
    utils.output.printKeyValueFmt("Grad accumulation", "{d}", .{grad_accum_steps});
    if (dataset_path) |path| {
        utils.output.printKeyValueFmt("Dataset", "{s}", .{path});
        utils.output.printKeyValueFmt("Dataset format", "{t}", .{dataset_format});
    } else {
        utils.output.printKeyValue("Dataset", "(synthetic)");
    }
    if (export_gguf_path) |path| {
        utils.output.printKeyValueFmt("Export GGUF", "{s}", .{path});
    }
    utils.output.println("", .{});

    // Create model from scratch
    utils.output.println("Initializing model with random weights...", .{});
    var model = abi.ai.training.TrainableModel.init(allocator, model_config) catch |err| {
        utils.output.printError("initializing model: {t}", .{err});
        return;
    };
    defer model.deinit();

    utils.output.println("Model initialized: {d} parameters", .{num_params});
    utils.output.println("", .{});

    // Prepare training tokens
    var train_tokens: []u32 = &.{};
    defer if (train_tokens.len > 0) allocator.free(train_tokens);

    if (dataset_path) |path| {
        // Load dataset if provided
        utils.output.println("Loading dataset from {s}...", .{path});

        // For text format without tokenizer, we need to create synthetic tokens
        // based on byte values (simple character-level encoding)
        if (dataset_format == .text) {
            const text = common.readTextFile(allocator, path) catch |err| {
                utils.output.printError("reading dataset: {t}", .{err});
                return;
            };
            defer allocator.free(text);

            // Simple byte-level tokenization
            train_tokens = allocator.alloc(u32, text.len) catch |err| {
                utils.output.printError("allocating tokens: {t}", .{err});
                return;
            };
            for (text, 0..) |byte, idx| {
                train_tokens[idx] = @as(u32, byte) % vocab_size;
            }

            if (dataset_max_tokens > 0 and train_tokens.len > dataset_max_tokens) {
                const trimmed = allocator.alloc(u32, dataset_max_tokens) catch |err| {
                    utils.output.printError("trimming tokens: {t}", .{err});
                    return;
                };
                @memcpy(trimmed, train_tokens[0..dataset_max_tokens]);
                allocator.free(train_tokens);
                train_tokens = trimmed;
            }
        } else if (dataset_format == .tokenbin) {
            train_tokens = abi.ai.database.readTokenBinFile(allocator, path) catch |err| {
                utils.output.printError("reading tokenbin: {t}", .{err});
                return;
            };
            if (dataset_max_tokens > 0 and train_tokens.len > dataset_max_tokens) {
                const trimmed = allocator.alloc(u32, dataset_max_tokens) catch |err| {
                    utils.output.printError("trimming tokens: {t}", .{err});
                    return;
                };
                @memcpy(trimmed, train_tokens[0..dataset_max_tokens]);
                allocator.free(train_tokens);
                train_tokens = trimmed;
            }
        } else {
            utils.output.printError("JSONL format requires a tokenizer (use --dataset-format text or tokenbin)", .{});
            return;
        }

        utils.output.println("Loaded {d} tokens", .{train_tokens.len});
        utils.output.println("", .{});
    } else {
        // Generate synthetic training data
        const num_tokens = @as(usize, max_seq_len) * batch_size * 10;
        train_tokens = allocator.alloc(u32, num_tokens) catch |err| {
            utils.output.printError("allocating synthetic tokens: {t}", .{err});
            return;
        };

        var rng = std.Random.DefaultPrng.init(42);
        for (train_tokens) |*t| {
            t.* = rng.random().intRangeLessThan(u32, 0, vocab_size);
        }
        utils.output.println("Generated {d} synthetic tokens for training", .{num_tokens});
        utils.output.println("", .{});
    }

    // Clamp tokens to vocab size
    common.clampTokens(train_tokens, vocab_size);

    // Create LLM training config
    var llm_config = abi.ai.training.LlmTrainingConfig{
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

    const export_path = export_gguf_path;
    const export_name = export_gguf_name orelse "abi-llama";

    // Export after training so we can attach a byte-level tokenizer.
    if (export_path != null) {
        llm_config.export_gguf_path = null;
    }
    llm_config.export_name = export_name;

    utils.output.println("Starting training from scratch...", .{});
    var timer = abi.shared.time.Timer.start() catch {
        utils.output.printError("Failed to start timer", .{});
        return;
    };

    const report = abi.ai.training.llm_trainer.trainLlm(allocator, &model, llm_config, train_tokens) catch |err| {
        utils.output.printError("Training failed: {t}", .{err});
        return;
    };

    const elapsed_ns = timer.read();
    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;

    utils.output.printHeader("Training Complete");
    utils.output.printKeyValueFmt("Final loss", "{d:.6}", .{report.final_loss});
    utils.output.printKeyValueFmt("Final accuracy", "{d:.2}%", .{report.final_accuracy * 100});
    utils.output.printKeyValueFmt("Total steps", "{d}", .{report.total_steps});
    utils.output.printKeyValueFmt("Tokens processed", "{d}", .{report.total_tokens});
    utils.output.printKeyValueFmt("Wall time", "{d:.2}s", .{elapsed_s});
    utils.output.printKeyValueFmt("Checkpoints saved", "{d}", .{report.checkpoints_saved});
    if (export_path) |path| {
        var tokenizer_build = ByteTokenizerBuild.init(allocator, vocab_size) catch |err| {
            utils.output.printError("preparing tokenizer export: {t}", .{err});
            return;
        };
        defer tokenizer_build.deinit(allocator);

        model.exportToGguf(allocator, path, .{
            .name = export_name,
            .tokenizer = tokenizer_build.toConfig(),
        }) catch |err| {
            utils.output.printError("exporting GGUF: {t}", .{err});
            return;
        };
        utils.output.printSuccess("Model exported to: {s}", .{path});

        // External quantization via llama-quantize
        if (external_quantize) |quant_type| {
            runExternalQuantize(path, quant_type);
        }
    }
}

pub fn runExternalQuantize(gguf_path: []const u8, quant_type: []const u8) void {
    const alloc = std.heap.page_allocator;
    utils.output.println("", .{});
    utils.output.println("External quantization: {s}", .{quant_type});

    // Initialize I/O backend for subprocess
    var io_backend = cli_io.initIoBackend(alloc);
    defer io_backend.deinit();
    const io = io_backend.io();

    // Check if llama-quantize is available
    const which_result = std.process.run(alloc, io, .{
        .argv = &.{ "which", "llama-quantize" },
    }) catch {
        printQuantizeHelp();
        return;
    };
    defer {
        alloc.free(which_result.stdout);
        alloc.free(which_result.stderr);
    }

    if (which_result.term != .exited or which_result.term.exited != 0) {
        printQuantizeHelp();
        return;
    }

    // Build output path: input.gguf -> input-Q4_K_M.gguf
    const dot_pos = std.mem.lastIndexOfScalar(u8, gguf_path, '.') orelse gguf_path.len;
    var out_buf: [512]u8 = undefined;
    const out_path = std.fmt.bufPrint(&out_buf, "{s}-{s}.gguf", .{
        gguf_path[0..dot_pos],
        quant_type,
    }) catch {
        utils.output.printError("output path too long", .{});
        return;
    };

    utils.output.println("Running: llama-quantize {s} {s} {s}", .{ gguf_path, out_path, quant_type });

    const result = std.process.run(alloc, io, .{
        .argv = &.{ "llama-quantize", gguf_path, out_path, quant_type },
    }) catch |err| {
        utils.output.printError("running llama-quantize: {t}", .{err});
        return;
    };
    defer {
        alloc.free(result.stdout);
        alloc.free(result.stderr);
    }

    if (result.term == .exited and result.term.exited == 0) {
        utils.output.printSuccess("Quantized model saved to: {s}", .{out_path});
    } else {
        utils.output.printError("llama-quantize failed", .{});
        if (result.stderr.len > 0) {
            utils.output.println("{s}", .{result.stderr});
        }
    }
}

pub fn printQuantizeHelp() void {
    utils.output.print(
        \\llama-quantize not found in PATH.
        \\
        \\To install llama.cpp tools:
        \\  macOS:  brew install llama.cpp
        \\  Linux:  git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make
        \\
        \\Common quantization types: q4_0, q4_k_m, q5_k_m, q8_0
        \\
    , .{});
}

pub fn printNewHelp() void {
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
        \\  --external-quantize <type>  Quantize via llama-quantize after export
        \\
        \\Examples:
        \\  abi train new --epochs 5
        \\  abi train new --hidden-dim 512 --num-layers 8 --num-heads 8
        \\  abi train new --dataset-path data.txt --epochs 10
        \\  abi train new --export-gguf model.gguf --export-name my-model
        \\  abi train new --export-gguf model.gguf --external-quantize q4_k_m
        \\
    ;
    utils.output.print("{s}", .{help_text});
}

test {
    std.testing.refAllDecls(@This());
}
