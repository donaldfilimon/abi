//! Training pipeline CLI command.
//!
//! Subcommands:
//! - train run [options]           - Run basic training pipeline
//! - train new [options]           - Create and train a new transformer from scratch
//! - train llm <model.gguf> [opts] - Train LLM from GGUF model file
//! - train vision [options]        - Train Vision Transformer (ViT)
//! - train clip [options]          - Train CLIP multimodal model
//! - train auto [options]          - Auto-train with seed data
//! - train self [options]          - Self-improvement pipeline (auto + Ralph + viz)
//! - train resume <checkpoint>     - Resume training from checkpoint
//! - train monitor [run-id]        - Monitor training progress (TUI dashboard)
//! - train generate-data [options] - Generate synthetic tokenized data
//! - train info                    - Show default training configuration
//! - train help                    - Show help message

const std = @import("std");
const command_mod = @import("../../command.zig");
const context_mod = @import("../../framework/context.zig");
const utils = @import("../../utils/mod.zig");

const run_train = @import("run_train.zig");
const new_model = @import("new_model.zig");
const llm_train = @import("llm_train.zig");
const vision = @import("vision.zig");
const auto = @import("auto.zig");
const self_train = @import("self.zig");
const monitor = @import("monitor.zig");
const info = @import("info.zig");
const data = @import("data.zig");

pub const meta: command_mod.Meta = .{
    .name = "train",
    .description = "Training pipeline (run, llm, vision, auto, self, resume, info)",
    .subcommands = &.{ "run", "new", "llm", "vision", "clip", "auto", "self", "resume", "monitor", "info", "generate-data", "help" },
    .children = &.{
        .{ .name = "run", .description = "Run basic training pipeline", .handler = command_mod.contextParserHandler(tRun) },
        .{ .name = "new", .description = "Create and train a new transformer from scratch", .handler = command_mod.contextParserHandler(tNew) },
        .{ .name = "llm", .description = "Train LLM from GGUF model file", .handler = command_mod.contextParserHandler(tLlm) },
        .{ .name = "vision", .description = "Train Vision Transformer (ViT)", .handler = command_mod.contextParserHandler(tVision) },
        .{ .name = "clip", .description = "Train CLIP multimodal model", .handler = command_mod.contextParserHandler(tClip) },
        .{ .name = "auto", .description = "Auto-train with seed data", .handler = command_mod.contextParserHandler(tAuto) },
        .{ .name = "self", .description = "Self-improvement pipeline", .handler = command_mod.contextParserHandler(tSelf) },
        .{ .name = "resume", .description = "Resume training from checkpoint", .handler = command_mod.contextParserHandler(tResume) },
        .{ .name = "monitor", .description = "Monitor training progress (TUI dashboard)", .handler = command_mod.contextParserHandler(tMonitor) },
        .{ .name = "info", .description = "Show default training configuration", .handler = command_mod.contextParserHandler(tInfo) },
        .{ .name = "generate-data", .description = "Generate synthetic tokenized data", .handler = command_mod.contextParserHandler(tGenerateData) },
    },
};

// Subcommand dispatch (mirrors ralph.zig pattern)

fn tRun(ctx: *const context_mod.CommandContext, parser: *utils.args.ArgParser) !void {
    try run_train.runTrain(ctx, parser.remaining());
}
fn tNew(ctx: *const context_mod.CommandContext, parser: *utils.args.ArgParser) !void {
    try new_model.runNewModel(ctx, parser.remaining());
}
fn tLlm(ctx: *const context_mod.CommandContext, parser: *utils.args.ArgParser) !void {
    try llm_train.runLlmTrain(ctx, parser.remaining());
}
fn tVision(ctx: *const context_mod.CommandContext, parser: *utils.args.ArgParser) !void {
    try vision.runVisionTrain(ctx, parser.remaining());
}
fn tClip(ctx: *const context_mod.CommandContext, parser: *utils.args.ArgParser) !void {
    try vision.runClipTrain(ctx, parser.remaining());
}
fn tAuto(ctx: *const context_mod.CommandContext, parser: *utils.args.ArgParser) !void {
    try auto.runAutoTrain(ctx, parser.remaining());
}
fn tSelf(ctx: *const context_mod.CommandContext, parser: *utils.args.ArgParser) !void {
    try self_train.runSelfTrain(ctx, parser.remaining());
}
fn tResume(ctx: *const context_mod.CommandContext, parser: *utils.args.ArgParser) !void {
    try monitor.runResume(ctx, parser.remaining());
}
fn tMonitor(ctx: *const context_mod.CommandContext, parser: *utils.args.ArgParser) !void {
    try monitor.runMonitor(ctx, parser.remaining());
}
fn tInfo(_: *const context_mod.CommandContext, _: *utils.args.ArgParser) !void {
    info.runInfo();
}
fn tGenerateData(_: *const context_mod.CommandContext, parser: *utils.args.ArgParser) !void {
    try data.runGenerateData(parser.remaining());
}

/// Run the train command with the provided arguments.
/// Only reached when no child matches (help / unknown).
pub fn run(_: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    if (args.len == 0) {
        printHelp();
        return;
    }
    const cmd = std.mem.sliceTo(args[0], 0);
    if (utils.args.matchesAny(cmd, &.{ "--help", "-h", "help" })) {
        printHelp();
        return;
    }
    // Unknown subcommand
    utils.output.printError("Unknown train command: {s}", .{cmd});
    if (command_mod.suggestSubcommand(meta, cmd)) |suggestion| {
        utils.output.println("Did you mean: {s}", .{suggestion});
    }
}

pub fn printHelp() void {
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
        \\  self [options]          Run self-improvement pipeline (auto + Ralph + optional viz)
        \\  resume <checkpoint>     Resume training from a checkpoint file
        \\  monitor [run-id]        Monitor training progress (TUI dashboard)
        \\  generate-data [options]  Generate synthetic tokenized training data
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
        \\  --use-gpu                  Enable GPU acceleration (default behavior)
        \\  --cpu-only, --no-gpu       Disable GPU/NPU acceleration and force CPU
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
        \\  --external-quantize <type> Quantize GGUF via llama-quantize (q4_0, q4_k_m, q5_k_m, q8_0)
        \\
        \\Synthetic data options (for 'generate-data'):
        \\  --num-samples <n>          Number of sequences (default: 1024)
        \\  --seq-length <n>           Tokens per sequence (default: 128)
        \\  --vocab-size <n>           Max token ID (default: 32000)
        \\  --output <path>            Output file path (default: synthetic.bin)
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
        \\  abi train self --multimodal --iterations 7
        \\  abi train self --visualize --visualize-frames 0
        \\  abi train generate-data --num-samples 1024 --seq-length 128 --vocab-size 32000
        \\  abi train generate-data --output /tmp/test.bin --num-samples 100 --seq-length 32
        \\  abi train resume ./checkpoints/model.ckpt
        \\  abi train info
        \\
    ;
    utils.output.print("{s}", .{help_text});
}

test {
    std.testing.refAllDecls(@This());
}
