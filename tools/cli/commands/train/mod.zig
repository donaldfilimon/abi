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

// Subcommand dispatch (mirrors ralph.zig pattern)

fn tRun(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    try run_train.runTrain(allocator, parser.remaining());
}
fn tNew(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    try new_model.runNewModel(allocator, parser.remaining());
}
fn tLlm(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    try llm_train.runLlmTrain(allocator, parser.remaining());
}
fn tVision(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    try vision.runVisionTrain(allocator, parser.remaining());
}
fn tClip(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    try vision.runClipTrain(allocator, parser.remaining());
}
fn tAuto(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    try auto.runAutoTrain(allocator, parser.remaining());
}
fn tSelf(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    try self_train.runSelfTrain(allocator, parser.remaining());
}
fn tResume(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    try monitor.runResume(allocator, parser.remaining());
}
fn tMonitor(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    try monitor.runMonitor(allocator, parser.remaining());
}
fn tInfo(_: std.mem.Allocator, _: *utils.args.ArgParser) !void {
    info.runInfo();
}
fn tGenerateData(_: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    try data.runGenerateData(parser.remaining());
}
pub fn trainUnknown(cmd: []const u8) void {
    std.debug.print("Unknown train command: {s}\n", .{cmd});
}
pub fn printHelpAlloc(_: std.mem.Allocator) void {
    printHelp();
}

const train_commands = [_]utils.subcommand.Command{
    .{ .names = &.{"run"}, .run = tRun },
    .{ .names = &.{"new"}, .run = tNew },
    .{ .names = &.{"llm"}, .run = tLlm },
    .{ .names = &.{"vision"}, .run = tVision },
    .{ .names = &.{"clip"}, .run = tClip },
    .{ .names = &.{"auto"}, .run = tAuto },
    .{ .names = &.{"self"}, .run = tSelf },
    .{ .names = &.{"resume"}, .run = tResume },
    .{ .names = &.{"monitor"}, .run = tMonitor },
    .{ .names = &.{"info"}, .run = tInfo },
    .{ .names = &.{"generate-data"}, .run = tGenerateData },
};

/// Run the train command with the provided arguments.
pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var parser = utils.args.ArgParser.init(allocator, args);
    try utils.subcommand.runSubcommand(
        allocator,
        &parser,
        &train_commands,
        null,
        printHelpAlloc,
        trainUnknown,
    );
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
    std.debug.print("{s}", .{help_text});
}
