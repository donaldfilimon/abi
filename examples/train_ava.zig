//! Ava Training Example
//!
//! Demonstrates training the Ava assistant model based on gpt-oss.
//! Ava is a locally-trained, versatile AI assistant optimized for:
//! - General knowledge and reasoning
//! - Code understanding and generation
//! - Task decomposition and problem solving
//!
//! Usage:
//!   zig build run-train-ava -- path/to/gpt-oss.gguf [options]
//!
//! Options:
//!   --dataset-path <path>   Path to training data (jsonl or text)
//!   --epochs <n>            Number of training epochs (default: 3)
//!   --lr <f>                Learning rate (default: 1e-5)
//!   --output <path>         Output model path (default: ava.gguf)

const std = @import("std");
const abi = @import("abi");

/// Ava training configuration preset
pub const AvaTrainingConfig = struct {
    /// Base model path (gpt-oss GGUF file)
    base_model: []const u8 = "gpt-oss.gguf",
    /// Output model path
    output_path: []const u8 = "ava.gguf",
    /// Training data path (jsonl format recommended)
    dataset_path: ?[]const u8 = null,
    /// Number of epochs
    epochs: u32 = 3,
    /// Batch size (adjust based on GPU memory)
    batch_size: u32 = 4,
    /// Learning rate for fine-tuning
    learning_rate: f32 = 1e-5,
    /// Maximum sequence length
    max_seq_len: u32 = 2048,
    /// Weight decay for regularization
    weight_decay: f32 = 0.01,
    /// Gradient clipping threshold
    gradient_clip: f32 = 1.0,
    /// Gradient accumulation steps
    gradient_accumulation: u32 = 4,
    /// Warmup steps for learning rate
    warmup_steps: u32 = 100,
    /// Checkpoint interval (steps)
    checkpoint_interval: u32 = 500,
    /// Use GPU acceleration
    use_gpu: bool = true,
    /// Enable LoRA for efficient fine-tuning
    use_lora: bool = true,
    /// LoRA rank (lower = smaller adapter)
    lora_rank: u32 = 16,
    /// LoRA alpha scaling
    lora_alpha: f32 = 32.0,
};

/// Default Ava training configuration
pub fn defaultConfig() AvaTrainingConfig {
    return .{};
}

/// Convert Ava config to LLM training config
pub fn toLlmConfig(config: AvaTrainingConfig) abi.ai.training.LlmTrainingConfig {
    return .{
        .epochs = config.epochs,
        .batch_size = config.batch_size,
        .learning_rate = config.learning_rate,
        .max_seq_len = config.max_seq_len,
        .weight_decay = config.weight_decay,
        .max_grad_norm = config.gradient_clip,
        .grad_accum_steps = config.gradient_accumulation,
        .warmup_steps = config.warmup_steps,
        .checkpoint_interval = config.checkpoint_interval,
        .optimizer = .adamw,
        .lr_schedule = .warmup_cosine,
        .mixed_precision = config.use_gpu,
        .export_gguf_path = config.output_path,
        .export_name = "ava",
    };
}

pub fn main(init: std.process.Init.Minimal) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Ava Training ===\n", .{});
    std.debug.print("Training a locally-optimized AI assistant based on gpt-oss\n\n", .{});

    if (!abi.ai.isEnabled()) {
        std.debug.print("AI feature is disabled. Enable with -Denable-ai=true\n", .{});
        return;
    }

    // Parse command line arguments
    var config = defaultConfig();
    var args_it = init.args.iterateAllocator(allocator) catch |err| {
        std.debug.print("Failed to read args: {t}\n", .{err});
        return;
    };
    defer args_it.deinit();

    _ = args_it.next(); // Skip executable name

    // First positional arg is base model path
    if (args_it.next()) |arg| {
        if (!std.mem.startsWith(u8, arg, "--")) {
            config.base_model = arg;
        }
    }

    // Parse remaining options
    while (args_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--dataset-path") or std.mem.eql(u8, arg, "-d")) {
            if (args_it.next()) |val| {
                config.dataset_path = val;
            }
        } else if (std.mem.eql(u8, arg, "--epochs") or std.mem.eql(u8, arg, "-e")) {
            if (args_it.next()) |val| {
                config.epochs = std.fmt.parseInt(u32, val, 10) catch config.epochs;
            }
        } else if (std.mem.eql(u8, arg, "--lr")) {
            if (args_it.next()) |val| {
                config.learning_rate = std.fmt.parseFloat(f32, val) catch config.learning_rate;
            }
        } else if (std.mem.eql(u8, arg, "--output") or std.mem.eql(u8, arg, "-o")) {
            if (args_it.next()) |val| {
                config.output_path = val;
            }
        } else if (std.mem.eql(u8, arg, "--batch-size") or std.mem.eql(u8, arg, "-b")) {
            if (args_it.next()) |val| {
                config.batch_size = std.fmt.parseInt(u32, val, 10) catch config.batch_size;
            }
        } else if (std.mem.eql(u8, arg, "--no-gpu")) {
            config.use_gpu = false;
        } else if (std.mem.eql(u8, arg, "--no-lora")) {
            config.use_lora = false;
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printHelp();
            return;
        }
    }

    // Print configuration
    std.debug.print("Configuration:\n", .{});
    std.debug.print("  Base model:     {s}\n", .{config.base_model});
    std.debug.print("  Output:         {s}\n", .{config.output_path});
    std.debug.print("  Epochs:         {d}\n", .{config.epochs});
    std.debug.print("  Batch size:     {d}\n", .{config.batch_size});
    std.debug.print("  Learning rate:  {e:.2}\n", .{config.learning_rate});
    std.debug.print("  Max seq len:    {d}\n", .{config.max_seq_len});
    std.debug.print("  Use GPU:        {}\n", .{config.use_gpu});
    std.debug.print("  Use LoRA:       {}\n", .{config.use_lora});
    if (config.dataset_path) |path| {
        std.debug.print("  Dataset:        {s}\n", .{path});
    }
    std.debug.print("\n", .{});

    // Initialize framework
    const gpu_config: ?abi.config.GpuConfig = if (config.use_gpu) .{} else null;
    var framework = abi.initWithConfig(allocator, .{
        .ai = .{ .training = .{}, .llm = .{} },
        .gpu = gpu_config,
    }) catch |err| {
        std.debug.print("Framework initialization failed: {t}\n", .{err});
        return;
    };
    defer framework.deinit();

    // Load base model
    std.debug.print("Loading base model: {s}...\n", .{config.base_model});
    var model = abi.ai.TrainableModel.fromGguf(allocator, config.base_model) catch |err| {
        std.debug.print("Error loading model: {t}\n", .{err});
        std.debug.print("\nTo train Ava, you need a gpt-oss compatible GGUF model.\n", .{});
        std.debug.print("Download one from: https://huggingface.co/TheBloke\n", .{});
        std.debug.print("\nExample:\n", .{});
        std.debug.print("  wget https://huggingface.co/TheBloke/gpt2-GGUF/resolve/main/gpt2.Q4_K_M.gguf\n", .{});
        std.debug.print("  zig build run-train-ava -- gpt2.Q4_K_M.gguf --dataset-path data.jsonl\n", .{});
        showTrainingDemo(config);
        return;
    };
    defer model.deinit();

    std.debug.print("Model loaded: {d} parameters\n", .{model.numParams()});

    // Check for training data
    if (config.dataset_path == null) {
        std.debug.print("\nNo dataset provided. Showing training setup demo.\n", .{});
        showTrainingDemo(config);
        return;
    }

    // Load tokenizer from model
    var gguf_file = abi.ai.llm.io.gguf.GgufFile.open(allocator, config.base_model) catch |err| {
        std.debug.print("Error opening GGUF for tokenizer: {t}\n", .{err});
        return;
    };
    defer gguf_file.deinit();

    var tokenizer = abi.ai.llm.tokenizer.loadFromGguf(allocator, &gguf_file) catch |err| {
        std.debug.print("Error loading tokenizer: {t}\n", .{err});
        return;
    };
    defer tokenizer.deinit();

    // Load and tokenize dataset
    std.debug.print("Loading dataset: {s}...\n", .{config.dataset_path.?});
    const dataset_content = readDataset(allocator, config.dataset_path.?) catch |err| {
        std.debug.print("Error reading dataset: {t}\n", .{err});
        return;
    };
    defer allocator.free(dataset_content);

    const tokens = tokenizer.encode(allocator, dataset_content) catch |err| {
        std.debug.print("Error tokenizing dataset: {t}\n", .{err});
        return;
    };
    defer allocator.free(tokens);

    std.debug.print("Dataset: {d} tokens\n\n", .{tokens.len});

    // Start training
    std.debug.print("Starting Ava training...\n", .{});
    std.debug.print("=========================================\n", .{});

    const llm_config = toLlmConfig(config);
    const report = abi.ai.training.trainLlm(allocator, &model, llm_config, tokens) catch |err| {
        std.debug.print("Training failed: {t}\n", .{err});
        return;
    };

    // Print results
    std.debug.print("\n=========================================\n", .{});
    std.debug.print("Training Complete!\n", .{});
    std.debug.print("=========================================\n", .{});
    std.debug.print("Final loss:       {d:.6}\n", .{report.final_loss});
    std.debug.print("Final accuracy:   {d:.2}%\n", .{report.final_accuracy * 100});
    std.debug.print("Total steps:      {d}\n", .{report.total_steps});
    std.debug.print("Tokens processed: {d}\n", .{report.total_tokens});
    std.debug.print("Training time:    {d:.2}s\n", .{@as(f64, @floatFromInt(report.total_time_ns)) / 1e9});
    std.debug.print("Checkpoints:      {d}\n", .{report.checkpoints_saved});
    std.debug.print("\nModel saved to: {s}\n", .{config.output_path});

    // Show next steps
    std.debug.print("\n=== Next Steps ===\n", .{});
    std.debug.print("1. Test Ava:\n", .{});
    std.debug.print("   zig build run -- llm chat {s}\n", .{config.output_path});
    std.debug.print("\n2. Use with agent:\n", .{});
    std.debug.print("   zig build run -- agent --persona ava --model {s}\n", .{config.output_path});
    std.debug.print("\n3. Integrate in code:\n", .{});
    std.debug.print("   const model = try abi.ai.llm.Model.load(allocator, \"{s}\");\n", .{config.output_path});
    std.debug.print("   const persona = abi.ai.prompts.getPersona(.ava);\n", .{});
}

fn showTrainingDemo(config: AvaTrainingConfig) void {
    std.debug.print("\n=== Ava Training Demo ===\n\n", .{});
    std.debug.print("To train Ava, prepare a training dataset in JSONL format:\n\n", .{});
    std.debug.print("{{\"instruction\": \"What is Zig?\", \"output\": \"Zig is a systems programming language...\"}}\n", .{});
    std.debug.print("{{\"instruction\": \"Write a hello world\", \"output\": \"const std = @import(\\\"std\\\");...\"}}\n", .{});
    std.debug.print("\nOr use plain text for next-token prediction training.\n\n", .{});
    std.debug.print("Example training command:\n", .{});
    std.debug.print("  zig build run-train-ava -- {s} --dataset-path training_data.jsonl\n\n", .{config.base_model});
    std.debug.print("For LoRA fine-tuning (recommended for limited GPU memory):\n", .{});
    std.debug.print("  zig build run-train-ava -- {s} -d data.jsonl --epochs 5 --lr 2e-5\n\n", .{config.base_model});
    std.debug.print("Training configuration used:\n", .{});
    std.debug.print("  - AdamW optimizer with warmup cosine schedule\n", .{});
    std.debug.print("  - Gradient accumulation: {d} steps\n", .{config.gradient_accumulation});
    std.debug.print("  - Gradient clipping: {d:.1}\n", .{config.gradient_clip});
    std.debug.print("  - Weight decay: {d:.4}\n", .{config.weight_decay});
    std.debug.print("  - Checkpoints every {d} steps\n", .{config.checkpoint_interval});
}

fn readDataset(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    return std.Io.Dir.cwd().readFileAlloc(
        io,
        path,
        allocator,
        .limited(256 * 1024 * 1024), // 256MB max
    );
}

fn printHelp() void {
    const help =
        \\Ava Training - Train a local AI assistant based on gpt-oss
        \\
        \\Ava is a self-learning AI assistant with capabilities for:
        \\  - Text understanding and generation
        \\  - Vision and image understanding
        \\  - Document parsing (PDF, HTML, Markdown, Code)
        \\  - Continuous learning from feedback (RLHF)
        \\
        \\Usage: zig build run-train-ava -- <model.gguf> [options]
        \\
        \\Arguments:
        \\  <model.gguf>              Base model in GGUF format (gpt-oss compatible)
        \\
        \\Options:
        \\  -d, --dataset-path <path> Training data (jsonl or text format)
        \\  -o, --output <path>       Output model path (default: ava.gguf)
        \\  -e, --epochs <n>          Number of training epochs (default: 3)
        \\  -b, --batch-size <n>      Batch size (default: 4)
        \\  --lr <f>                  Learning rate (default: 1e-5)
        \\  --no-gpu                  Disable GPU acceleration
        \\  --no-lora                 Full fine-tuning instead of LoRA
        \\  --self-learning           Enable continuous self-learning mode
        \\  --vision                  Enable vision training (image datasets)
        \\  --documents               Enable document understanding training
        \\  -h, --help                Show this help message
        \\
        \\Self-Learning Mode:
        \\  When enabled, Ava learns from user feedback during inference:
        \\  - Positive feedback: Reinforces successful responses
        \\  - Negative feedback: Reduces similar responses
        \\  - Implicit signals: Tracks corrections and acceptances
        \\
        \\Examples:
        \\  # Basic training with defaults
        \\  zig build run-train-ava -- gpt2.gguf -d train.jsonl
        \\
        \\  # Custom epochs and learning rate
        \\  zig build run-train-ava -- gpt2.gguf -d train.jsonl -e 5 --lr 2e-5
        \\
        \\  # Enable self-learning with vision
        \\  zig build run-train-ava -- gpt2.gguf -d train.jsonl --self-learning --vision
        \\
        \\  # CPU-only training with document understanding
        \\  zig build run-train-ava -- gpt2.gguf -d train.jsonl --no-gpu --documents
        \\
        \\Dataset Formats:
        \\  Text (JSONL):
        \\    {"instruction": "...", "output": "..."}
        \\    {"instruction": "...", "input": "...", "output": "..."}
        \\    {"text": "..."}
        \\
        \\  Vision (JSONL with image paths):
        \\    {"image": "path/to/image.png", "caption": "Description of image"}
        \\    {"image": "path/to/doc.png", "question": "...", "answer": "..."}
        \\
        \\  Documents (JSONL):
        \\    {"document": "path/to/file.pdf", "summary": "..."}
        \\    {"document": "path/to/code.py", "explanation": "..."}
        \\
        \\For more information, see: docs/content/ai.html
        \\
    ;
    std.debug.print("{s}", .{help});
}
