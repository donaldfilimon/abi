const types = @import("../framework/types.zig");

pub const CommandOverride = struct {
    name: []const u8,
    source_id: ?[]const u8 = null,
    default_subcommand: ?[:0]const u8 = null,
    visibility: ?types.Visibility = null,
    risk: ?types.RiskLevel = null,
    ui: ?types.UiMeta = null,
    options: ?[]const types.OptionInfo = null,
    middleware_tags: ?[]const []const u8 = null,
};

const llm_options = [_]types.OptionInfo{
    .{ .flag = "--model", .description = "Model path or name" },
    .{ .flag = "--prompt", .description = "Input prompt text" },
    .{ .flag = "--backend", .description = "Provider backend (ollama, mlx, etc.)" },
    .{ .flag = "--fallback", .description = "Comma-separated fallback providers" },
    .{ .flag = "--temperature", .description = "Sampling temperature" },
    .{ .flag = "--max-tokens", .description = "Maximum tokens to generate" },
    .{ .flag = "--stream", .description = "Enable streaming output" },
    .{ .flag = "--json", .description = "Output in JSON format" },
};

const train_options = [_]types.OptionInfo{
    .{ .flag = "--epochs", .description = "Number of training epochs" },
    .{ .flag = "--batch-size", .description = "Training batch size" },
    .{ .flag = "--learning-rate", .description = "Learning rate" },
    .{ .flag = "--optimizer", .description = "Optimizer (sgd, adam, adamw)" },
    .{ .flag = "--checkpoint-path", .description = "Checkpoint save path" },
    .{ .flag = "--checkpoint-interval", .description = "Steps between checkpoints" },
    .{ .flag = "--mixed-precision", .description = "Enable mixed precision training" },
    .{ .flag = "--use-gpu", .description = "Enable GPU acceleration" },
    .{ .flag = "--cpu-only", .description = "Force CPU-only training" },
};

const model_options = [_]types.OptionInfo{
    .{ .flag = "--json", .description = "Output in JSON format" },
    .{ .flag = "--no-size", .description = "Hide file sizes in list" },
    .{ .flag = "--output", .description = "Output file path" },
    .{ .flag = "--no-verify", .description = "Skip checksum verification" },
    .{ .flag = "--force", .description = "Force removal without confirmation" },
    .{ .flag = "--reset", .description = "Reset cache directory to default" },
};

const ui_options = [_]types.OptionInfo{
    .{ .flag = "--theme", .description = "Set TUI color theme" },
    .{ .flag = "--list-themes", .description = "List available themes" },
    .{ .flag = "--refresh-ms", .description = "Dashboard refresh interval in ms" },
    .{ .flag = "--layers", .description = "Neural network layer sizes (comma-separated)" },
    .{ .flag = "--frames", .description = "Number of animation frames (0=infinite)" },
};

const ralph_options = [_]types.OptionInfo{
    .{ .flag = "--task", .description = "Task description for inline execution" },
    .{ .flag = "--gate", .description = "Run quality gate after execution" },
    .{ .flag = "--auto-skill", .description = "Auto-extract skill after run" },
    .{ .flag = "--iterations", .description = "Number of loop iterations" },
    .{ .flag = "--config", .description = "Path to ralph.yml config" },
};

const bench_options = [_]types.OptionInfo{
    .{ .flag = "--json", .description = "Output results in JSON format" },
    .{ .flag = "--output", .description = "Write results to file" },
};

const db_options = [_]types.OptionInfo{
    .{ .flag = "--db", .description = "Database file path" },
    .{ .flag = "--id", .description = "Record ID" },
    .{ .flag = "--embed", .description = "Embedding text" },
    .{ .flag = "--top-k", .description = "Number of results to return" },
    .{ .flag = "--out", .description = "Backup output path" },
    .{ .flag = "--in", .description = "Restore input path" },
    .{ .flag = "--path", .description = "Legacy path shorthand" },
};

const profile_options = [_]types.OptionInfo{
    .{ .flag = "--json", .description = "Output in JSON format" },
    .{ .flag = "--force", .description = "Force delete without confirmation" },
};

const lsp_options = [_]types.OptionInfo{
    .{ .flag = "--path", .description = "Source file path" },
    .{ .flag = "--line", .description = "Line number (0-based)" },
    .{ .flag = "--character", .description = "Character offset (0-based)" },
    .{ .flag = "--method", .description = "LSP method name" },
    .{ .flag = "--params", .description = "JSON parameters" },
    .{ .flag = "--new-name", .description = "New name for rename" },
};

const mcp_options = [_]types.OptionInfo{
    .{ .flag = "--db", .description = "Database path for MCP server" },
    .{ .flag = "--transport", .description = "Transport type (stdio)" },
};

const network_options = [_]types.OptionInfo{
    .{ .flag = "--cluster-id", .description = "Cluster identifier" },
    .{ .flag = "--address", .description = "Node address (host:port)" },
};

pub const command_overrides = [_]CommandOverride{
    .{ .name = "llm", .default_subcommand = "providers", .options = &llm_options, .ui = .{ .category = .ai, .shortcut = 2 } },
    .{ .name = "train", .default_subcommand = "info", .options = &train_options, .ui = .{ .category = .ai, .shortcut = 3 } },
    .{ .name = "model", .default_subcommand = "list", .options = &model_options, .ui = .{ .category = .ai } },
    .{ .name = "ui", .default_subcommand = "launch", .options = &ui_options, .ui = .{ .category = .ai } },
    .{ .name = "ralph", .default_subcommand = "status", .options = &ralph_options, .ui = .{ .category = .ai } },
    .{ .name = "bench", .default_subcommand = "quick", .options = &bench_options, .ui = .{ .category = .tools, .shortcut = 8 } },
    .{ .name = "db", .default_subcommand = "stats", .options = &db_options, .ui = .{ .category = .data, .shortcut = 4 } },
    .{ .name = "profile", .default_subcommand = "show", .options = &profile_options },
    .{ .name = "lsp", .options = &lsp_options },
    .{ .name = "mcp", .options = &mcp_options },
    .{ .name = "network", .default_subcommand = "status", .options = &network_options, .ui = .{ .category = .system, .shortcut = 7 } },
    .{ .name = "gpu", .default_subcommand = "summary", .ui = .{ .category = .system, .shortcut = 6 } },
    .{ .name = "agent", .ui = .{ .category = .ai, .shortcut = 1 } },
    .{ .name = "explore", .ui = .{ .category = .data, .shortcut = 5 } },
    .{ .name = "simd", .ui = .{ .category = .tools, .shortcut = 9 } },
    .{ .name = "toolchain", .default_subcommand = "status", .risk = .caution, .ui = .{ .category = .system } },
    .{ .name = "task", .default_subcommand = "list", .ui = .{ .category = .data } },
    .{ .name = "completions", .visibility = .hidden, .ui = .{ .include_in_launcher = false } },
    .{ .name = "clean", .risk = .destructive, .ui = .{ .risk_badge = "destructive" } },
};
