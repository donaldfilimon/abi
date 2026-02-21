const std = @import("std");

/// All CLI smoke-test command vectors.
///
/// Each entry is a slice of argument strings passed to the CLI executable.
/// The build step runs every entry and expects a zero exit code.
pub const cli_commands = [_][]const []const u8{
    // Top-level flags
    &.{"--help"},
    &.{"--version"},

    // Direct subcommands
    &.{"system-info"},
    &.{ "db", "stats" },
    &.{ "gpu", "status" },
    &.{ "gpu", "backends" },
    &.{ "gpu", "devices" },
    &.{ "gpu", "summary" },
    &.{ "gpu", "default" },
    &.{ "task", "list" },
    &.{ "task", "stats" },
    &.{ "config", "show" },
    &.{ "config", "validate" },

    // Help for every top-level command
    &.{ "help", "llm" },
    &.{ "help", "gpu" },
    &.{ "help", "db" },
    &.{ "help", "train" },
    &.{ "help", "model" },
    &.{ "help", "config" },
    &.{ "help", "task" },
    &.{ "help", "network" },
    &.{ "help", "discord" },
    &.{ "help", "bench" },
    &.{ "help", "plugins" },
    &.{ "help", "completions" },
    &.{ "help", "multi-agent" },
    &.{ "help", "profile" },
    &.{ "help", "convert" },
    &.{ "help", "embed" },
    &.{ "help", "toolchain" },
    &.{ "help", "explore" },
    &.{ "help", "simd" },
    &.{ "help", "agent" },
    &.{ "help", "status" },
    &.{ "help", "mcp" },
    &.{ "help", "acp" },
    &.{ "help", "gpu-dashboard" },

    // Nested help (subcommand-level)
    &.{ "help", "llm", "generate" },
    &.{ "help", "llm", "chat" },
    &.{ "help", "train", "run" },
    &.{ "help", "train", "llm" },
    &.{ "help", "db", "add" },
    &.{ "help", "bench", "simd" },
    &.{ "help", "discord", "commands" },
    &.{ "help", "ralph" },
    &.{ "help", "gendocs" },
    &.{ "help", "db", "query" },
    &.{ "help", "db", "serve" },
    &.{ "help", "db", "backup" },
    &.{ "help", "db", "restore" },
    &.{ "help", "db", "optimize" },
    &.{ "help", "task", "add" },
    &.{ "help", "task", "edit" },
    &.{ "help", "ralph", "run" },
    &.{ "help", "ralph", "super" },
    &.{ "help", "ralph", "multi" },
    &.{ "help", "ralph", "skills" },
    &.{ "help", "convert", "dataset" },
    &.{ "help", "convert", "model" },
    &.{ "help", "convert", "embeddings" },
    &.{ "help", "discord", "info" },
    &.{ "help", "discord", "guilds" },
    &.{ "help", "plugins", "enable" },
    &.{ "help", "plugins", "disable" },

    // Functional subcommands
    &.{ "llm", "info" },
    &.{ "llm", "list" },
    &.{ "llm", "demo" },
    &.{ "llm", "demo", "--prompt", "Hello" },
    &.{ "llm", "list-local" },
    &.{ "llm", "serve", "--help" },
    &.{ "llm", "bench", "--help" },
    &.{ "llm", "download", "--help" },
    &.{ "train", "info" },
    &.{ "train", "auto" },
    &.{ "train", "auto", "--help" },
    &.{ "train", "run", "--help" },
    &.{ "train", "new", "--help" },
    &.{ "train", "llm", "--help" },
    &.{ "train", "vision", "--help" },
    &.{ "train", "clip", "--help" },
    &.{ "train", "resume", "--help" },
    &.{ "train", "monitor", "--help" },
    &.{ "train", "generate-data", "--help" },
    &.{ "model", "list" },
    &.{ "model", "path" },
    &.{ "model", "info", "--help" },
    &.{ "network", "list" },
    &.{ "network", "status" },
    &.{ "discord", "status" },
    &.{ "discord", "commands", "list" },
    &.{ "plugins", "list" },
    &.{ "plugins", "info", "openai-connector" },
    &.{ "plugins", "search" },
    &.{"bench"},
    &.{ "bench", "list" },
    &.{ "bench", "quick" },
    &.{ "bench", "simd" },
    &.{ "bench", "micro", "hash" },
    &.{ "bench", "micro", "alloc" },
    &.{ "bench", "micro", "noop" },
    &.{ "bench", "micro", "parse" },
    &.{ "completions", "bash" },
    &.{ "completions", "zsh" },
    &.{ "completions", "fish" },
    &.{ "completions", "powershell" },
    &.{ "multi-agent", "info" },
    &.{ "multi-agent", "list" },
    &.{ "multi-agent", "status" },
    &.{ "multi-agent", "run", "--help" },
    &.{ "multi-agent", "create", "--help" },
    &.{ "toolchain", "status" },
    &.{ "toolchain", "path" },
    &.{ "mcp", "tools" },
    &.{ "acp", "card" },
    &.{ "acp", "serve", "--help" },
    &.{"version"},
    &.{"status"},
    &.{ "ralph", "help" },
    &.{ "ralph", "status" },
    &.{ "ralph", "skills" },
    &.{ "ralph", "gate", "--help" },
    &.{"gendocs"},
    &.{ "profile", "show" },
    &.{ "profile", "list" },
    &.{ "gpu", "list" },
    &.{ "agent", "--help" },
    &.{ "tui", "--help" },
    &.{ "embed", "--help" },
    &.{ "explore", "--help" },

    // DB subcommands with --help
    &.{ "db", "add", "--help" },
    &.{ "db", "query", "--help" },
    &.{ "db", "optimize" },
    &.{ "db", "backup", "--help" },
    &.{ "db", "restore", "--help" },
    &.{ "db", "serve", "--help" },

    // Config init
    &.{ "config", "init", "--help" },

    // Aliases
    &.{"info"},
    &.{ "ls", "stats" },
};

/// Register CLI smoke tests as a build step.
///
/// Creates a "cli-tests" step that runs every command vector in
/// `cli_commands` against the provided CLI executable and asserts a zero
/// exit code.
pub fn addCliTests(b: *std.Build, exe: *std.Build.Step.Compile) *std.Build.Step {
    const step = b.step("cli-tests", "Run smoke test of CLI commands");
    for (&cli_commands) |args| {
        const run_cmd = b.addRunArtifact(exe);
        run_cmd.addArgs(args);
        step.dependOn(&run_cmd.step);
    }
    return step;
}
