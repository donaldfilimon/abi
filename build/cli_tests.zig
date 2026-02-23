const std = @import("std");

/// All CLI smoke-test command vectors.
///
/// Each entry is a slice of argument strings passed to the CLI executable.
/// The build step runs every entry and expects a zero exit code.
pub const cli_commands = [_][]const []const u8{
    // ── Top-level flags ─────────────────────────────────────────────────
    &.{"--help"},
    &.{"--version"},

    // ── Direct subcommands ──────────────────────────────────────────────
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

    // ── Help for every top-level command ─────────────────────────────────
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
    &.{ "help", "ui" },

    // ── Nested help (subcommand-level) ──────────────────────────────────
    &.{ "help", "llm", "run" },
    &.{ "help", "llm", "session" },
    &.{ "help", "llm", "providers" },
    &.{ "help", "llm", "plugins" },
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

    // ── Functional subcommands ──────────────────────────────────────────
    &.{ "llm", "run", "--help" },
    &.{ "llm", "session", "--help" },
    &.{ "llm", "providers" },
    &.{ "llm", "plugins", "list" },
    &.{ "llm", "serve", "--help" },
    &.{ "ui", "launch", "--help" },
    &.{ "ui", "gpu", "--help" },
    &.{ "ui", "train", "--help" },
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
    &.{ "train", "run", "--epochs", "1", "--batch-size", "4", "--sample-count", "16" },
    &.{ "train", "self", "--help" },
    &.{ "train", "help" },
    &.{ "bench", "compare-training" },
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
    &.{ "ui", "--help" },
    &.{ "embed", "--help" },
    &.{ "explore", "--help" },

    // ── DB subcommands with --help ──────────────────────────────────────
    &.{ "db", "add", "--help" },
    &.{ "db", "query", "--help" },
    &.{ "db", "optimize" },
    &.{ "db", "backup", "--help" },
    &.{ "db", "restore", "--help" },
    &.{ "db", "serve", "--help" },

    // ── Config init ─────────────────────────────────────────────────────
    &.{ "config", "init", "--help" },

    // ── Aliases ─────────────────────────────────────────────────────────
    &.{"info"},
    &.{ "ls", "stats" },
};

/// Options for the exhaustive CLI integration test runner.
pub const CliTestsFullOptions = struct {
    env_file: ?[]const u8 = null,
    allow_blocked: bool = false,
    id_prefixes: ?[]const []const u8 = null,
    timeout_scale: f64 = 1.0,
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
        run_cmd.setCwd(b.path("."));
        run_cmd.addArgs(args);
        step.dependOn(&run_cmd.step);
    }
    return step;
}

/// Register the exhaustive CLI behavioral verification step.
///
/// Invokes `tools/scripts/run_cli_full_matrix.py` which handles preflight,
/// isolation, PTY probing, and full command-tree coverage.
pub fn addCliTestsFull(b: *std.Build, options: CliTestsFullOptions) *std.Build.Step {
    const step = b.step("cli-tests-full", "Run exhaustive behavioral CLI command-tree tests");

    const matrix_gen = b.addSystemCommand(&.{
        "zig", "run", "tools/cli/full_matrix_main.zig", "--", "--json-out", b.pathFromRoot(".zig-cache/matrix.json")
    });
    
    const run_full = b.addSystemCommand(&.{
        "zig", "run", "tools/cli/tests/runner.zig", "--", "--matrix", b.pathFromRoot(".zig-cache/matrix.json"), "--bin", b.pathFromRoot("zig-out/bin/abi")
    });
    run_full.step.dependOn(&matrix_gen.step);
    run_full.setCwd(b.path("."));
    if (options.env_file) |env_file|
        run_full.addArgs(&.{ "--env-file", env_file });
    if (options.id_prefixes) |id_prefixes| {
        for (id_prefixes) |prefix| {
            run_full.addArgs(&.{ "--id-prefix", prefix });
        }
    }
    if (options.allow_blocked)
        run_full.addArg("--allow-blocked");
    step.dependOn(&run_full.step);
    return step;
}

/// Register a focused nested-command CLI verification step.
///
/// Runs only explicit `nested.*` vectors from the full matrix.
pub fn addCliTestsNested(b: *std.Build, options: CliTestsFullOptions) *std.Build.Step {
    const step = b.step("cli-tests-nested", "Run nested CLI command-tree behavioral tests");
    const run_nested = b.addSystemCommand(&.{
        "python3",
        "tools/scripts/run_cli_full_matrix.py",
        "--repo",
        b.pathFromRoot("."),
        "--id-prefix",
        "nested.",
        "--timeout-scale",
        b.fmt("{d}", .{options.timeout_scale}),
    });
    run_nested.setCwd(b.path("."));
    if (options.env_file) |env_file|
        run_nested.addArgs(&.{ "--env-file", env_file });
    if (options.allow_blocked)
        run_nested.addArg("--allow-blocked");
    step.dependOn(&run_nested.step);
    return step;
}
