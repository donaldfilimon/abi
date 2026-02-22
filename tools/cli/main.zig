//! CLI entrypoint for ABI Framework
//!
//! Provides comprehensive CLI for database, GPU, AI, network operations.
//!
//! Usage:
//!   abi <command> [options]
//!
//! Commands:
//!   help, --help     Show help message
//!   version          Show framework version
//!   system-info      Show system and framework status
//!   db               Database operations (add, query, stats, optimize, backup)
//!   agent            Run AI agent (interactive or one-shot)
//!   bench            Run performance benchmarks (all, simd, memory, ai, quick)
//!   gpu              GPU commands (backends, devices, summary, default)
//!   gpu-dashboard    Interactive GPU + Agent monitoring dashboard
//!   network          Manage network registry (list, register, status)
//!   multi-agent      Run multi-agent workflows
//!   explore          Search and explore codebase
//!   simd             Run SIMD performance demo
//!   config           Configuration management (init, show, validate)
//!   discord          Discord bot operations (status, guilds, send, commands)
//!   llm              LLM inference (info, generate, chat, bench, download)
//!   model            Model management (list, download, remove, search)
//!   embed            Generate embeddings (openai, mistral, cohere, ollama)
//!   train            Training pipeline (run, resume, info)
//!   convert          Dataset conversion tools (tokenbin, text, jsonl, wdbx)
//!   task             Task management (add, list, done, stats)
//!   tui              Launch interactive TUI command menu
//!   plugins          Plugin management (list, enable, disable, info)
//!   profile          User profile and settings management
//!   completions      Shell completions (bash, zsh, fish, powershell)
//!   status           Show framework health and component status
//!   toolchain        Zig/ZLS toolchain (install, update, status)
//!   ralph            Ralph orchestrator (init, run, status, gate, improve, skills)

const std = @import("std");
const cli = @import("cli");

pub fn main(init: std.process.Init.Minimal) void {
    cli.mainWithArgs(init.args, init.environ) catch |err| {
        switch (err) {
            error.InvalidArgument => std.process.exit(1),
            else => {
                std.debug.print("error: {t}\n", .{err});
                std.process.exit(1);
            },
        }
    };
}
