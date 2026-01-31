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
//!   system-info      Show framework information
//!   db               Database operations (init, add, query, stats, backup)
//!   gpu              GPU commands (backends, devices, summary)
//!   network          Network management (list, register, status)
//!   agent            AI agent operations
//!   explore          Code exploration
//!   config           Configuration management
//!   bench            Benchmarks and micro-benchmarks
//!   llm              Local LLM commands (info, generate, chat, bench)
//!   embed            Embeddings commands
//!   train            Training pipeline commands
//!   discord          Discord integration commands
//!   simd             SIMD demo
//!   tui              Interactive terminal UI launcher

const std = @import("std");
const cli = @import("cli");

pub fn main(init: std.process.Init.Minimal) !void {
    return cli.mainWithArgs(init.args, init.environ);
}
