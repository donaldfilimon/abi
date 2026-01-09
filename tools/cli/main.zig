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
//!   info             Show framework information
//!   db               Database operations (init, add, query, stats)
//!   gpu              GPU commands (backends, devices, summary)
//!   network          Network management (list, register, status)
//!   agent            AI agent operations
//!   explore          Code exploration

pub fn main() !void {
    return @import("cli").main();
}
