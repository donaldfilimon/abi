//! `example-plugin` — real implementation.
//!
//! Ported from `src/plugins/example-plugin/mod.zig`. Every output string is
//! contract: `plugin_run` over MCP and `abi plugin run` both surface it verbatim,
//! and `mcp-tool-calls-args.jsonl` pins the default branch byte-for-byte.

use crate::{Plugin, PluginError, split_command};

/// The enabled `example-plugin` implementation.
pub struct Mod;

impl Plugin for Mod {
    const NAME: &'static str = "example-plugin";
    const VERSION: &'static str = "0.1.0";
    const DESCRIPTION: &'static str = "Minimal example plugin used by registry generation tests.";
    const TARGET_FEATURE: &'static str = "plugins";

    fn run(input: &str) -> Result<String, PluginError> {
        if let Some(provider) = input.strip_prefix("__context__:") {
            if provider == "example-info" {
                return Ok("example-plugin v0.1.0 (target: plugins)".to_string());
            }
            return Ok(format!("unknown context provider: {provider}"));
        }

        if let Some(rest) = input.strip_prefix("__cmd__:") {
            let (cmd, payload) = split_command(rest);
            if cmd == "echo" || cmd == "say" {
                if payload.is_empty() {
                    return Ok("example-plugin echo: (empty)".to_string());
                }
                return Ok(format!("example-plugin echo: {payload}"));
            }
            return Ok(format!("example-plugin: unknown command '{cmd}'"));
        }

        Ok(format!("example-plugin received input (len={})", input.len()))
    }
}
