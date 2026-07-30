//! `ai-plugin` — real implementation.
//!
//! Ported from `src/plugins/ai-plugin/mod.zig`. Every output string is contract:
//! `plugin_run` over MCP and `abi plugin run` both surface it verbatim.

use crate::{Plugin, PluginError, split_command};

/// The enabled `ai-plugin` implementation.
pub struct Mod;

impl Plugin for Mod {
    const NAME: &'static str = "ai-plugin";
    const VERSION: &'static str = "0.1.0";
    const DESCRIPTION: &'static str = "Example reference plugin targeting the feat-ai gate.";
    const TARGET_FEATURE: &'static str = "ai";

    fn run(input: &str) -> Result<String, PluginError> {
        if let Some(rest) = input.strip_prefix("__cmd__:") {
            let (cmd, _payload) = split_command(rest);
            if cmd == "profile" || cmd == "ai-status" {
                return Ok(
                    "ai-plugin: profile status (reference plugin — see `abi backends`)".to_string(),
                );
            }
            if cmd == "models" {
                return Ok(
                    "ai-plugin: model list (reference plugin — see `abi complete --help`)"
                        .to_string(),
                );
            }
            return Ok(format!("ai-plugin: unknown command '{cmd}'"));
        }

        Ok(format!("ai-plugin event (bytes={})", input.len()))
    }
}
