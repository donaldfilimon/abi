//! `nn-plugin` — real implementation.
//!
//! Ported from `src/plugins/nn-plugin/mod.zig`. Every output string is contract:
//! `plugin_run` over MCP and `abi plugin run` both surface it verbatim.

use crate::{Plugin, PluginError, split_command};

/// The enabled `nn-plugin` implementation.
pub struct Mod;

impl Plugin for Mod {
    const NAME: &'static str = "nn-plugin";
    const VERSION: &'static str = "0.1.0";
    const DESCRIPTION: &'static str = "Example reference plugin targeting the feat-nn gate.";
    const TARGET_FEATURE: &'static str = "nn";

    fn run(input: &str) -> Result<String, PluginError> {
        if let Some(rest) = input.strip_prefix("__cmd__:") {
            let (cmd, _payload) = split_command(rest);
            if cmd == "nn-train" {
                return Ok(
                    "nn-plugin: train (reference plugin — use `abi nn` / `abi train` for real demos)"
                        .to_string(),
                );
            }
            if cmd == "nn-sample" {
                return Ok(
                    "nn-plugin: sample (reference plugin — use `abi nn` for real demos)".to_string(),
                );
            }
            return Ok(format!("nn-plugin: unknown command '{cmd}'"));
        }

        Ok(format!("nn-plugin event (bytes={})", input.len()))
    }
}
