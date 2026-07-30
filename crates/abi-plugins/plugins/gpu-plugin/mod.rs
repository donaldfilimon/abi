//! `gpu-plugin` — real implementation.
//!
//! Ported from `src/plugins/gpu-plugin/mod.zig`. Every output string is contract:
//! `plugin_run` over MCP and `abi plugin run` both surface it verbatim.

use crate::{Plugin, PluginError, split_command};

/// The enabled `gpu-plugin` implementation.
pub struct Mod;

impl Plugin for Mod {
    const NAME: &'static str = "gpu-plugin";
    const VERSION: &'static str = "0.1.0";
    const DESCRIPTION: &'static str = "Example reference plugin targeting the feat-gpu gate.";
    const TARGET_FEATURE: &'static str = "gpu";

    fn run(input: &str) -> Result<String, PluginError> {
        if let Some(rest) = input.strip_prefix("__cmd__:") {
            let (cmd, _payload) = split_command(rest);
            if cmd == "gpu-status" || cmd == "gpu" {
                return Ok(
                    "gpu-plugin: status (reference plugin — see `abi backends` / dashboard System pane)"
                        .to_string(),
                );
            }
            if cmd == "gpu-info" {
                return Ok(
                    "gpu-plugin: info (reference plugin — Metal link vs accelerated is runtime state)"
                        .to_string(),
                );
            }
            return Ok(format!("gpu-plugin: unknown command '{cmd}'"));
        }

        Ok(format!("gpu-plugin event (bytes={})", input.len()))
    }
}
