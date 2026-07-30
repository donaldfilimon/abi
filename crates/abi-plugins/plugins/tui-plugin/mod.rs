//! `tui-plugin` — real implementation.
//!
//! Ported from `src/plugins/tui-plugin/mod.zig`. Every output string is contract:
//! `plugin_run` over MCP and `abi plugin run` both surface it verbatim.

use crate::{Plugin, PluginError, split_command};

/// The enabled `tui-plugin` implementation.
pub struct Mod;

impl Plugin for Mod {
    const NAME: &'static str = "tui-plugin";
    const VERSION: &'static str = "0.1.0";
    const DESCRIPTION: &'static str = "Example reference plugin targeting the feat-tui gate.";
    const TARGET_FEATURE: &'static str = "tui";

    fn run(input: &str) -> Result<String, PluginError> {
        if let Some(rest) = input.strip_prefix("__cmd__:") {
            let (cmd, _payload) = split_command(rest);
            if cmd == "hello" || cmd == "hi" {
                return Ok("tui-plugin: hello".to_string());
            }
            if cmd == "ping" {
                return Ok("tui-plugin: pong".to_string());
            }
            return Ok(format!("tui-plugin: unknown command '{cmd}'"));
        }

        Ok(format!("tui-plugin event (bytes={})", input.len()))
    }
}
